"""Constrain one greedy Qwen generation to a complete, numbered JSON plan.

Only JSON punctuation and step labels are forced. Action text is selected from
the model's logits. EOS/newline/quote predictions delimit actions; EOS becomes
available to the model only after the final array bracket has been emitted.
"""

from collections import deque

import torch


class FullPlanLogitsProcessor:
    """Stateful decoder for one sequence, one beam, and one generate() call."""

    MAX_BOUNDARY_PREFIX_TOKENS = 8

    def __init__(self, tokenizer, action_count: int, max_new_tokens: int):
        if action_count < 1 or tokenizer.eos_token_id is None:
            raise ValueError("Full-plan decoding needs a positive action count and an EOS token.")
        self.tokenizer = tokenizer
        self.action_count = action_count
        self.eos_token_id = int(tokenizer.eos_token_id)
        self.prefixes = [self._encode('["Step 1: ')] + [
            self._encode(f'", "Step {i}: ') for i in range(2, action_count + 1)
        ]
        self.closing = self._encode('"]')
        overhead = sum(map(len, self.prefixes)) + len(self.closing) + 1
        # Reserve room for every remaining action, separator and final EOS.
        self.max_action_tokens = min(
            128, (max_new_tokens - overhead) // action_count - self.MAX_BOUNDARY_PREFIX_TOKENS
        )
        if self.max_action_tokens < 4:
            raise ValueError(
                f"planner_max_new_tokens={max_new_tokens} is too small for {action_count} actions; "
                "increase it (2048 is sufficient for the 15-action default)."
            )
        vocab_ids = sorted(set(tokenizer.get_vocab().values()))
        decoded = tokenizer.batch_decode(
            [[token_id] for token_id in vocab_ids], skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        specials = set(tokenizer.all_special_ids)
        self.content_ids = []
        self.initial_content_ids = []
        self.boundary_prefixes = {self.eos_token_id: []}
        for token_id, text in zip(vocab_ids, decoded):
            if token_id in specials:
                continue
            if self._safe_content(text):
                self.content_ids.append(token_id)
                if text.strip():
                    self.initial_content_ids.append(token_id)
                continue
            # A BPE token can combine a word/period with a closing quote. Keep
            # that text, then force the correct separator, without losing it.
            boundary = next((i for i, char in enumerate(text) if char in '"\r\n'), None)
            if boundary is not None:
                prefix = text[:boundary]
                if not prefix or self._safe_content(prefix):
                    prefix_ids = self._encode(prefix) if prefix else []
                    if len(prefix_ids) <= self.MAX_BOUNDARY_PREFIX_TOKENS:
                        self.boundary_prefixes[token_id] = prefix_ids
        if not self.initial_content_ids:
            raise ValueError("Tokenizer has no usable action-text tokens.")
        self.pending = deque(self.prefixes[0])
        self.action_number = 1
        self.action_tokens = 0
        self.finished = False
        self.capped_action_numbers = []
        self._previous_length = None
        self._masks = None

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    @staticmethod
    def _safe_content(text: str) -> bool:
        # JSON strings cannot contain literal control characters, quotes or
        # backslashes. Reject incomplete Unicode byte pieces as well.
        return bool(text) and not any(char in '"\\\ufffd' or ord(char) < 32 for char in text)

    def _consume(self, token_id: int) -> None:
        if self.pending:
            expected = self.pending.popleft()
            if token_id != expected:
                raise RuntimeError("Another decoding rule changed the forced full-plan structure.")
        else:
            self.action_tokens += 1

    def _close_action(self, prefix_ids: list[int], *, capped: bool = False) -> None:
        if capped:
            self.capped_action_numbers.append(self.action_number)
        self.pending.extend(prefix_ids)
        if self.action_number == self.action_count:
            self.pending.extend(self.closing)
            self.finished = True
        else:
            self.pending.extend(self.prefixes[self.action_number])
            self.action_number += 1
        self.action_tokens = 0

    @staticmethod
    def _force(scores, token_id: int):
        scores.fill_(-torch.inf)
        scores[0, token_id] = 0
        return scores

    def __call__(self, input_ids, scores):
        if input_ids.shape[0] != 1 or scores.shape[0] != 1:
            raise ValueError("Full-plan decoding supports one sequence and num_beams=1.")
        length = input_ids.shape[1]
        if self._previous_length is not None:
            if length != self._previous_length + 1:
                raise RuntimeError("Full-plan decoding requires ordinary autoregressive generation.")
            self._consume(int(input_ids[0, -1].item()))
        self._previous_length = length
        if self.pending:
            return self._force(scores, self.pending[0])
        if self.finished:
            return self._force(scores, self.eos_token_id)

        candidate = int(scores[0].argmax().item())
        if self.action_tokens and candidate in self.boundary_prefixes:
            self._close_action(self.boundary_prefixes[candidate])
            return self._force(scores, self.pending[0])
        if self.action_tokens >= self.max_action_tokens:
            self._close_action([], capped=True)
            return self._force(scores, self.pending[0])

        if self._masks is None:
            self._masks = []
            for ids in (self.initial_content_ids, self.content_ids):
                mask = torch.ones(scores.shape[-1], dtype=torch.bool, device=scores.device)
                valid_ids = [token_id for token_id in ids if token_id < scores.shape[-1]]
                mask[valid_ids] = False
                self._masks.append(mask)
        return scores.masked_fill(self._masks[int(self.action_tokens > 0)].unsqueeze(0), -torch.inf)

    def metadata(self) -> dict:
        return {
            "mode": "numbered_json_constraints",
            "action_count": self.action_count,
            "max_action_tokens": self.max_action_tokens,
            "capped_action_numbers": list(self.capped_action_numbers),
            "complete": self.finished and not self.pending,
            "no_repeat_ngram_size": 8,
        }
