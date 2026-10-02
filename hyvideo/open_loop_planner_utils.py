"""Full-plan parsing and exact-duration scheduling, without model imports."""

import json
import math
import re

from hyvideo.qwen_planner_utils import TASK_COMPLETED_TOKEN, clean_planner_prediction


OPEN_LOOP_SYSTEM_PROMPT = (
    "You are an action-sequence planner. Given an initial observation image and a task goal, "
    "plan the ENTIRE sequence of actions needed to complete the task, from start to finish. "
    "Output only the requested JSON array of action descriptions."
)


def frame_schedule(duration_seconds: float, fps: int, clip_frames: int, drop_first: bool = True) -> list[int]:
    """Frames retained from each bounded clip; only the final clip is trimmed."""
    if not math.isfinite(duration_seconds) or duration_seconds <= 0 or fps <= 0:
        raise ValueError("target_duration_seconds and output_fps must be positive and finite.")
    exact_frames = duration_seconds * fps
    target_frames = round(exact_frames)
    if target_frames < 1 or not math.isclose(exact_frames, target_frames, abs_tol=1e-8):
        raise ValueError("target_duration_seconds * output_fps must be a positive whole number of frames.")
    if clip_frames < 2 or (clip_frames - 1) % 4:
        raise ValueError("step_video_length must have the form 4k+1 and be at least 5.")
    schedule = [min(target_frames, clip_frames)]
    remaining = target_frames - schedule[0]
    capacity = clip_frames - int(drop_first)
    while remaining:
        keep = min(remaining, capacity)
        schedule.append(keep)
        remaining -= keep
    return schedule


def build_full_plan_prompt(task_goal: str, action_count: int, duration_seconds: float) -> str:
    template = json.dumps([f"Step {i}: <describe the action>" for i in range(1, action_count + 1)])
    return (
        f"Task goal: {task_goal}\n"
        "The image shows the initial state before any action has been executed.\n"
        f"Plan the complete task as exactly {action_count} ordered action segments for a "
        f"{duration_seconds:g}-second video (about {duration_seconds / action_count:.2f} seconds per segment).\n"
        "Cover the whole task, including the final finishing action. Each segment must describe a "
        "concrete visible action as one clear sentence. Do not predict only the next action.\n"
        f"Return ONLY a valid JSON array of exactly {action_count} strings, one action per string, "
        f"numbered Step 1 through Step {action_count}. Replace every placeholder in this template "
        f"with a task-specific action and do not add more entries:\n{template}"
    )


def parse_full_plan(raw: str, processor, expected_actions: int) -> list[str]:
    """Reject partial plans instead of repeating one action to fill the video."""
    text = clean_planner_prediction(raw, processor, collapse_whitespace=False).replace(TASK_COMPLETED_TOKEN, "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        if text.startswith(("[", "{")):
            raise ValueError(
                f"Qwen returned incomplete or invalid JSON ({exc.msg} at character {exc.pos}). "
                "The response is saved in open_loop_plan.json; no missing actions will be invented."
            ) from exc
        # Also accept the numbered format familiar to next-step checkpoints.
        pattern = (r"(?:^|\s)Step\s+(\d+)\s*[.):]\s+" if re.match(r"Step\s+\d+", text, re.IGNORECASE)
                   else r"^\s*(\d+)\s*[.):]\s+")
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE | re.MULTILINE))
        if not matches or text[:matches[0].start()].strip():
            raise ValueError("Qwen did not return a JSON action list or a numbered full action sequence.")
        numbers = [int(match.group(1)) for match in matches]
        if numbers != list(range(1, len(matches) + 1)):
            raise ValueError("Full-plan step numbers must start at 1 and be consecutive.")
        payload = [text[match.end():matches[i+1].start() if i+1 < len(matches) else len(text)].strip()
                   for i, match in enumerate(matches)]
    if isinstance(payload, dict):
        payload = payload.get("actions")
    if not isinstance(payload, list) or not all(isinstance(action, str) and action.strip() for action in payload):
        raise ValueError("The full plan must contain nonempty action strings.")
    if len(payload) != expected_actions:
        raise ValueError(
            f"Qwen returned {len(payload)} actions; the duration schedule requires {expected_actions}. "
            "No video was generated from this incomplete or mismatched plan."
        )
    labels = [re.match(r"^Step\s+(\d+)\s*:\s*", action.strip(), flags=re.IGNORECASE) for action in payload]
    if any(labels) and [int(label.group(1)) if label else None for label in labels] != list(range(1, expected_actions + 1)):
        raise ValueError("Full-plan step numbers must start at 1 and be consecutive, without duplicates.")
    actions = [re.sub(r"^Step\s+\d+\s*:\s*", "", action.strip(), flags=re.IGNORECASE) for action in payload]
    if not all(actions):
        raise ValueError("Every planned step must contain an action, not only a step label.")
    return actions
