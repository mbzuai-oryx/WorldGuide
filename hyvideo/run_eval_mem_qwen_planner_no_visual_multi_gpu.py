"""Closed-loop memory evaluation with action-history-only Qwen planning.

At step 1 Qwen receives the task goal alone. Later it receives the goal and
the last K predicted/executed actions (default 3). No observation images or
videos are sent to Qwen at any step. Hunyuan still uses the usual generated
reference frames and mode-2 memory, as in run_eval_mem_qwen_planner_multi_gpu.py.

The existing strict memory preset is retained: 33 frames per step, model_type=bi,
chunk_latent_frames=9. GPU selection uses only HIP_VISIBLE_DEVICES. Sharding,
resume, stop conditions, and per-sample prediction summaries are unchanged.
See hyvideo/QWEN_PLANNER_EVAL.md for a complete command.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hyvideo import run_eval_mem_qwen_planner_multi_gpu as memory_planner
from hyvideo import run_step_prompts_v5_qwen_planner_multi_gpu as planner_multi_gpu


def _with_no_visual_memory_preset(argv: list[str]) -> list[str]:
    if any(token in ("-h", "--help") for token in argv):
        return argv
    supplied = {token.split("=", 1)[0] for token in argv if token.startswith("--")}
    if "--planner_visual_feedback" in supplied:
        raise ValueError(
            "run_eval_mem_qwen_planner_no_visual_multi_gpu.py fixes "
            "planner_visual_feedback=false; remove --planner_visual_feedback."
        )
    if "--output_prefix" not in supplied:
        argv = [*argv, "--output_prefix", "mem_planner_no_visual_"]
    return memory_planner._with_planner_memory_preset(
        [*argv, "--planner_visual_feedback", "false"]
    )


def main() -> None:
    planner_multi_gpu.main(_with_no_visual_memory_preset(sys.argv[1:]))


if __name__ == "__main__":
    main()
