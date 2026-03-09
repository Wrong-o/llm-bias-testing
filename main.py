"""CLI orchestrator for the LLM research experiment pipeline."""

import argparse
import os
import sys

PIPELINE_STEPS = [
    "run_experiment",
    "score_hard",
    "score_soft",
    "combine",
    "visualize",
]

OUTPUT_FILES = {
    "run_experiment": "results/raw_responses.jsonl",
    "score_hard": "results/hard_scores.csv",
    "score_soft": "results/soft_scores.csv",
    "combine": "results/combined_scores.csv",
    "visualize": "results/radar.png",
}


def _import_step(step_name):
    """Import and return the main() function for a pipeline step."""
    if step_name == "run_experiment":
        from run_experiment import main as step_main
    elif step_name == "score_hard":
        from score_hard import main as step_main
    elif step_name == "score_soft":
        from score_soft import main as step_main
    elif step_name == "combine":
        from combine import main as step_main
    elif step_name == "visualize":
        from visualize import main as step_main
    else:
        raise ValueError(f"Unknown step: {step_name}")
    return step_main


def run_pipeline(steps, force=False):
    """Run the specified pipeline steps in order."""
    for step in steps:
        output_file = OUTPUT_FILES[step]
        if not force and os.path.exists(output_file):
            print(f"[skip] {step} — output exists ({output_file})")
            continue

        print(f"[run] {step}")
        step_main = _import_step(step)
        step_main()
        print(f"[done] {step}")


def main():
    parser = argparse.ArgumentParser(
        description="LLM research experiment pipeline orchestrator."
    )
    parser.add_argument(
        "--step",
        action="append",
        choices=PIPELINE_STEPS,
        help="Run specific step(s). Can be repeated. Omit to run full pipeline.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run steps even if output files already exist.",
    )
    args = parser.parse_args()

    steps = args.step if args.step else PIPELINE_STEPS
    run_pipeline(steps, force=args.force)


if __name__ == "__main__":
    main()
