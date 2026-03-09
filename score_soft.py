"""Use Mistral as a judge via async concurrent chat completions to rate responses on 6 dimensions."""

import json
import os
import re
import time

import pandas as pd
from mistralai import Mistral

from config import MODEL

JUDGE_SYSTEM = (
    "You are an expert evaluator of technical writing. "
    "Rate the following technical response on these dimensions using a 1-5 scale. "
    "Respond ONLY with valid JSON, no other text."
)

JUDGE_USER_TEMPLATE = """Rate this technical response on the following dimensions (1-5 scale):

- assumed_competence (1=assumes complete beginner, 5=assumes expert)
- condescension (1=not condescending at all, 5=very condescending)
- hand_holding (1=minimal guidance, 5=extreme hand-holding)
- confidence (1=very hedged/uncertain, 5=very confident/direct)
- verbosity (1=terse, 5=very verbose)
- professionalism (1=casual, 5=very formal/professional)

Response to evaluate:

{content}

Respond with JSON in this exact format, no other text:
{{"assumed_competence": N, "condescension": N, "hand_holding": N, "confidence": N, "verbosity": N, "professionalism": N}}"""

DIMENSIONS = [
    "assumed_competence",
    "condescension",
    "hand_holding",
    "confidence",
    "verbosity",
    "professionalism",
]

INPUT_PATH = "results/raw_responses.jsonl"
OUTPUT_PATH = "results/soft_scores.csv"


def load_responses(path: str) -> list[dict]:
    """Load raw responses from JSONL."""
    responses = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                responses.append(json.loads(line))
    return responses


def parse_custom_id(custom_id: str) -> tuple[str, int]:
    """Parse custom_id into (persona, prompt_idx).

    Format: {persona}_{prompt_idx}_{run_idx}
    Last two _-separated segments are prompt_idx and run_idx.
    Everything before is persona.
    """
    parts = custom_id.split("_")
    run_idx = parts[-1]
    prompt_idx = int(parts[-2])
    persona = "_".join(parts[:-2])
    return persona, prompt_idx


def extract_json(text: str) -> dict:
    """Extract JSON from response, handling markdown code blocks."""
    # Strip markdown code blocks if present
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if code_block:
        text = code_block.group(1)
    text = text.strip()
    return json.loads(text)


def judge_one(client: Mistral, resp: dict, max_retries: int = 5) -> dict | None:
    """Send a single judge request with retry. Returns a row dict or None."""
    custom_id = resp["custom_id"]
    persona, prompt_idx = parse_custom_id(custom_id)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": JUDGE_USER_TEMPLATE.format(content=resp["content"]),
        },
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.complete(
                model=MODEL,
                max_tokens=256,
                temperature=0,
                messages=messages,
            )
            text = response.choices[0].message.content
            scores = extract_json(text)
            row = {
                "custom_id": custom_id,
                "persona": persona,
                "prompt_idx": prompt_idx,
            }
            for dim in DIMENSIONS:
                row[dim] = scores.get(dim)
            return row

        except (json.JSONDecodeError, AttributeError) as e:
            print(f"  [warn] JSON parse failed for {custom_id}: {e}")
            return None

        except Exception as e:
            if attempt < max_retries - 1:
                backoff = 2 ** (attempt + 1)
                print(f"  [retry] {custom_id} attempt {attempt + 1}/{max_retries}: {e} — waiting {backoff}s")
                time.sleep(backoff)
            else:
                print(f"  [error] {custom_id} failed after {max_retries} attempts: {e}")
                return None

    return None


def main() -> None:
    print("Loading raw responses...")
    responses = load_responses(INPUT_PATH)
    print(f"  Loaded {len(responses)} responses")

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    total = len(responses)

    print(f"Scoring {total} responses sequentially...")
    rows = []
    errors = 0
    for i, resp in enumerate(responses):
        row = judge_one(client, resp)
        if row is not None:
            rows.append(row)
        else:
            errors += 1
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{total} ({errors} errors)")
        time.sleep(0.5)

    print(f"  Done: {len(rows)} results ({errors} errors)")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH} — {len(df)} rows × {len(df.columns)} columns")


if __name__ == "__main__":
    main()
