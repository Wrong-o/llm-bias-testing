"""Run persona x prompt x run combinations via Mistral chat completions."""

import argparse
import json
import os
import time

from mistralai import Mistral

from config import MODEL, MAX_TOKENS, TEMPERATURE, PERSONAS, PROMPTS, RUNS_PER_COMBO

MAX_RETRIES = 5
DELAY = 0.5


def build_requests(personas: dict | None = None) -> list[dict]:
    """Build request specs. If personas is None, use all from config."""
    if personas is None:
        personas = PERSONAS
    requests = []
    for persona_key, system_text in personas.items():
        for prompt_idx, prompt in enumerate(PROMPTS):
            for run_idx in range(RUNS_PER_COMBO):
                custom_id = f"{persona_key}_{prompt_idx}_{run_idx}"
                messages = []
                if system_text:
                    messages.append({"role": "system", "content": system_text})
                messages.append({"role": "user", "content": prompt})
                requests.append({
                    "custom_id": custom_id,
                    "messages": messages,
                })
    return requests


def call_with_retry(client: Mistral, spec: dict) -> dict | None:
    """Call chat.complete with retry logic. Returns result dict or None."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.complete(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=spec["messages"],
            )
            content = response.choices[0].message.content
            return {"custom_id": spec["custom_id"], "content": content}
        except Exception as e:
            if attempt < MAX_RETRIES:
                backoff = 2 ** attempt
                print(f"[retry] {spec['custom_id']} attempt {attempt}/{MAX_RETRIES}: {e} — waiting {backoff}s")
                time.sleep(backoff)
            else:
                print(f"[error] {spec['custom_id']} failed after {MAX_RETRIES} attempts: {e}")
                return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", type=str, help="Run only this persona (must be a key in PERSONAS)")
    args, _ = parser.parse_known_args()

    os.makedirs("results", exist_ok=True)
    output_path = "results/raw_responses.jsonl"

    # Determine which personas to run
    if args.persona:
        if args.persona not in PERSONAS:
            print(f"[error] Unknown persona: {args.persona}. Available: {list(PERSONAS.keys())}")
            return
        personas = {args.persona: PERSONAS[args.persona]}
    else:
        personas = None  # all

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    requests = build_requests(personas)
    print(f"[init] Built {len(requests)} requests")

    # If filtering by persona, load existing data and strip old entries for that persona
    existing_lines = []
    if args.persona and os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                # Keep lines that don't belong to the persona we're replacing
                parts = entry["custom_id"].split("_")
                entry_persona = "_".join(parts[:-2])
                if entry_persona != args.persona:
                    existing_lines.append(line)
        print(f"[merge] Keeping {len(existing_lines)} existing lines, replacing {args.persona}")

    # Run requests sequentially
    new_lines = []
    errors = 0
    start_time = time.time()
    for i, spec in enumerate(requests):
        result = call_with_retry(client, spec)
        if result is not None:
            new_lines.append(json.dumps(result))
            print(f"[{i + 1}/{len(requests)}] ✓ {spec['custom_id']} ({len(result['content'])} chars)")
        else:
            errors += 1
            print(f"[{i + 1}/{len(requests)}] ✗ {spec['custom_id']} FAILED")
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(requests) - i - 1) / rate
            print(f"[progress] {i + 1}/{len(requests)} | {errors} errors | "
                  f"{elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining")
        time.sleep(DELAY)

    # Write output
    all_lines = existing_lines + new_lines
    with open(output_path, "w") as f:
        for line in all_lines:
            f.write(line + "\n")

    print(f"[results] Wrote {len(all_lines)} total lines to {output_path} ({len(new_lines)} new, {errors} errors)")


if __name__ == "__main__":
    main()
