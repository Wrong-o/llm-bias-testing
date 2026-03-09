"""Run all persona x prompt x run combinations via async concurrent Mistral chat completions."""

import asyncio
import json
import os

from mistralai import Mistral

from config import MODEL, MAX_TOKENS, TEMPERATURE, PERSONAS, PROMPTS, RUNS_PER_COMBO

CONCURRENCY_LIMIT = 5
MAX_RETRIES = 3


def build_requests() -> list[dict]:
    """Build the list of request specs with custom_id and messages."""
    requests = []
    for persona_key, system_text in PERSONAS.items():
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


async def call_with_retry(
    client: Mistral,
    spec: dict,
    semaphore: asyncio.Semaphore,
    progress: dict,
) -> dict | None:
    """Call chat.complete_async with semaphore gating and retry logic."""
    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await client.chat.complete_async(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    messages=spec["messages"],
                )
                content = response.choices[0].message.content
                progress["done"] += 1
                if progress["done"] % 50 == 0:
                    print(f"[progress] {progress['done']}/{progress['total']} completions")
                return {"custom_id": spec["custom_id"], "content": content}
            except Exception as e:
                if attempt < MAX_RETRIES:
                    error_str = str(e)
                    is_rate_limit = "429" in error_str or "rate" in error_str.lower() or "RBAC" in error_str
                    wait = (2 ** attempt) * (5 if is_rate_limit else 1)
                    print(f"[retry] {spec['custom_id']} attempt {attempt}/{MAX_RETRIES} "
                          f"failed: {e} — retrying in {wait}s")
                    await asyncio.sleep(wait)
                else:
                    print(f"[error] {spec['custom_id']} failed after {MAX_RETRIES} attempts: {e} — skipping")
                    progress["done"] += 1
                    if progress["done"] % 50 == 0:
                        print(f"[progress] {progress['done']}/{progress['total']} completions")
                    return None


async def async_main() -> None:
    os.makedirs("results", exist_ok=True)
    output_path = "results/raw_responses.jsonl"

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    requests = build_requests()
    print(f"[init] Built {len(requests)} requests")

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    progress = {"done": 0, "total": len(requests)}

    tasks = [call_with_retry(client, spec, semaphore, progress) for spec in requests]
    results = await asyncio.gather(*tasks)

    count = 0
    with open(output_path, "w") as f:
        for result in results:
            if result is not None:
                f.write(json.dumps(result) + "\n")
                count += 1

    print(f"[results] Wrote {count} lines to {output_path}")
    if count < len(requests):
        print(f"[warn] {len(requests) - count} requests failed and were skipped")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
