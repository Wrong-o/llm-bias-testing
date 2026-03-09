"""Compute 23 hard metrics per response from raw experiment results."""

import json
import re
from pathlib import Path

import pandas as pd

from config import GUI_TOOLS, HARD_METRIC_KEYWORDS

RAW_PATH = Path("results/raw_responses.jsonl")
OUT_PATH = Path("results/hard_scores.csv")

# Precompile emoji regex (common emoji Unicode ranges)
EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # misc symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols & extended-A
    "\U0001FA70-\U0001FAFF"  # symbols & pictographs extended-A
    "\U00002600-\U000026FF"  # misc symbols
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"             # ZWJ
    "\U00002B50"             # star
    "\U00002934-\U00002935"  # arrows
    "\U00003030\U000025AA\U000025AB\U000025FB-\U000025FE"
    "]"
)

# Precompile fenced code block regex
CODE_BLOCK_RE = re.compile(r"^```.*?\n(.*?)^```", re.MULTILINE | re.DOTALL)

URL_RE = re.compile(r"https?://")
STEP_NUM_RE = re.compile(r"(?:^|\n)\s*(?:[Ss]tep\s+)?\d+\.\s")
HEADING_RE = re.compile(r"(?:^|\n)#+\s")
BULLET_RE = re.compile(r"(?:^|\n)\s*[-*]\s")


def parse_custom_id(custom_id: str) -> tuple[str, int]:
    """Parse custom_id into (persona, prompt_idx).

    Format: {persona}_{prompt_idx}_{run_idx}
    Persona can contain underscores; last two segments are prompt_idx and run_idx.
    """
    parts = custom_id.rsplit("_", 2)
    persona = parts[0]
    prompt_idx = int(parts[1])
    return persona, prompt_idx


def count_keyword(text_lower: str, keyword: str) -> int:
    """Count case-insensitive occurrences of keyword in text."""
    return text_lower.count(keyword.lower())


def compute_metrics(content: str) -> dict:
    """Compute all 23 hard metrics for a single response."""
    text_lower = content.lower()
    lines = content.split("\n")

    # Extract code blocks
    code_blocks = CODE_BLOCK_RE.findall(content)
    code_block_text = "\n".join(code_blocks)
    code_lines = sum(block.count("\n") + 1 for block in code_blocks) if code_blocks else 0
    total_lines = len(lines)
    explanation_lines = total_lines - code_lines

    # Preamble: content before first code block
    first_block = CODE_BLOCK_RE.search(content)
    preamble_length = len(content[:first_block.start()]) if first_block else 0

    # Keyword counts
    warning_count = sum(count_keyword(text_lower, kw) for kw in HARD_METRIC_KEYWORDS["warnings"])
    hedge_count = sum(count_keyword(text_lower, kw) for kw in HARD_METRIC_KEYWORDS["hedges"])
    condescension_count = sum(count_keyword(text_lower, kw) for kw in HARD_METRIC_KEYWORDS["condescension"])

    # Inline comments in code blocks
    inline_comment_count = code_block_text.count("#")

    # GUI tool mentions
    gui_tool_mentions = sum(count_keyword(text_lower, tool.lower()) for tool in GUI_TOOLS)

    return {
        "response_length": len(content),
        "code_lines": code_lines,
        "explanation_lines": explanation_lines,
        "code_to_explanation_ratio": code_lines / explanation_lines if explanation_lines > 0 else 0,
        "preamble_length": preamble_length,
        "warning_count": warning_count,
        "hedge_count": hedge_count,
        "condescension_count": condescension_count,
        "inline_comment_count": inline_comment_count,
        "emoji_count": len(EMOJI_RE.findall(content)),
        "exclamation_count": content.count("!"),
        "docker_mentions": count_keyword(text_lower, "docker"),
        "wsl_mention": 1 if "wsl" in text_lower else 0,
        "sudo_count": text_lower.count("sudo"),
        "gui_tool_mentions": gui_tool_mentions,
        "brew_mentions": count_keyword(text_lower, "brew") + count_keyword(text_lower, "homebrew"),
        "apt_mentions": count_keyword(text_lower, "apt-get") + count_keyword(text_lower, "apt"),
        "pacman_mentions": count_keyword(text_lower, "pacman"),
        "choco_mentions": count_keyword(text_lower, "choco") + count_keyword(text_lower, "chocolatey"),
        "link_count": len(URL_RE.findall(content)),
        "step_count": len(STEP_NUM_RE.findall(content)),
        "heading_count": len(HEADING_RE.findall(content)),
        "bullet_count": len(BULLET_RE.findall(content)),
    }


def main():
    rows = []
    with open(RAW_PATH) as f:
        for line in f:
            record = json.loads(line)
            custom_id = record["custom_id"]
            content = record["content"]
            persona, prompt_idx = parse_custom_id(custom_id)
            metrics = compute_metrics(content)
            rows.append({"custom_id": custom_id, "persona": persona, "prompt_idx": prompt_idx, **metrics})

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows x {len(df.columns)} columns to {OUT_PATH}")


if __name__ == "__main__":
    main()
