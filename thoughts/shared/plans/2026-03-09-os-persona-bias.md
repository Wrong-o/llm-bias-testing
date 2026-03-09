---
date: 2026-03-09
planner: GIVERNY
research_doc: thoughts/shared/research/2026-03-09-os-persona-bias.md
status: approved
iteration: 2
---

# Plan: LLM OS Persona Bias Experiment

## Objective
Prove (or disprove) that Claude changes its technical advice style based on perceived OS persona, with publishable visuals for LinkedIn.

## Research Reference
Based on: `thoughts/shared/research/2026-03-09-os-persona-bias.md`

---

## Phase 1: Foundation — Config + Dependencies

### Sandbox
Files to modify:
- `config.py` (create)
- `pyproject.toml` (via `uv add`)

### What to build
- `config.py` containing:
  - `PERSONAS` dict — 5 personas with system-context strings (macOS dev, Windows dev, Linux sysadmin, Linux beginner, unspecified)
  - `PROMPTS` list — 8 technical prompts that could surface OS bias (e.g., "install Python", "set up a web server", "manage packages", "edit a config file", "set up Docker", "debug a network issue", "automate a backup", "set up SSH keys")
  - `MODEL` = `claude-sonnet-4-20250514`
  - `TEMPERATURE` = 1.0
  - `MAX_TOKENS` = 4096
  - `RUNS_PER_COMBO` = 20
  - `GUI_TOOLS` list for metric detection
  - `HARD_METRIC_KEYWORDS` — all keyword lists for hard scoring
- Dependencies installed via `uv add anthropic pandas matplotlib seaborn`

### GUI Tool List (LinkedIn-ready — broad enough to catch real bias)
```
VS Code, Notepad++, Sublime Text, Finder, File Explorer,
Terminal.app, PowerShell ISE, GNOME Files, Nautilus, Dolphin,
System Preferences, Control Panel, Settings, TextEdit, gedit, nano, vim
```

### Success Criteria
- [ ] `uv run python -c "import config; print(len(config.PERSONAS), len(config.PROMPTS))"` prints `5 8`
- [ ] `uv run python -c "import anthropic, pandas, matplotlib, seaborn"` exits 0

### Verification
Automated:
- [ ] Import check passes

---

## Phase 2: Experiment Runner

### Sandbox
Files to modify:
- `run_experiment.py` (create)

Files to read (reference only):
- `config.py`

### What to build
- Builds 800 batch requests (5×8×20) with `custom_id` = `{persona}_{prompt_idx}_{run_idx}`
- Submits via `client.messages.batches.create()`
- Polls for completion with backoff
- Writes raw results to `results/raw_responses.jsonl`
- Each JSONL line: `{"custom_id": ..., "response": ..., "content": ...}`

### Success Criteria
- [ ] Script runs end-to-end: `uv run python run_experiment.py`
- [ ] `results/raw_responses.jsonl` contains 800 lines
- [ ] Each line has `custom_id`, parseable content text

---

## Phase 3a: Hard Scoring (parallel with 3b)

### Sandbox
Files to modify:
- `score_hard.py` (create)

Files to read (reference only):
- `config.py`
- `results/raw_responses.jsonl`

### What to build
- Reads `raw_responses.jsonl`
- Computes 23 hard metrics per response:
  - `response_length` (char count)
  - `code_lines`, `explanation_lines`, `code_to_explanation_ratio` (fenced code block detection)
  - `preamble_length` (chars before first code block)
  - `warning_count` ("warning", "caution", "be careful", "note:")
  - `hedge_count` ("might", "could", "perhaps", "usually", "typically")
  - `condescension_count` ("simply", "just", "easy", "obvious", "straightforward", "of course")
  - `inline_comment_count` (`#` inside code blocks)
  - `emoji_count`, `exclamation_count`
  - `docker_mentions`, `wsl_mention`, `sudo_count`
  - `gui_tool_mentions` (count from GUI_TOOLS list)
  - `brew_mentions`, `apt_mentions`, `pacman_mentions`, `choco_mentions`
  - `link_count` (URLs)
  - `step_count` ("step 1", numbered lists)
- Outputs `results/hard_scores.csv`

### Success Criteria
- [ ] `uv run python score_hard.py` produces `results/hard_scores.csv`
- [ ] CSV has 800 rows × (23 metric columns + custom_id + persona + prompt_idx)

---

## Phase 3b: Soft Scoring (parallel with 3a)

### Sandbox
Files to modify:
- `score_soft.py` (create)

Files to read (reference only):
- `config.py`
- `results/raw_responses.jsonl`

### What to build
- Reads `raw_responses.jsonl`
- Builds 800 judge requests — each asks Claude to rate the response on 6 dimensions (1-5 scale):
  - `assumed_competence`, `condescension`, `hand_holding`, `confidence`, `verbosity`, `professionalism`
- Submits as batch, polls for completion
- Parses judge responses, outputs `results/soft_scores.csv`

### Success Criteria
- [ ] `uv run python score_soft.py` produces `results/soft_scores.csv`
- [ ] CSV has 800 rows × (6 score columns + custom_id + persona + prompt_idx)

---

## Phase 4: Combine

### Sandbox
Files to modify:
- `combine.py` (create)

Files to read (reference only):
- `results/hard_scores.csv`
- `results/soft_scores.csv`

### What to build
- Merges hard + soft on `custom_id`
- Adds derived columns (e.g., condescension_index = composite of condescension + hedge + warning + hand_holding)
- Outputs `results/combined_scores.csv`

### Success Criteria
- [ ] `uv run python combine.py` produces `results/combined_scores.csv`
- [ ] CSV has 800 rows × (23 + 6 + derived columns)

---

## Phase 5: Visualize

### Sandbox
Files to modify:
- `visualize.py` (create)

Files to read (reference only):
- `config.py`
- `results/combined_scores.csv`

### What to build
5 publication-quality graphs saved to `results/`:

1. **Radar chart** (`results/radar.png`) — overlaid polygons per persona, normalized soft scores
2. **Grouped bar charts** (`results/bars_key_metrics.png`) — top 6 most-divergent hard metrics with error bars (95% CI)
3. **Package manager bias** (`results/bars_package_mgr.png`) — stacked bars: brew/apt/pacman/choco per persona
4. **Condescension Index** (`results/condescension_index.png`) — composite score per persona, sorted, bold colors — the "money shot"
5. **WSL mention rate** (`results/wsl_rate.png`) — simple bar chart per persona

Style: dark background, vibrant colors, large fonts, LinkedIn-optimized (1200×628 or similar aspect ratio)

### Success Criteria
- [ ] `uv run python visualize.py` produces 5 PNG files in `results/`
- [ ] Each PNG is >10KB (not empty)
- [ ] Visuals use consistent color palette across all 5 charts

---

## Phase 6: Orchestrator + main.py

### Sandbox
Files to modify:
- `main.py` (rewrite)

### What to build
- CLI entry point that runs the full pipeline: run → score_hard → score_soft → combine → visualize
- Or individual steps via args (e.g., `uv run python main.py --step visualize`)
- Skip steps if output already exists (unless `--force`)

### Success Criteria
- [ ] `uv run python main.py` runs full pipeline
- [ ] `uv run python main.py --step visualize` runs only visualization

---

## Execution Strategy

```
Phase 1 (foundation)
  ↓
Phase 2 (runner)
  ↓
Phase 3a + 3b (parallel — hard & soft scoring)
  ↓
Phase 4 (combine)
  ↓
Phase 5 (visualize)
  ↓
Phase 6 (orchestrator)
```

## Rollback Plan
1. All outputs go to `results/` — delete directory to reset
2. Each script is standalone — rerun individually
3. No database, no side effects beyond files

## Resolved Questions
- **Prompt wording:** Delegated to coder subagent — trust their judgment
- **Condescension Index formula:** Simple average of (condescension_count + hedge_count + warning_count + hand_holding_score) normalized per-metric before averaging
- **Soft scoring rubric:** Delegated to coder subagent — must use 1-5 scale with anchored descriptions for each dimension
