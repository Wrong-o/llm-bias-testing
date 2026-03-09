---
date: 2026-03-09
researcher: GIVERNY
git_commit: (no commits yet)
branch: master
topic: "LLM OS Persona Bias Test"
status: complete
---

# Research: LLM OS Persona Bias Test

## Summary

Greenfield project. The repo is a bare `uv`-initialized Python 3.12 project with no dependencies and a stub `main.py`. No experiment code, no results, no dependencies installed. Everything needs to be built from scratch per the spec.

## File Locations

### Existing files
- `main.py:1` - Stub hello-world, will be replaced
- `pyproject.toml:1` - Bare project config, no dependencies
- `.python-version` - Python 3.12
- `.gitignore` - Standard Python ignores

### To be created (per spec)
```
config.py          # PERSONAS, PROMPTS, model settings
run_experiment.py  # Batch API runner
score_hard.py      # String-counting metrics
score_soft.py      # LLM-as-judge via Batch API
combine.py         # Merge hard + soft into final CSV
visualize.py       # All graphs (radar, bars, stacked, composite)
results/           # Output directory
```

## Dependencies Required

```toml
dependencies = [
    "anthropic",      # Batch API SDK
    "pandas",         # Data manipulation
    "matplotlib",     # Plotting
    "seaborn",        # Statistical visualization
]
```

## API Details

- **Batch API method:** `client.messages.batches.create(requests=[...])`
- Each request: `{"custom_id": str, "params": {model, max_tokens, temperature, messages}}`
- Temperature param supported per-request ✓
- Model: `claude-sonnet-4-20250514`

## Scale

- 5 personas × 8 prompts × 20 runs = **800 API calls** (main experiment)
- 800 calls for soft scoring (LLM-as-judge) = **1600 total batch requests**
- Estimated cost: ~$2-5

## Key Design Decisions from Spec

1. **Temperature 1.0** — maximize behavioral variance
2. **max_tokens 4096** — allow full responses
3. **Minimal prompt template** — `{persona}\n\n{prompt}`, no system prompt specified
4. **JSONL for raw storage** — re-scorable without re-running
5. **Hard metrics are free** — pure string counting, run locally
6. **Soft metrics via separate batch** — LLM-as-judge for subjective dimensions

## Metrics Summary

### Hard (22 metrics)
response_length, code_lines, explanation_lines, code_to_explanation_ratio,
preamble_length, warning_count, hedge_count, condescension_count,
inline_comment_count, emoji_count, exclamation_count,
docker_mentions, wsl_mention, sudo_count, gui_tool_mentions,
brew_mentions, apt_mentions, pacman_mentions,
link_count, step_count

### Soft (6 dimensions, LLM-judged)
assumed_competence, condescension, hand_holding, confidence, verbosity, professionalism

## Graphs (5 total)
1. Radar chart — hero image, overlaid polygons per persona
2. Grouped bar charts — key metrics with error bars
3. Package manager bias — stacked bars
4. Condescension Index — composite score bar chart ("money shot")
5. WSL mention rate — simple bar chart

## Decisions (confirmed with DEV)

- **System prompt:** None. Expose raw parametric bias without system prompt influence.
- **choco_mentions:** Added to hard metrics (Chocolatey = Windows package manager). Total hard metrics now 23.
- **GUI tool list:** To be defined in plan phase. Candidates: VS Code, Notepad++, Finder, File Explorer, Terminal.app, PowerShell ISE, GNOME Files, Nautilus, Dolphin, System Preferences, Control Panel, Settings app.

## Open Questions

- **Batch polling:** Need to implement polling/waiting for batch completion. SDK likely has `client.messages.batches.retrieve(batch_id)` or similar.
