# Tower of Hanoi Experiment

Replicate Tower of Hanoi experiments with Claude API to analyze LLM performance patterns.

## Setup

1. Install dependencies with `uv`:
   ```bash
   uv sync
   ```

2. Copy `.env.example` to `.env` and add your Anthropic API key:
   ```bash
   cp .env.example .env
   ```

## Usage

Run experiments with configurable parameters:

```bash
# Default: 7 disks, 10 repetitions, temperature 0.0
uv run python tower_of_hanoi_experiment.py

# Custom parameters
uv run python tower_of_hanoi_experiment.py --disks 5 --repetitions 20 --temperature 0.5

# Full options
uv run python tower_of_hanoi_experiment.py \
  --disks 7 \
  --repetitions 10 \
  --temperature 0.0 \
  --model claude-3-5-sonnet-20241022 \
  --output-dir ./results
```

## Output

Results are saved in structured format:
- `response.txt` - Full Claude API response
- `moves.json` - Extracted move sequences (compatible with HTML visualizer)
- `validation.json` - Move validation results
- `summary.json` - Aggregate statistics

## Visualization

Load the `moves.json` files into `html/tower-of-hanoi-html` to visualize the solutions.