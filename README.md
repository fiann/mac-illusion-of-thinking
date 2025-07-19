# Replicate the Tower of Hanoi Experiment from the paper "The Illusion of Thinking"

Replicate Tower of Hanoi experiments with Claude API to analyze LLM performance patterns. The original paper is [The Illusion of Thinking](https://ml-site.cdn-apple.com/papers/the-illusion-of-thinking.pdf).

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
  --model claude-3-7-sonnet-latest \
  --output-dir ./results \
  --include-hint  # Include hint about expected number of moves
```

### Command Line Options

- `--disks`: Number of disks to use (default: 7)
- `--repetitions`: Number of times to run the experiment (default: 10)
- `--temperature`: LLM temperature setting (default: 0.0)
- `--model`: Claude model to use (default: claude-3-7-sonnet-latest)
- `--output-dir`: Directory for results (default: ./results)
- `--include-hint`: Include a hint about the expected number of moves in the prompt (default: off)

When `--include-hint` is enabled, the prompt will include "Note: The solution requires N moves" where N is 2^disks - 1.

## Output

Results are saved in structured format:

- `response.txt` - Full Claude API response
- `moves.json` - Extracted move sequences (compatible with HTML visualizer)
- `validation.json` - Move validation results
- `summary.json` - Aggregate statistics

## Visualization

Load the `moves.json` files into `html/tower-of-hanoi-html` to visualize the solutions.
