#!/usr/bin/env python3
import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import click
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

load_dotenv()


class TowerOfHanoiExperiment:
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.prompt_template = self._load_prompt_template()
    
    def _load_prompt_template(self) -> str:
        prompt_path = Path("tower-of-hanoi-prompt.md")
        if prompt_path.exists():
            return prompt_path.read_text()
        else:
            return """You are a helpful assistant. Solve this puzzle for me.

There are three pegs and n disks of different sizes stacked on the first peg. The disks are
numbered from 1 (smallest) to n (largest). Disk moves in this puzzle should follow:

1. Only one disk can be moved at a time.
2. Each move consists of taking the upper disk from one stack and placing it on top of
   another stack.
3. A larger disk may not be placed on top of a smaller disk.

The goal is to move the entire stack to the third peg.

Example: With 3 disks numbered 1 (smallest), 2, and 3 (largest), the initial state is [[3, 2, 1],
[], []], and a solution might be:

```
moves = [[1 , 0 , 2] , [2 , 0 , 1] , [1 , 2 , 1] , [3 , 0 , 2] ,
[1 , 1 , 0] , [2 , 1 , 2] , [1 , 0 , 2]]
```

This means: Move disk 1 from peg 0 to peg 2, then move disk 2 from peg 0 to peg 1, and so on.
Requirements:

- When exploring potential solutions in your thinking process, always include the corre-
  sponding complete list of moves.
- The positions are 0-indexed (the leftmost peg is 0).
- Ensure your final answer includes the complete list of moves in the format:
  moves = [[disk id, from peg, to peg], ...]
"""
    
    def _build_prompt(self, num_disks: int) -> str:
        prompt = self.prompt_template
        initial_state = [list(range(num_disks, 0, -1)), [], []]
        prompt += f"\n\nNow solve for {num_disks} disks. The initial state is {initial_state}."
        return prompt
    
    async def run_single_experiment(self, num_disks: int, temperature: float) -> Dict:
        prompt = self._build_prompt(num_disks)
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text if response.content else ""
            
            return {
                "success": True,
                "response": content,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": ""
            }
    
    def extract_moves(self, response: str) -> Optional[List[List[int]]]:
        patterns = [
            r'moves\s*=\s*\[\s*((?:\[[^\]]+\]\s*,?\s*)+)\]',
            r'```\s*moves\s*=\s*\[\s*((?:\[[^\]]+\]\s*,?\s*)+)\]\s*```',
            r'Final answer:\s*moves\s*=\s*\[\s*((?:\[[^\]]+\]\s*,?\s*)+)\]'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                try:
                    moves_str = f"[{match.group(1)}]"
                    moves_str = re.sub(r'\s+', ' ', moves_str)
                    moves = eval(moves_str)
                    return moves
                except:
                    continue
        
        return None
    
    def validate_moves(self, moves: List[List[int]], num_disks: int) -> Dict:
        pegs = [list(range(num_disks, 0, -1)), [], []]
        errors = []
        
        for i, move in enumerate(moves):
            if len(move) != 3:
                errors.append(f"Move {i+1}: Invalid format {move}")
                continue
            
            disk, from_peg, to_peg = move
            
            if from_peg not in [0, 1, 2] or to_peg not in [0, 1, 2]:
                errors.append(f"Move {i+1}: Invalid peg number")
                continue
            
            if not pegs[from_peg]:
                errors.append(f"Move {i+1}: Source peg {from_peg} is empty")
                continue
            
            if pegs[from_peg][-1] != disk:
                errors.append(f"Move {i+1}: Disk {disk} is not on top of peg {from_peg}")
                continue
            
            if pegs[to_peg] and pegs[to_peg][-1] < disk:
                errors.append(f"Move {i+1}: Cannot place disk {disk} on smaller disk {pegs[to_peg][-1]}")
                continue
            
            pegs[from_peg].pop()
            pegs[to_peg].append(disk)
        
        is_solved = pegs[2] == list(range(num_disks, 0, -1))
        
        return {
            "valid": len(errors) == 0,
            "solved": is_solved,
            "errors": errors,
            "final_state": pegs,
            "num_moves": len(moves)
        }


async def run_experiment(
    num_disks: int,
    repetitions: int,
    temperature: float,
    model: str,
    output_dir: Path
):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
    
    experiment = TowerOfHanoiExperiment(api_key, model)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = output_dir / f"experiment_{num_disks}_disks_temp_{temperature}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "num_disks": num_disks,
        "repetitions": repetitions,
        "temperature": temperature,
        "model": model,
        "timestamp": timestamp
    }
    
    with open(exp_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    results = []
    
    async def run_single(run_id: int):
        run_dir = exp_dir / f"run_{run_id}"
        run_dir.mkdir(exist_ok=True)
        
        result = await experiment.run_single_experiment(num_disks, temperature)
        
        with open(run_dir / "response.txt", "w") as f:
            f.write(result["response"])
        
        if result["success"]:
            moves = experiment.extract_moves(result["response"])
            
            if moves:
                with open(run_dir / "moves.json", "w") as f:
                    json.dump(moves, f)
                
                validation = experiment.validate_moves(moves, num_disks)
                with open(run_dir / "validation.json", "w") as f:
                    json.dump(validation, f, indent=2)
                
                result["moves"] = moves
                result["validation"] = validation
            else:
                result["moves"] = None
                result["validation"] = {"valid": False, "errors": ["Could not extract moves"]}
        
        return run_id, result
    
    tasks = [run_single(i) for i in range(1, repetitions + 1)]
    
    for task in tqdm.as_completed(tasks, desc="Running experiments"):
        run_id, result = await task
        results.append((run_id, result))
    
    results.sort(key=lambda x: x[0])
    
    summary = {
        "total_runs": repetitions,
        "successful_api_calls": sum(1 for _, r in results if r["success"]),
        "moves_extracted": sum(1 for _, r in results if r.get("moves") is not None),
        "valid_solutions": sum(1 for _, r in results if r.get("validation", {}).get("valid", False)),
        "puzzles_solved": sum(1 for _, r in results if r.get("validation", {}).get("solved", False)),
        "average_moves": sum(len(r.get("moves", [])) for _, r in results if r.get("moves")) / max(1, sum(1 for _, r in results if r.get("moves"))),
        "optimal_moves": 2**num_disks - 1
    }
    
    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nExperiment completed!")
    print(f"Results saved to: {exp_dir}")
    print(f"\nSummary:")
    print(f"  - Successful API calls: {summary['successful_api_calls']}/{summary['total_runs']}")
    print(f"  - Moves extracted: {summary['moves_extracted']}/{summary['total_runs']}")
    print(f"  - Valid solutions: {summary['valid_solutions']}/{summary['total_runs']}")
    print(f"  - Puzzles solved: {summary['puzzles_solved']}/{summary['total_runs']}")
    print(f"  - Average moves: {summary['average_moves']:.1f} (optimal: {summary['optimal_moves']})")


@click.command()
@click.option('--disks', default=7, help='Number of disks')
@click.option('--repetitions', default=10, help='Number of times to run the experiment')
@click.option('--temperature', default=0.0, help='LLM temperature setting')
@click.option('--model', default='claude-3-5-sonnet-20241022', help='Claude model to use')
@click.option('--output-dir', default='./results', type=click.Path(), help='Directory for results')
def main(disks: int, repetitions: int, temperature: float, model: str, output_dir: str):
    output_path = Path(output_dir)
    asyncio.run(run_experiment(disks, repetitions, temperature, model, output_path))


if __name__ == "__main__":
    main()