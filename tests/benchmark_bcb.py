import argparse
import json
import os
import subprocess
import re
import sys
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def strip_ansi(text):
    """Remove ANSI escape sequences from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def extract_code(text):
    """Extract code from markdown blocks or return clean text."""
    text = strip_ansi(text)
    # Match the last python code block
    blocks = re.findall(r"```python\n(.*?)\n```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    # Fallback to any code block
    blocks = re.findall(r"```\n(.*?)\n```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    
    # If no blocks found, look for text that starts with 'def ' or 'import '
    # This is a risky fallback but can catch some raw outputs
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(("def ", "import ", "from ")):
            return "\n".join(lines[i:]).strip()
            
    return text.strip()

def run_task(task_id, prompt, model=None, verbose=False):
    """Run dev-council CLI headlessly for a single task."""
    cmd = [sys.executable, "dev_council.py", "--print", "--accept-all"]
    if model:
        cmd.extend(["--model", model])
    cmd.append(prompt)
    
    # Set UTF-8 environment variable for the child process
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    
    try:
        if verbose:
            print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True, 
            encoding='utf-8',
            env=env
        )
        return extract_code(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"\nError running task {task_id}: {e.stderr}")
        return ""
    except Exception as e:
        print(f"\nUnexpected error for task {task_id}: {e}")
        return ""

def main():
    parser = argparse.ArgumentParser(description="BigCodeBench Evaluation Harness for dev-council")
    parser.add_argument("--subset", choices=["instruct", "complete"], default="instruct", help="Benchmark subset")
    parser.add_argument("--n-tasks", type=int, default=None, help="Number of tasks to run (default: all)")
    parser.add_argument("--model", type=str, default=None, help="Model override (e.g. local/llama3)")
    parser.add_argument("--output", type=str, default="samples.jsonl", help="Output JSONL file")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    args = parser.parse_args()

    print(f"Loading BigCodeBench dataset (split: v0.1.3)...")
    try:
        # BigCodeBench uses version numbers as split names (v0.1.0_hf, v0.1.3, v0.1.3, etc.)
        # 'instruct' and 'complete' are prompt types within the same dataset rows.
        ds = load_dataset("bigcode/bigcodebench", split="v0.1.3")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    tasks = ds
    if args.n_tasks:
        tasks = ds.select(range(min(args.n_tasks, len(ds))))

    print(f"Running {len(tasks)} tasks...")
    
    with open(args.output, "w", encoding="utf-8") as f:
        for task in tqdm(tasks, desc="Evaluating"):
            task_id = task["task_id"]
            # Map subset choice to correct field in the dataset
            prompt_key = "instruct_prompt" if args.subset == "instruct" else "complete_prompt"
            prompt = task[prompt_key]
            
            # Enforce code-only output for the benchmark
            if args.subset == "instruct":
                prompt += "\n\nCRITICAL: You must provide the final, complete, self-contained Python code in a single markdown block (```python ... ```) at the end of your response. Do not include any conversational text after the code block."
            
            solution = run_task(task_id, prompt, model=args.model, verbose=args.verbose)
            
            result = {
                "task_id": task_id,
                "solution": solution
            }
            f.write(json.dumps(result) + "\n")
            f.flush()

    print(f"\nGeneration complete. Results saved to: {args.output}")
    print("\nNext steps for evaluation:")
    print(f"1. pip install bigcodebench")
    print(f"2. bigcodebench.sanitize --samples {args.output} --calibrate")
    print(f"3. bigcodebench.evaluate --subset {args.subset} --samples {args.output.replace('.jsonl', '-sanitized-calibrated.jsonl')}")

if __name__ == "__main__":
    main()
