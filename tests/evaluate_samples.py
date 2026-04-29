import json
import subprocess
import tempfile
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def run_test(code, test_code, task_id):
    """Execute code + test in a temporary file."""
    full_code = f"{code}\n\n{test_code}"
    
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as tmp:
        tmp.write(full_code)
        tmp_path = tmp.name

    try:
        # Run with a 10s timeout to prevent infinite loops
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return True, ""
        else:
            return False, result.stderr or result.stdout
    except subprocess.TimeoutExpired:
        return False, "Timed out after 10 seconds"
    except Exception as e:
        return False, str(e)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def main():
    input_path = "samples-sanitized.jsonl"
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run sanitize_samples.py first.")
        return

    print("Loading BigCodeBench dataset tests...")
    ds = load_dataset("bigcode/bigcodebench", split="v2.7.0")
    task_map = {row["task_id"]: row for row in ds}

    results = []
    passed = 0
    total = 0

    print(f"Evaluating samples from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            task_id = data["task_id"]
            solution = data["solution"]
            
            if task_id not in task_map:
                print(f"Skipping unknown task: {task_id}")
                continue
                
            task_data = task_map[task_id]
            test_code = task_data["test"]
            
            # BigCodeBench tests often expect specific imports or setup
            # We combine the task entry point with the solution and test
            is_pass, error = run_test(solution, test_code, task_id)
            
            total += 1
            if is_pass:
                passed += 1
                print(f"[PASS] {task_id}")
            else:
                print(f"[FAIL] {task_id}")
                if error:
                    # Show first line of error
                    print(f"   Error: {error.strip().splitlines()[0]}")

    if total > 0:
        print(f"\nFinal Score: {passed}/{total} ({passed/total:.1%})")
    else:
        print("\nNo tasks evaluated.")

if __name__ == "__main__":
    main()
