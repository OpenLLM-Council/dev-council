import json
import re
from pathlib import Path

def extract_python_code(text):
    """Cleanly extract Python code from markdown or raw text."""
    if not text:
        return ""
    
    # 1. Try to find triple-backtick blocks
    blocks = re.findall(r"```python\n(.*?)\n```", text, re.DOTALL)
    if not blocks:
        blocks = re.findall(r"```\n(.*?)\n```", text, re.DOTALL)
        
    if blocks:
        # Use the last block (usually where the final answer is)
        code = blocks[-1].strip()
    else:
        # 2. Fallback: take the whole text but strip common markdown headers
        code = text.strip()
        lines = code.splitlines()
        clean_lines = []
        for line in lines:
            if not line.startswith(("#", "Here is", "Below is", "```")):
                clean_lines.append(line)
        code = "\n".join(clean_lines).strip()
        
    return code

def sanitize(input_path, output_path):
    print(f"Sanitizing {input_path}...")
    
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        print(f"Error: {input_path} not found.")
        return

    sanitized_count = 0
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                task_id = data.get("task_id")
                solution = data.get("solution", "")
                
                # Extract clean code
                clean_code = extract_python_code(solution)
                
                # Save sanitized version
                result = {
                    "task_id": task_id,
                    "solution": clean_code
                }
                f_out.write(json.dumps(result) + "\n")
                sanitized_count += 1
            except Exception as e:
                print(f"Error processing line: {e}")

    print(f"Successfully sanitized {sanitized_count} tasks.")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    sanitize("samples.jsonl", "samples-sanitized.jsonl")
