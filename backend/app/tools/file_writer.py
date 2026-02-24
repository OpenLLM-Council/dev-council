import os
import re


def parse_code_blocks(llm_output: str) -> list[dict]:
    """
    Parses fenced code blocks with file-path labels from LLM output.

    Expects blocks like:
        ```src/services/AuthService.js
        // code here
        ```

    Returns a list of dicts: [{"path": "src/services/AuthService.js", "content": "..."}]
    """
    # Match ```<label>\n<content>``` — label must look like a file path
    pattern = re.compile(
        r"```([^\n`]+)\n(.*?)```",
        re.DOTALL,
    )

    # Known language identifiers to skip
    language_names = {
        "python", "javascript", "js", "ts", "typescript",
        "bash", "sh", "json", "yaml", "yml", "html", "css",
        "sql", "go", "rust", "java", "cpp", "c", "ruby", "rb",
        "text", "plaintext", "markdown", "md", "xml", "toml",
    }

    blocks = []
    for match in pattern.finditer(llm_output):
        label = match.group(1).strip()
        content = match.group(2)

        # Skip pure language names with no path separator or extension
        lower = label.lower()
        if lower in language_names:
            continue

        # Must look like a file: contain a dot or a slash
        if "." not in label and "/" not in label and "\\" not in label:
            continue

        # Normalize forward slashes → OS path separator
        normalized_path = os.path.join(*label.replace("\\", "/").split("/"))

        blocks.append({"path": normalized_path, "content": content})

    return blocks



def apply_search_replace_blocks(file_path: str, content: str) -> bool:
    """
    Looks for <<<<, ====, >>>> blocks in the content.
    If found, applies them as search/replace operations to the file at file_path.
    Returns True if blocks were found and applied (or attempted), False otherwise.
    """
    pattern = re.compile(r"<<<<\n(.*?)\n====\n(.*?)\n>>>>", re.DOTALL)
    blocks = list(pattern.finditer(content))
    
    if not blocks:
        return False
        
    if not os.path.exists(file_path):
        # If there are blocks but the file doesn't exist, we can't patch it.
        # Fall back to returning False so the caller can just write the content out directly.
        return False

    with open(file_path, "r", encoding="utf-8") as f:
        file_content = f.read()

    for match in blocks:
        search_text = match.group(1)
        replace_text = match.group(2)
        
        # Simple exact string replacement
        # Depending on how the LLM outputs it, we might need to handle slight whitespace differences,
        # but exact match is safest for now.
        if search_text in file_content:
            file_content = file_content.replace(search_text, replace_text, 1)
        else:
            # If the search block wasn't found verbatim, we skip this block. 
            # In a more robust system, we would log this failure to the agent.
            pass

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(file_content)
        
    return True


def write_code_files(llm_output: str, base_dir: str) -> list[str]:
    """
    Parses the LLM output and writes each code block to the given base_dir.
    Supports writing whole files or applying SEARCH/REPLACE blocks to existing files.

    Args:
        llm_output: The raw string output from the coder LLM.
        base_dir: The base directory to write files into (e.g. "outputs/my_project/code/milestone_1").

    Returns:
        List of written file paths (absolute).
    """
    blocks = parse_code_blocks(llm_output)
    written = []

    if not blocks:
        # Fallback: write the whole output as a single file if no blocks parsed
        fallback_path = os.path.join(base_dir, "generated_code.txt")
        os.makedirs(base_dir, exist_ok=True)
        with open(fallback_path, "w", encoding="utf-8") as f:
            f.write(llm_output)
        written.append(fallback_path)
        return written

    for block in blocks:
        file_path = os.path.join(base_dir, block["path"])
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Try to apply as search/replace blocks first
        is_patched = apply_search_replace_blocks(file_path, block["content"])
        
        if not is_patched:
            # If no <<<< blocks were found, treat it as a full file rewrite
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(block["content"])
                
        written.append(file_path)

    return written
