import os

def read_code_directory(directory: str) -> str:
    """
    Reads all files in the given directory and returns their content formatted for an LLM.
    
    Args:
        directory (str): The path to the directory to read.
        
    Returns:
        str: A string containing the relative path and content of each file.
    """
    if not os.path.exists(directory):
        return "Directory does not exist."

    output = []
    for root, dirs, files in os.walk(directory):
        # Skip common non-code or large directories
        if any(skip in root for skip in [".git", "__pycache__", "node_modules", "venv", ".venv"]):
            continue
            
        for file in files:
            # Skip hidden files or common binary files
            if file.startswith('.') or file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.pdf', '.exe', '.pyc')):
                continue
                
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, directory)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    output.append(f"### File: {relative_path}\n```{relative_path}\n{content}\n```\n")
            except Exception as e:
                output.append(f"### File: {relative_path}\nError reading file: {e}\n")
                
    return "\n".join(output) if output else "No readable code files found."
