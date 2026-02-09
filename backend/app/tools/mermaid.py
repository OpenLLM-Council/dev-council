import re
import mermaidian as mm
import os


def clean_mermaid_code(code: str) -> str:
    code = code.strip()

    code = re.sub(r"```mermaid", "", code, flags=re.IGNORECASE)
    code = re.sub(r"```", "", code)

    match = re.search(r"(graph\s+(LR|TD|TB|RL)[\s\S]*)", code)
    if match:
        code = match.group(1)

    return "\n" + code.strip()


def generate_flow_diagram(mermaid_code: str) -> str:
    mermaid_code = clean_mermaid_code(mermaid_code)
    if hasattr(mermaid_code, "content"):
        mermaid_code = mermaid_code.content

    image_bytes = mm.get_mermaid_diagram("png", mermaid_code)

    os.makedirs("outputs", exist_ok=True)
    mm.save_diagram_as_image(path="outputs/flow_diagram.png", diagram=image_bytes)

    return "File saved to outputs/flow_diagram.png"
