import re
import os


def clean_mermaid_code(code: str) -> str:
    code = code.strip()

    code = re.sub(r"```mermaid", "", code, flags=re.IGNORECASE)
    code = re.sub(r"```", "", code)

    match = re.search(r"(graph\s+(LR|TD|TB|RL|BT)[\s\S]*)", code)
    if match:
        code = match.group(1)

    return "\n" + code.strip()


def generate_flow_diagram(mermaid_code: str, project_path: str) -> str:
    """Generate flow diagram as HTML with embedded Mermaid code."""
    mermaid_code = clean_mermaid_code(mermaid_code)
    if hasattr(mermaid_code, "content"):
        mermaid_code = mermaid_code.content

    os.makedirs(f"{project_path}", exist_ok=True)
    
    # Generate HTML file with Mermaid diagram
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Flow Diagram</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({{ startOnLoad: true }});</script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            background-color: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            max-width: 1200px;
            width: 90%;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .mermaid {{
            display: flex;
            justify-content: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>System Flow Diagram</h1>
        <div class="mermaid">
{mermaid_code}
        </div>
    </div>
</body>
</html>"""
    
    html_path = f"{project_path}/flow_diagram.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Also try to generate PNG using mermaidian if available
    try:
        import mermaidian as mm
        image_bytes = mm.get_mermaid_diagram("png", mermaid_code)
        png_path = f"{project_path}/flow_diagram.png"
        mm.save_diagram_as_image(path=png_path, diagram=image_bytes)
        return f"Diagram saved: {html_path} and {png_path}"
    except Exception as e:
        return f"HTML diagram saved to {html_path} (PNG generation skipped: {str(e)})"
