import re
import os
from plantuml import PlantUML

def extract_plantuml_blocks(code: str) -> list[str]:
    """Extracts all plantuml code blocks from the LLM response."""
    # Find all content between ```plantuml and ```
    blocks = re.findall(r"```plantuml\n(.*?)\n```", code, flags=re.IGNORECASE | re.DOTALL)
    
    # Fallback to general block extraction if ```plantuml isn't used perfectly
    if not blocks:
        blocks = re.findall(r"```[a-zA-Z]*\n(.*?)\n```", code, flags=re.DOTALL)
        
    # Final fallback if no blocks are found
    if not blocks:
        clean_code = code.strip()
        if clean_code:
            blocks = [clean_code]
            
    return [b.strip() for b in blocks if b.strip()]


def generate_flow_diagram(mermaid_code: dict | str, project_path: str) -> str:
    """Generate flow diagrams as HTML with embedded PlantUML diagrams."""
    
    # Normalize input and extract valid blocks
    diagram_blocks = {}
    if isinstance(mermaid_code, dict):
        for dtype, content in mermaid_code.items():
            if hasattr(content, "content"):
                content = content.content
            extracted = extract_plantuml_blocks(content)
            if extracted:
                diagram_blocks[dtype] = extracted[0]
            else:
                diagram_blocks[dtype] = content
    else:
        # Backward compatibility for flat strings
        if hasattr(mermaid_code, "content"):
            mermaid_code = mermaid_code.content
        blocks = extract_plantuml_blocks(mermaid_code)
        for i, block in enumerate(blocks, 1):
            diagram_blocks[f"Diagram {i}"] = block

    os.makedirs(f"{project_path}", exist_ok=True)
    
    plantuml_server = PlantUML(url='http://www.plantuml.com/plantuml/svg/')
    
    # Generate HTML file with multiple PlantUML diagrams
    divs = ""
    for dtype, block in diagram_blocks.items():
        try:
            # We construct the image URL. Ensure block starts with @startuml
            clean_block = block
            if not clean_block.startswith("@startuml"):
                clean_block = f"@startuml\n{clean_block}\n@enduml"
                
            img_url = plantuml_server.get_url(clean_block)
            divs += f'''
            <div class="diagram-section">
                <h2>{dtype}</h2>
                <div class="plantuml-diagram">
                    <img src="{img_url}" alt="PlantUML {dtype}" style="max-width: 100%; height: auto; border: 1px solid #ddd; padding: 10px; border-radius: 4px; background: white;" />
                </div>
            </div>
            '''
        except Exception as e:
            print(f"Failed to generate PlantUML for block {dtype}: {e}")

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>System Architecture Diagrams</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 40px 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            max-width: 1200px;
            width: 100%;
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 2px solid #eaeaea;
            padding-bottom: 10px;
        }}
        .diagram-section {{
            margin-bottom: 50px;
            padding-bottom: 30px;
            border-bottom: 1px solid #eee;
        }}
        .diagram-section:last-child {{
            border-bottom: none;
        }}
        .diagram-section h2 {{
            color: #555;
            font-size: 1.5em;
            margin-bottom: 20px;
        }}
        .plantuml-diagram {{
            display: flex;
            justify-content: center;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>System Architecture Diagrams</h1>
        {divs}
    </div>
</body>
</html>"""
    
    html_path = f"{project_path}/flow_diagram.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return f"Diagrams saved as HTML at: {html_path}"
