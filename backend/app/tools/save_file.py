import os
import json
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from pydantic import BaseModel
from typing import Union, Any
from markdown_pdf import MarkdownPdf, Section


def _serialize_content(obj: Any) -> str:
    """Convert various object types to string for file writing."""
    if isinstance(obj, BaseMessage):
        return obj.content
    elif isinstance(obj, BaseModel):
        return obj.model_dump_json(indent=2)
    elif isinstance(obj, dict):
        serialized_dict = {}
        for k, v in obj.items():
            if isinstance(v, BaseMessage):
                serialized_dict[k] = v.content
            elif isinstance(v, list):
                serialized_dict[k] = [_serialize_content(item) if isinstance(item, BaseMessage) else item for item in v]
            else:
                serialized_dict[k] = v
        return json.dumps(serialized_dict, indent=2, default=str)
    else:
        return str(obj)


def save_file(file_name: str, content: Any) -> str:
    """
    Useful for saving files. Supports string, dict, Pydantic model, or LangChain message content.
    Args:
        file_name (str): The name/path of the file to save (relative to outputs/).
        content (Any): The content to save to the file.
    """
    file_path = os.path.join("outputs", file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    content_str = _serialize_content(content)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content_str)
    return f"File saved to {file_path}"


def markdown_to_pdf(markdown_file_name: str, pdf_file_name: str) -> str:
    """
    Convert a markdown file to PDF using markdown-pdf (Windows-compatible).
    Args:
        markdown_file_name (str): Path to the markdown file (e.g., "project_plan.md")
        pdf_file_name (str): Output PDF file name (e.g., "project_plan.pdf")
    Returns:
        str: Success message with PDF file path
    """
    os.makedirs("outputs", exist_ok=True)
    
    md_path = os.path.join("outputs", markdown_file_name)
    pdf_path = os.path.join("outputs", pdf_file_name)
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    
    # Read markdown file
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
    
    # Create PDF with markdown-pdf
    pdf = MarkdownPdf()
    pdf.add_section(Section(md_content))
    pdf.save(pdf_path)
    
    return f"PDF saved to {pdf_path}"


