from langchain.agents import create_agent
from app.tools.mermaid import generate_flow_diagram
from app.core.config import settings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

FLOW_DIAGRAM_TEMPLATE = """
You are an expert in creating flow diagrams using Mermaid.

You will be given the **Software Requirements Specification (SRS)** of a project.
Your task is to return a **mermaid code** that represents the system workflow.

---

## STRICT WORKFLOW (CRITICAL)

1. Read and understand the provided SRS.
2. Create a clear and complete **Mermaid flow diagram** representing:
   - Main system flow
   - Major components
   - Decision points (if any)
3. Don't use ```mermaid\n and ``` at the end of the code.
4. Just return the string of mermaid code.

## SAMPLE CODE

graph LR
    A[Square Rect] -- Link text --> B((Circle))
    A --> C(Round Rect)
    B --> D{{Rhombus}}
    C --> D

---

## INPUT
You will receive the SRS content as input.

## OUTPUT
Return the mermaid code for the flow diagram which should follow the sample code.

## CRITICAL
- Don't provide response in bullet points or numbered list.
- Don't provide any explanation.
- Just return the mermaid code as a string.

---

Begin

"""

prompt = ChatPromptTemplate.from_messages(
    [("system", FLOW_DIAGRAM_TEMPLATE), ("human", "{input}")]
)


def get_flow_diagram_agent():
    llm = ChatOllama(
        model=settings.MISTRAL_LLM,
        base_url=settings.OLLAMA_URL,
        temperature=settings.OLLAMA_TEMPERATURE,
    )

    agent = prompt | llm

    return agent
