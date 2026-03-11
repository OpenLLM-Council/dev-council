from langchain.agents import create_agent
from app.tools.mermaid import generate_flow_diagram
from app.core.config import settings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

FLOW_DIAGRAM_TEMPLATE = """
You are an expert in creating diagrams using PlantUML.

You will be given the **Software Requirements Specification (SRS)** of a project.
Your task is to return a **{diagram_type}**.

---

## STRICT WORKFLOW (CRITICAL)

1. Read and understand the provided SRS.
2. Create a clear and complete {diagram_type} based on the SRS.
3. Wrap your diagram inside a standard markdown plantuml block.
4. DO NOT provide any text or explanation outside of your plantuml block.

## SAMPLE FORMAT

```plantuml
@startuml
actor User
User -> Application: Requests
Application -> Database: Query
Database --> Application: Response
Application --> User: Result
@enduml
```

---

## INPUT
You will receive the SRS content as input.

## OUTPUT
Return exactly ONE markdown block containing the requested {diagram_type} in PlantUML.

## CRITICAL
- Do not provide any explanation outside of the code block.
- Wrap the diagram purely in ` ```plantuml ... ``` ` block with `@startuml` and `@enduml`.

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
