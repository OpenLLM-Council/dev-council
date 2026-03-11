from langchain.agents import create_agent
from app.tools.mermaid import generate_flow_diagram
from app.core.config import settings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

FLOW_DIAGRAM_TEMPLATE = """
You are an expert in creating diagrams using PlantUML.

You will be given the **Software Requirements Specification (SRS)** of a project.
Your task is to return **three** distinct PlantUML diagrams.

---

## STRICT WORKFLOW (CRITICAL)

1. Read and understand the provided SRS.
2. Create three clear and complete PlantUML diagrams:
   - **System Flow Diagram**: Main system flow, major components, decision points.
   - **UML Class Diagram**: Core classes, their properties, methods, and relationships.
   - **ERD (Entity-Relationship Diagram)**: Database tables, fields, and relationships.
3. Wrap each individual diagram inside a standard markdown plantuml block (` ```plantuml ... ``` `). 
4. DO NOT provide any text or explanation outside of your plantuml blocks.

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

```plantuml
@startuml
class Application {{
  +String name
  +run()
}}
@enduml
```

```plantuml
@startuml
entity "CUSTOMER" as e01 {{
  *id : number <<generated>>
  --
  name : text
}}
entity "ORDER" as e02 {{
  *id : number <<generated>>
  --
  customer_id : number <<FK>>
}}
e01 ||..o{{ e02
@enduml
```

---

## INPUT
You will receive the SRS content as input.

## OUTPUT
Return exactly three markdown blocks containing the requested PlantUML diagrams.

## CRITICAL
- Do not provide any explanation outside of the code blocks.
- Wrap each diagram purely in ` ```plantuml ... ``` ` blocks with `@startuml` and `@enduml`.

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
