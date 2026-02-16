from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from app.core.config import settings

TECH_STACK_AGENT_PROMPT = """
You are an expert & senior software developer lead.

Your task is to recommend the best possible technology stack based on the user requirement.

## INPUT
- SRS Document

## REQUIRED OUTPUT FORMAT (CRITICAL)
Your FINAL output MUST be a **Markdown file** containing ONLY a single Markdown table.
Do NOT include any introductory text, concluding remarks, or explanations.
Do NOT use code blocks (```markdown or ```).
Just the raw markdown table.

### TABLE FORMAT
| Category | Technologies |
|----------|------------|
| Frontend | React, Next.js |
| Backend | Node.js, Express.js |
| Database | MongoDB |
| Deployment | Docker |
... (Add more categories as needed)

## RULES
- Recommend the best possible technology stack based on the user requirement.
- The technology stack should be scalable, maintainable, cost-effective, and easy to scale.
- OUTPUT ONLY THE TABLE.
"""


def get_tech_stack_agent():
    llm = ChatOllama(
        model=settings.DEEPSEEK_LLM,
        base_url=settings.OLLAMA_URL,
        temperature=settings.OLLAMA_TEMPERATURE,
    )

    agent = create_agent(
        model=llm,
        system_prompt=TECH_STACK_AGENT_PROMPT,
    )

    return agent
