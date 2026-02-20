from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings

INSTRUCTIONS_PROMPT = """You are a technical writer creating a developer setup guide.

You will receive:
- The **milestone** that was implemented
- The **tech stack** used
- The **generated code files** (as fenced blocks)

## OUTPUT FORMAT (STRICT)

Output a clean Markdown README with:

# How to Run: <milestone name>

## Prerequisites
- List required tools/runtimes with versions

## Installation
```bash
# numbered install steps
```

## Running the Project
```bash
# exact command to run
```

## Expected Output
- What the user should see when it works

## RULES
- Be specific — use actual file names and commands from the code.
- Keep it SHORT — under 200 words total.
- NO extra sections, NO fluff.

Begin.
"""

instructions_prompt = ChatPromptTemplate.from_messages(
    [("system", INSTRUCTIONS_PROMPT), ("human", "{input}")]
)


def get_instructions_agent():
    """Creates a run-instructions generator using the GPT model."""
    llm = ChatOllama(
        model=settings.GPT_LLM,
        base_url=settings.OLLAMA_URL,
        temperature=0,
    )
    return instructions_prompt | llm
