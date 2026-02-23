from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings

REVIEWER_PROMPT = """You are a senior code reviewer.

You will receive:
- The **milestone** description
- The **chosen implementation approach**
- The **generated code** to review

## YOUR JOB
Review the code for:
1. Correctness — does it actually implement the milestone?
2. Completeness — are all required files/functions present?
3. Alignment — does it follow the chosen approach?
4. Quality — no obvious bugs, proper structure

## OUTPUT FORMAT (STRICT)

Start with EXACTLY one of:
- `APPROVED` — if the code is good to go
- `NEEDS_REVISION` — if changes are required

Then on a new line, provide brief bullet-point feedback (max 5 points).

Example:
APPROVED
- Clean implementation
- Follows the chosen approach correctly

Example:
NEEDS_REVISION
- Missing error handling in the API route
- Database connection not closed properly

## RULES
- Be concise and specific.
- Do NOT rewrite the code yourself.
- Do NOT output anything before APPROVED or NEEDS_REVISION.

Begin.
"""

reviewer_prompt = ChatPromptTemplate.from_messages(
    [("system", REVIEWER_PROMPT), ("human", "{input}")]
)


def get_reviewer_agent():
    """Creates a code reviewer chain using the GPT model."""
    llm = ChatOllama(
        model=settings.GPT_LLM,
        base_url=settings.OLLAMA_URL,
        temperature=0,
    )
    return reviewer_prompt | llm
