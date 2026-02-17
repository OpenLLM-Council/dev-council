from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings

CONSENSUS_PROMPT = """You are a senior software engineer.

You are given a milestone to implement, along with the project SRS and tech stack for context.

Propose a **short, concise implementation approach** for the milestone.

## OUTPUT FORMAT (STRICT)
- **Approach**: 2-3 sentences summarizing your strategy.
- **Steps**: A numbered list of 4-6 key implementation steps (one line each).

## RULES
- Keep the ENTIRE response under 200 words.
- Be specific — reference the tech stack.
- NO lengthy explanations, NO meta-commentary.
- Output ONLY the approach (no preamble).
"""

MANAGER_DECISION_PROMPT = """You are the Manager AI.

Multiple LLMs have proposed implementation approaches for a milestone.
Your job is to evaluate all proposals and pick the BEST one.

## OUTPUT FORMAT (STRICT)
- **Chosen LLM**: Name of the LLM you are assigning this task to.
- **Reason**: 1-2 sentences on why this approach is best.
- **Final Approach**: Copy the chosen approach as-is.

## RULES
- Keep the ENTIRE response under 300 words.
- Be decisive — pick ONE winner.
- NO lengthy commentary.
"""

proposal_prompt = ChatPromptTemplate.from_messages(
    [("system", CONSENSUS_PROMPT), ("human", "{input}")]
)

decision_prompt = ChatPromptTemplate.from_messages(
    [("system", MANAGER_DECISION_PROMPT), ("human", "{input}")]
)


def get_consensus_agent(model_name: str):
    """Factory: creates a consensus chain for a given Ollama model."""
    llm = ChatOllama(
        model=model_name,
        base_url=settings.OLLAMA_URL,
        temperature=settings.OLLAMA_TEMPERATURE,
    )
    return proposal_prompt | llm


def get_manager_decision_agent():
    """Creates the manager decision chain using the main GPT model."""
    llm = ChatOllama(
        model=settings.GPT_LLM,
        base_url=settings.OLLAMA_URL,
        temperature=settings.OLLAMA_TEMPERATURE,
    )
    return decision_prompt | llm
