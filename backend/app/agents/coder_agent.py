from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings

CODER_PROMPT = """You are a senior software developer assigned to implement a specific milestone.

You will receive:
- The **milestone** description
- The **chosen implementation approach** (Consensus)
- The **tech stack**
- The **project SRS** for context

## OUTPUT FORMAT (STRICT)

Output ONLY code files using fenced code blocks. 
To ensure files are saved correctly, the fenced code block MUST use the EXACT relative file path as the language label. Do NOT use "python" or "javascript" as the label. 

Examples of correct code blocks:

```src/services/auth_service.py
def login():
    pass
```

```frontend/package.json
{{
  "name": "my-app"
}}
```

The label MUST be the real file path — NOT a description, NOT "relative/path/to/file".

## RULES
- Labels are real file paths only (e.g. `src/index.js`, `app/models/user.py`, `config.yaml`).
- Use ONLY the tech stack provided.
- Write complete, working, production-ready files.
- Include all necessary imports, configs, and entry points.
- NO prose, NO explanations — ONLY fenced code blocks.

Begin.
"""

CODER_REVISION_PROMPT = """You are a senior software developer. Your previously written code needs changes based on reviewer feedback.

## OUTPUT FORMAT (STRICT)

Output ONLY the corrected code files as fenced code blocks.
The fenced code block MUST use the EXACT relative file path as the language label.

Example of a correct code block:

```src/services/auth_service.py
# corrected code
```

## RULES
- Labels are real file paths (e.g. `src/index.js`, `app/models.py`).
- Address ALL reviewer feedback.
- Write complete files — not diffs or snippets.
- NO prose, NO explanations — ONLY fenced code blocks.

Begin.
"""

proposal_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CODER_PROMPT),
        ("human", "{input}"),
    ]
)

revision_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CODER_REVISION_PROMPT),
        ("human", "{input}"),
    ]
)


def get_coder_agent(model_name: str, revision: bool = False):
    """Factory: creates a coder chain for the given Ollama model."""
    llm = ChatOllama(
        model=model_name,
        base_url=settings.OLLAMA_URL,
        temperature=settings.OLLAMA_TEMPERATURE,
        num_predict=4096,  # cap output tokens to prevent infinite generation
    )
    prompt = revision_prompt if revision else proposal_prompt
    return prompt | llm
