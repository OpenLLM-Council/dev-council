import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    GPT_LLM = os.getenv("GPT_LLM", "qwen2.5:1.5b")
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", 0))


settings = Settings()
