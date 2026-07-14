import os
from dotenv import load_dotenv
from models import ModelProvider

load_dotenv()

DEFAULT_MODEL = "us.anthropic.claude-opus-4-6-v1"
PROVIDER = "bedrock"

MODEL_PARAMETERS = {
    "us.anthropic.claude-opus-4-6-v1": {"temperature": 0.1},
    "gemini-2.0-flash": {"temperature": 0.1, "top_p": 0.9},
    "gemini-2.5-flash": {"temperature": 0.1, "top_p": 0.9},
    "gemma3:4b": {"temperature": 0.1, "top_p": 0.9},
}

MODEL_PROVIDER_MAPPING = {
    "us.anthropic.claude-opus-4-6-v1": ModelProvider.BEDROCK,
    "gemini-2.0-flash": ModelProvider.GEMINI,
    "gemini-2.5-flash": ModelProvider.GEMINI,
    "gemma3:4b": ModelProvider.OLLAMA,
}

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
