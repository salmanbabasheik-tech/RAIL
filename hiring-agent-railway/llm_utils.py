import logging
from typing import Any
from models import ModelProvider, GeminiProvider, BedrockProvider
from prompt import MODEL_PROVIDER_MAPPING, GEMINI_API_KEY, AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION

logger = logging.getLogger(__name__)


def extract_json_from_response(response_text: str) -> str:
    response_text = response_text.strip()
    if "<think>" in response_text:
        think_start = response_text.find("<think>")
        think_end = response_text.find("</think>")
        if think_start != -1 and think_end != -1:
            response_text = response_text[:think_start] + response_text[think_end + 8:]
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    return response_text


def initialize_llm_provider(model_name: str) -> Any:
    model_provider = MODEL_PROVIDER_MAPPING.get(model_name, ModelProvider.OLLAMA)

    if model_provider == ModelProvider.BEDROCK:
        if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
            logger.warning("AWS credentials not found")
        else:
            logger.info(f"Using AWS Bedrock with {model_name}")
            return BedrockProvider(access_key=AWS_ACCESS_KEY, secret_key=AWS_SECRET_KEY, region=AWS_REGION)

    if model_provider == ModelProvider.GEMINI:
        if GEMINI_API_KEY:
            logger.info(f"Using Gemini with {model_name}")
            return GeminiProvider(api_key=GEMINI_API_KEY)

    from models import OllamaProvider
    logger.info(f"Using Ollama with {model_name}")
    return OllamaProvider()
