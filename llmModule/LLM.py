from dotenv import load_dotenv
from llmModule.GoogleGeminiStrategy import GeminiStrategy
from llmModule.GGUFStrategy import GGUFStrategy
import os

# Load variables once at initialization
load_dotenv()


class LLM:
    """A helper class to switch between models easily with language support."""

    SYSTEM_PROMPTS = {
        "en": "You are a professional systems engineer assistant. Keep answers concise and technical. Reply in English.",
        "fr": "Vous êtes un assistant ingénieur système professionnel. Restez concis et technique. Répondez en français.",
        "sw": "Wewe ni msaidizi wa mhandisi wa mifumo. Toa majibu mafupi ya kiufundi. Jibu kwa Kiswahili."
    }

    @staticmethod
    def get_model(provider=None, lang="en"):
        prompt = LLM.SYSTEM_PROMPTS.get(lang, LLM.SYSTEM_PROMPTS["en"])

        # Dynamic check: Use parameter if provided, otherwise fetch from Spaces Variables
        resolved_provider = provider or os.getenv("LLM_PROVIDER", "gemini").lower()

        if resolved_provider == "gemini":
            gemini_api = os.getenv("GEMINI_API_KEY")
            if not gemini_api:
                raise ValueError("GEMINI_API_KEY is missing from environment secrets.")
            return GeminiStrategy(
                api_key=gemini_api,
                system_prompt=prompt
            )

        elif resolved_provider in ["qwen", "gguf", "local"]:
            local_path = os.getenv("LOCAL_MODEL_PATH")
            return GGUFStrategy(
                model_path=local_path,
                system_prompt=prompt
            )

        raise ValueError(f"Unsupported LLM provider configuration: '{resolved_provider}'")