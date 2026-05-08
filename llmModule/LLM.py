from dotenv import load_dotenv
from llmModule.GoogleGeminiStrategy import GeminiStrategy
from llmModule.GGUFStrategy import GGUFStrategy
import os

class LLM:
    """A helper class to switch between models easily with language support."""

    # Dictionary of prompts to ensure the LLM replies in the correct language
    SYSTEM_PROMPTS = {
        "en": "You are a professional systems engineer assistant. Keep answers concise and technical. Reply in English.",
        "fr": "Vous êtes un assistant ingénieur système professionnel. Restez concis et technique. Répondez en français.",
        "sw": "Wewe ni msaidizi wa mhandisi wa mifumo. Toa majibu mafupi ya kiufundi. Jibu kwa Kiswahili."
    }

    @staticmethod
    def get_model(provider="gemini", lang="en"):
        prompt = LLM.SYSTEM_PROMPTS.get(lang, LLM.SYSTEM_PROMPTS["en"])

        load_dotenv()
        gemini_api = os.getenv("GEMINI_API_KEY")

        if provider == "gemini":
            return GeminiStrategy(
                api_key=gemini_api,
                system_prompt=prompt
            )
        elif provider == "qwen":
            return GGUFStrategy(
                model_path="./models/qwen2.5-7b-instruct-q4_k_m.gguf",
                system_prompt=prompt
            )
        return None


