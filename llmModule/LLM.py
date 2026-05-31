from dotenv import load_dotenv
from llmModule.GoogleGeminiStrategy import GeminiStrategy
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

        # Hardwired to Gemini (gemma-4-26b-a4b-it) for infrastructure testing.
        # To switch to GGUF, comment out the Gemini block and
        # uncomment the GGUF block beneath it.

        # --- GEMINI (active) ---
        gemini_api = os.getenv("GEMINI_API_KEY")
        if not gemini_api:
            raise ValueError("GEMINI_API_KEY is missing from environment secrets.")
        return GeminiStrategy(
            api_key=gemini_api,
            system_prompt=prompt
        )

        # --- GGUF / local (inactive) ---
        # from llmModule.GGUFStrategy import GGUFStrategy
        # local_path = os.getenv("LOCAL_MODEL_PATH")
        # if not local_path:
        #     raise ValueError("LOCAL_MODEL_PATH is missing from environment secrets.")
        # return GGUFStrategy(
        #     model_path=local_path,
        #     system_prompt=prompt
        # )