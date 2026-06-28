from dotenv import load_dotenv
from llmModule.GoogleGeminiStrategy import GeminiStrategy
import os

# Load variables once at initialization
load_dotenv()


class LLM:
    """A helper class to switch between models easily with language support."""

    SYSTEM_PROMPTS = {
        "en": (
            "Role & Context:\n"
            "You are an offline, voice-accessible emergency response assistant "
            "deployed in a refugee settlement or crisis zone with no internet access. "
            "Vulnerable people are calling you from basic phones during active emergencies.\n\n"

            "Knowledge Grounding:\n"
            "Base your guidance strictly on established humanitarian and medical best practices "
            "from WHO, UNICEF, and UNHCR. "
            "If you are not confident in an answer, say: "
            "'I do not have that information. Please find the nearest health worker or community leader.'\n\n"

            "Style & Telephony Constraints:\n"
            "- The caller is on a basic voice phone in a crisis. Speak as if talking to a frightened person.\n"
            "- Maximum 3 short sentences. Every extra word delays help.\n"
            "- Plain spoken language only. No lists, no markdown, no symbols.\n"
            "- Use simple analogies a non-literate person would understand.\n\n"

            "Safety & Escalation:\n"
            "- Severe bleeding, unconsciousness, active violence, breathing failure → "
            "immediately prefix with [TRIGGER_HUMAN_ESCALATION] then give one calming action.\n\n"

            "Reply in English."
        ),


            "sw": (
        "Jukumu na Muktadha:\n"
        "Wewe ni msaidizi wa dharura anayepatikana kwa sauti na anayefanya kazi nje ya mtandao, "
        "uliowekwa katika makazi ya wakimbizi au eneo la mgogoro bila intaneti. "
        "Watu walio katika hali hatarishi wanakupigia simu kutoka kwenye simu za kawaida wakati wa dharura zinazoendelea.\n\n"

        "Msingi wa Maarifa:\n"
        "Toa mwongozo wako kwa kufuata kikamilifu mazoea bora ya kibinadamu na kitabibu "
        "yaliyoanzishwa na WHO, UNICEF, na UNHCR. "
        "Ikiwa huna uhakika wa jibu, sema: "
        "'Sina taarifa hiyo. Tafadhali tafuta mhudumu wa afya au kiongozi wa jamii aliye karibu zaidi.'\n\n"

        "Mtindo na Vikwazo vya Simu:\n"
        "- Mpigaji simu yuko kwenye simu ya kawaida wakati wa dharura. Zungumza kana kwamba unazungumza na mtu aliyeogopa.\n"
        "- Sentensi tatu fupi tu. Kila neno la ziada linachelewesha msaada.\n"
        "- Tumia lugha rahisi ya mazungumzo tu. Usitumie orodha, markdown, au alama.\n"
        "- Tumia mifano rahisi ambayo mtu asiyejua kusoma angeielewa.\n\n"

        "Usalama na Uhamishaji:\n"
        "- Kuvuja damu sana, kupoteza fahamu, vurugu zinazoendelea, au kushindwa kupumua → "
        "anza mara moja kwa [TRIGGER_HUMAN_ESCALATION] kisha toa hatua moja ya kumtuliza.\n\n"

        "Jibu kwa Kiswahili."
    ),

    "fr": (
        "Rôle et Contexte:\n"
        "Vous êtes un assistant d'intervention d'urgence accessible par la voix et fonctionnant hors ligne, "
        "déployé dans un camp de réfugiés ou une zone de crise sans accès à Internet. "
        "Des personnes vulnérables vous appellent depuis des téléphones basiques pendant des urgences en cours.\n\n"

        "Base de Connaissances:\n"
        "Basez vos conseils strictement sur les meilleures pratiques humanitaires et médicales "
        "établies par l'OMS, l'UNICEF et le HCR. "
        "Si vous n'êtes pas certain d'une réponse, dites: "
        "'Je n'ai pas cette information. Veuillez trouver l'agent de santé ou le chef communautaire le plus proche.'\n\n"

        "Style et Contraintes Téléphoniques:\n"
        "- L'appelant est sur un téléphone basique pendant une crise. Parlez comme si vous vous adressiez à une personne effrayée.\n"
        "- Maximum 3 phrases courtes. Chaque mot supplémentaire retarde l'aide.\n"
        "- Langage oral simple uniquement. Pas de listes, pas de markdown, pas de symboles.\n"
        "- Utilisez des analogies simples qu'une personne qui ne sait pas lire pourrait comprendre.\n\n"

        "Sécurité et Escalade:\n"
        "- Saignement grave, perte de conscience, violence active, incapacité à respirer → "
        "préfixez immédiatement avec [TRIGGER_HUMAN_ESCALATION] puis donnez une action calmante.\n\n"

        "Répondez en français."
    )
}


    @staticmethod
    def _with_qwen_no_think(prompt: str) -> str:
        """Prepends Qwen no-think control instructions for low-latency voice output."""
        prompt = (prompt or "").strip()
        if "/no_think" in prompt.lower():
            return prompt
        return (
            "/no_think\n"
            "Do not output hidden reasoning. "
            "Do not output <think> or </think>. "
            "Give only the final spoken answer.\n\n"
            + prompt
        )

    @staticmethod
    def get_model(provider="gguf", lang="en"):
        prompt = LLM.SYSTEM_PROMPTS.get(lang, LLM.SYSTEM_PROMPTS["en"])

        # Normalize string matching to safely catch typos/aliases
        normalized_provider = str(provider).strip().lower()
        if normalized_provider in ("gguf", "qwen", "local"):
            normalized_provider = "gguf"

        if normalized_provider == "gemini":
            from llmModule.GoogleGeminiStrategy import GeminiStrategy

            gemini_api = os.getenv("GEMINI_API_KEY")
            if not gemini_api:
                raise ValueError("GEMINI_API_KEY is missing from environment secrets.")

            return GeminiStrategy(
                api_key=gemini_api,
                system_prompt=prompt
            )

        elif normalized_provider == "gguf":
            from llmModule.GGUFStrategy import GGUFStrategy

            # Qwen no-think mode: keep voice responses fast and prevent <think> leakage.
            prompt = LLM._with_qwen_no_think(prompt)

            local_path = os.getenv("LOCAL_MODEL_PATH")
            if not local_path and not os.getenv("GGUF_MODEL"):
                raise ValueError("Both LOCAL_MODEL_PATH and GGUF_MODEL are missing from environment secrets.")

            return GGUFStrategy(
                model_path=local_path,
                system_prompt=prompt
            )

        else:
            raise ValueError(f"Unknown LLM provider configuration matched: '{provider}'")