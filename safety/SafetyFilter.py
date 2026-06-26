import os
import re
import unicodedata
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(slots=True)
class SafetyResult:
    category: str
    severity: str
    matched_terms: list[str] = field(default_factory=list)
    source_text: str = ""


SAFETY_KEYWORDS: dict[str, dict[str, list[str]]] = {
    "en": {
        "medical_emergency": [
            "can't breathe", "cannot breathe", "difficulty breathing", "shortness of breath",
            "chest pain", "unconscious", "not waking up", "bleeding heavily", "heavy bleeding",
            "seizure", "convulsion", "stroke", "heart attack", "poison", "poisoning",
            "overdose", "severe burn", "severe burns", "pregnant and bleeding", "labor pains",
        ],
        "violence": [
            "attacking me", "being attacked", "they have a gun", "has a gun", "gun",
            "knife", "being beaten", "beating me", "kidnapped", "abducted", "rape",
            "robbery", "domestic violence", "someone is following me", "threatening to kill",
        ],
        "self_harm": [
            "kill myself", "suicide", "end my life", "hurt myself", "harm myself",
            "overdose myself", "i want to die", "i don't want to live",
        ],
        "operator_request": ["operator", "human", "agent", "speak to someone", "talk to someone", "press zero"],
    },
    "fr": {
        "medical_emergency": [
            "je ne peux pas respirer", "je n'arrive pas a respirer", "difficulte a respirer",
            "douleur a la poitrine", "mal a la poitrine", "inconscient", "ne se reveille pas",
            "saigne beaucoup", "saignement abondant", "crise", "convulsion", "avc",
            "accident vasculaire cerebral", "crise cardiaque", "empoisonnement", "surdose",
            "brulure grave", "enceinte et saigne",
        ],
        "violence": [
            "on m'attaque", "on m attaque", "il a une arme", "elle a une arme", "arme a feu",
            "pistolet", "couteau", "on me bat", "kidnappe", "enleve", "viol",
            "vol a main armee", "violence domestique", "menace de me tuer",
        ],
        "self_harm": [
            "me suicider", "suicide", "mettre fin a ma vie", "me faire du mal",
            "me blesser", "je veux mourir", "je ne veux plus vivre",
        ],
        "operator_request": ["operateur", "humain", "agent", "parler a quelqu'un", "parler a quelqu un", "zero"],
    },
    "sw": {
        "medical_emergency": [
            "siwezi kupumua", "shida kupumua", "napata shida kupumua", "maumivu ya kifua",
            "kifua kinauma", "amepoteza fahamu", "hapumui", "hataki kuamka", "damu nyingi",
            "kutokwa na damu nyingi", "degedege", "mshtuko", "kiharusi", "mshtuko wa moyo",
            "sumu", "amekunywa sumu", "kuzidisha dawa", "amezidisha dawa", "kuungua sana",
        ],
        "violence": [
            "wananishambulia", "ananivamia", "ana bunduki", "bunduki", "kisu", "napigwa",
            "ananipiga", "nimetekwa", "kutekwa", "ubakaji", "kubakwa", "wizi",
            "vurugu za nyumbani", "anatishia kuniua",
        ],
        "self_harm": [
            "kujiua", "nataka kujiua", "kujimaliza", "kumaliza maisha yangu",
            "kujiumiza", "kujidhuru", "sitaki kuishi", "nataka kufa",
        ],
        "operator_request": ["operator", "mhudumu", "mtu", "ongea na mtu", "zungumza na mtu", "sifuri"],
    },
}

SEVERITY = {
    "medical_emergency": "critical",
    "violence": "critical",
    "self_harm": "critical",
    "operator_request": "transfer",
}


class SafetyFilter:
    """Deterministic multilingual safety filter for ASR and LLM text."""

    def __init__(self, enabled: bool | None = None):
        self.enabled = _env_bool("SAFETY_FILTER_ENABLED", True) if enabled is None else enabled

    @staticmethod
    def _normalize(text: str) -> str:
        text = unicodedata.normalize("NFKD", text or "")
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = text.lower()
        text = re.sub(r"[^a-z0-9'\s]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def detect(self, text: str, lang: str = "en") -> SafetyResult | None:
        if not self.enabled:
            return None
        normalized = self._normalize(text)
        if not normalized:
            return None

        rules = SAFETY_KEYWORDS.get(lang) or SAFETY_KEYWORDS["en"]
        for category in ("self_harm", "medical_emergency", "violence", "operator_request"):
            matched: list[str] = []
            for term in rules.get(category, []):
                norm_term = self._normalize(term)
                if norm_term and norm_term in normalized:
                    matched.append(term)
            if matched:
                return SafetyResult(
                    category=category,
                    severity=SEVERITY.get(category, "critical"),
                    matched_terms=matched,
                    source_text=text,
                )
        return None
