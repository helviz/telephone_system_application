"""
LLM Service - Google Gemini API
Uses gemini-2.0-flash for ultra-low latency voice responses (~200-400ms TTFT).
No local model, no warm-up, no GPU needed.
"""
import os
import time

import httpx
import structlog
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

log = structlog.get_logger()

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
MAX_TOKENS      = int(os.getenv("MAX_TOKENS", "150"))
TEMPERATURE     = float(os.getenv("TEMPERATURE", "0.7"))

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

SYSTEM_PROMPT = """You are a helpful voice AI assistant on a phone call.
Rules:
- Keep ALL responses to 1-2 SHORT sentences maximum.
- Never use bullet points, lists, markdown, or special characters.
- Speak naturally as if talking on the phone.
- Be direct and concise.
- If you don't know something, say so briefly.
- Never reveal you are an AI language model unless directly asked."""

app = FastAPI(title="LLM Service (Gemini)", version="2.0.0")

_http_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set")
        _http_client = httpx.AsyncClient(
            base_url=GEMINI_BASE_URL,
            timeout=httpx.Timeout(30.0, connect=5.0),
            headers={"Content-Type": "application/json"},
        )
    return _http_client


@app.on_event("startup")
async def startup():
    if not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY not set — LLM service will fail on requests")
        return
    log.info("warming_up_gemini", model=GEMINI_MODEL)
    try:
        await _generate_internal("Hello")
        log.info("gemini_ready", model=GEMINI_MODEL)
    except Exception as e:
        log.warning("warmup_failed", error=str(e))


@app.get("/health")
async def health():
    if not GEMINI_API_KEY:
        return {"status": "error", "reason": "GEMINI_API_KEY not configured"}
    return {"status": "ok", "model": GEMINI_MODEL, "provider": "google-gemini"}


class GenerateRequest(BaseModel):
    prompt: str
    system: str | None = None
    max_tokens: int = MAX_TOKENS
    temperature: float = TEMPERATURE


@app.post("/generate")
async def generate(req: GenerateRequest):
    t_start = time.perf_counter()

    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Empty prompt")

    response_text = await _generate_internal(
        prompt=req.prompt,
        system=req.system or SYSTEM_PROMPT,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )

    latency_ms = round((time.perf_counter() - t_start) * 1000)
    log.info("generated", latency_ms=latency_ms, model=GEMINI_MODEL,
             response_preview=response_text[:80])

    return {"response": response_text, "model": GEMINI_MODEL, "latency_ms": latency_ms}


async def _generate_internal(
    prompt: str,
    system: str = SYSTEM_PROMPT,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
) -> str:
    client = get_client()

    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
            "topP": 0.9,
        },
    }

    url = f"/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    try:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()

        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError(f"No candidates in Gemini response: {data}")

        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            raise ValueError(f"No parts in candidate: {candidates[0]}")

        text = "".join(p.get("text", "") for p in parts).strip()

        finish_reason = candidates[0].get("finishReason", "STOP")
        if finish_reason not in ("STOP", "MAX_TOKENS"):
            log.warning("unexpected_finish_reason", reason=finish_reason)

        return text

    except httpx.HTTPStatusError as e:
        body = e.response.text[:300]
        log.error("gemini_http_error", status=e.response.status_code, body=body)
        if e.response.status_code == 429:
            raise HTTPException(status_code=429, detail="Gemini rate limit — retry shortly")
        if e.response.status_code == 401:
            raise HTTPException(status_code=401, detail="Invalid GEMINI_API_KEY")
        raise HTTPException(status_code=502, detail=f"Gemini error {e.response.status_code}: {body}")
    except Exception as e:
        log.error("gemini_error", error=str(e))
        raise HTTPException(status_code=503, detail=f"LLM request failed: {e}")
