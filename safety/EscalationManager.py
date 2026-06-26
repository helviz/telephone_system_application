import asyncio
import os
from urllib.parse import quote

import httpx


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


class EscalationManager:
    """Provider-specific transfer logic driven by OPERATOR_PHONE_NUMBER."""

    def __init__(self):
        self.operator_number = os.getenv("OPERATOR_PHONE_NUMBER", "").strip()
        self.enabled = _env_bool("SAFETY_TRANSFER_ENABLED", True)

    @staticmethod
    def _public_base_url() -> str:
        host = os.getenv("PUBLIC_HOST", "").strip()
        scheme = os.getenv("PUBLIC_SCHEME", "https").strip() or "https"
        if not host:
            return ""
        return f"{scheme}://{host}"

    async def transfer_to_operator(
        self,
        *,
        provider: str,
        lang: str,
        reason: str,
        twilio_call_sid: str | None = None,
        telnyx_call_control_id: str | None = None,
    ) -> bool:
        if not self.enabled:
            print(f"[Escalation] Transfer disabled. reason={reason}")
            return False
        if not self.operator_number:
            print("[Escalation] OPERATOR_PHONE_NUMBER is not set; cannot transfer.")
            return False

        provider = (provider or "").lower()
        try:
            if provider == "twilio":
                return await self._transfer_twilio(twilio_call_sid, lang, reason)
            if provider == "telnyx":
                return await self._transfer_telnyx(telnyx_call_control_id, reason)
            print(f"[Escalation] Unknown provider={provider!r}; cannot transfer.")
            return False
        except Exception as exc:
            print(f"[Escalation] Transfer failed provider={provider} reason={reason}: {exc}")
            return False

    async def _transfer_twilio(self, call_sid: str | None, lang: str, reason: str) -> bool:
        if not call_sid:
            print("[Escalation][Twilio] Missing CallSid; cannot update live call.")
            return False
        base_url = self._public_base_url()
        if not base_url:
            print("[Escalation][Twilio] PUBLIC_HOST missing; cannot build transfer callback URL.")
            return False

        account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
        auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
        if not account_sid or not auth_token:
            print("[Escalation][Twilio] TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN missing.")
            return False

        transfer_url = f"{base_url}/twilio/transfer?lang={quote(lang)}&reason={quote(reason)}"

        def _update_call():
            from twilio.rest import Client
            client = Client(account_sid, auth_token)
            client.calls(call_sid).update(method="POST", url=transfer_url)

        await asyncio.to_thread(_update_call)
        print(f"[Escalation][Twilio] Updated live call {call_sid} to transfer URL.")
        return True

    async def _transfer_telnyx(self, call_control_id: str | None, reason: str) -> bool:
        if not call_control_id:
            print("[Escalation][Telnyx] Missing call_control_id; cannot transfer live call.")
            return False
        api_key = os.getenv("TELNYX_API_KEY", "").strip()
        if not api_key:
            print("[Escalation][Telnyx] TELNYX_API_KEY missing.")
            return False

        url = f"https://api.telnyx.com/v2/calls/{call_control_id}/actions/transfer"
        payload = {
            "to": self.operator_number,
            "from": os.getenv("TELNYX_FROM_NUMBER", "").strip() or None,
        }
        payload = {k: v for k, v in payload.items() if v}
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code >= 300:
                print(f"[Escalation][Telnyx] Transfer HTTP {response.status_code}: {response.text[:300]}")
                return False

        print(f"[Escalation][Telnyx] Transfer command sent. reason={reason}")
        return True
