import asyncio
import os
from urllib.parse import quote
from xml.sax.saxutils import escape, quoteattr

import httpx


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _clean_secret(name: str) -> str:
    """Read a secret from env without leaking it in logs."""
    value = os.getenv(name, "").strip().strip('"').strip("'")
    if value.lower().startswith("bearer "):
        value = value.split(None, 1)[1].strip()
    return value


class EscalationManager:
    """Provider-specific transfer logic driven by OPERATOR_PHONE_NUMBER.

    Telnyx note:
    - For a Telnyx TeXML Application, this uses the TeXML REST Update Call API
      and replaces the live call's TeXML with <Dial><Number>...</Number></Dial>.
    - If you ever move this Telnyx number to a Call Control Application instead,
      set TELNYX_TRANSFER_MODE=call_control to use the older /actions/transfer path.
    """

    def __init__(self):
        self.operator_number = os.getenv("OPERATOR_PHONE_NUMBER", "").strip()
        self.enabled = _env_bool("SAFETY_TRANSFER_ENABLED", True)
        self.telnyx_transfer_mode = os.getenv("TELNYX_TRANSFER_MODE", "texml").strip().lower()

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
                # For TeXML apps this value must be the Telnyx CallSid from the
                # TeXML/websocket request, not the Call Control ID.
                return await self._transfer_telnyx(telnyx_call_control_id, lang, reason)
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

    async def _transfer_telnyx(self, live_call_id: str | None, lang: str, reason: str) -> bool:
        if self.telnyx_transfer_mode in {"call_control", "call-control", "voice_api", "voice-api"}:
            return await self._transfer_telnyx_call_control(live_call_id, reason)
        return await self._transfer_telnyx_texml(live_call_id, lang, reason)

    async def _transfer_telnyx_texml(self, call_sid: str | None, lang: str, reason: str) -> bool:
        """Transfer a live TeXML call to OPERATOR_PHONE_NUMBER using <Dial><Number>."""
        if not call_sid:
            print("[Escalation][Telnyx TeXML] Missing CallSid; cannot update live TeXML call.")
            return False

        account_sid = os.getenv("TELNYX_ACCOUNT_SID", "").strip()
        if not account_sid:
            print("[Escalation][Telnyx TeXML] TELNYX_ACCOUNT_SID missing.")
            return False

        api_key = _clean_secret("TELNYX_API_KEY")
        if not api_key:
            print("[Escalation][Telnyx TeXML] TELNYX_API_KEY missing.")
            return False
        if any(ch.isspace() for ch in api_key):
            print("[Escalation][Telnyx TeXML] TELNYX_API_KEY malformed: remove spaces, quotes, or Bearer prefix.")
            return False

        texml = self._build_telnyx_transfer_texml(lang=lang, reason=reason)
        url = f"https://api.telnyx.com/v2/texml/Accounts/{quote(account_sid, safe='')}/Calls/{quote(call_sid, safe='')}"
        payload = {"Texml": texml}
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, data=payload, headers=headers)
            if response.status_code >= 300:
                print(f"[Escalation][Telnyx TeXML] Update Call HTTP {response.status_code}: {response.text[:300]}")
                return False

        print(f"[Escalation][Telnyx TeXML] Live call updated to <Dial><Number>. reason={reason}")
        return True

    def _build_telnyx_transfer_texml(self, *, lang: str, reason: str) -> str:
        timeout = os.getenv("TELNYX_TRANSFER_TIMEOUT_SECS", "30").strip() or "30"
        try:
            timeout_int = max(5, min(120, int(timeout)))
        except Exception:
            timeout_int = 30

        caller_id = os.getenv("TELNYX_FROM_NUMBER", "").strip()
        caller_id_attr = f" callerId={quoteattr(caller_id)}" if caller_id else ""

        status_callback = self._telnyx_transfer_status_callback(lang=lang, reason=reason)
        status_attrs = ""
        if status_callback:
            status_attrs = (
                f" statusCallback={quoteattr(status_callback)}"
                ' statusCallbackEvent="initiated ringing answered completed"'
                ' statusCallbackMethod="POST"'
            )

        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Response>'
            f'<Dial timeout="{timeout_int}"{caller_id_attr}>'
            f'<Number{status_attrs}>{escape(self.operator_number)}</Number>'
            '</Dial>'
            '</Response>'
        )

    def _telnyx_transfer_status_callback(self, *, lang: str, reason: str) -> str:
        # Optional. Enable only if your app has a /telnyx/transfer-status route.
        if not _env_bool("TELNYX_TRANSFER_STATUS_CALLBACK_ENABLED", False):
            return ""
        base_url = self._public_base_url()
        if not base_url:
            return ""
        return f"{base_url}/telnyx/transfer-status?lang={quote(lang)}&reason={quote(reason)}"

    async def _transfer_telnyx_call_control(self, call_control_id: str | None, reason: str) -> bool:
        """Legacy path for Telnyx Call Control Applications, not TeXML Applications."""
        if not call_control_id:
            print("[Escalation][Telnyx CallControl] Missing call_control_id; cannot transfer live call.")
            return False

        api_key = _clean_secret("TELNYX_API_KEY")
        if not api_key:
            print("[Escalation][Telnyx CallControl] TELNYX_API_KEY missing.")
            return False
        if any(ch.isspace() for ch in api_key):
            print("[Escalation][Telnyx CallControl] TELNYX_API_KEY malformed: remove spaces, quotes, or Bearer prefix.")
            return False

        url = f"https://api.telnyx.com/v2/calls/{quote(call_control_id, safe='')}/actions/transfer"
        payload = {
            "to": self.operator_number,
            "from": os.getenv("TELNYX_FROM_NUMBER", "").strip() or None,
            "timeout_secs": int(os.getenv("TELNYX_TRANSFER_TIMEOUT_SECS", "30") or "30"),
        }
        payload = {k: v for k, v in payload.items() if v}
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code >= 300:
                print(f"[Escalation][Telnyx CallControl] Transfer HTTP {response.status_code}: {response.text[:300]}")
                return False

        print(f"[Escalation][Telnyx CallControl] Transfer command sent. reason={reason}")
        return True
