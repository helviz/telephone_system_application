import asyncio
import html
import os
from urllib.parse import quote

import httpx


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _clean_bearer_token(raw: str) -> str:
    """Return a raw Telnyx API token even if the env accidentally includes quotes/Bearer."""
    token = (raw or "").strip().strip('"').strip("'")
    if token.lower().startswith("bearer "):
        token = token.split(None, 1)[1].strip()
    return token


def _xml_escape(value: str) -> str:
    return html.escape(value or "", quote=True)


class EscalationManager:
    """
    Provider-specific transfer logic driven by OPERATOR_PHONE_NUMBER.

    Telnyx default is TeXML because this project uses a Telnyx TeXML Application.
    Set TELNYX_TRANSFER_MODE=call_control only if the Telnyx number is moved to a
    Call Control / Voice API application.
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
        telnyx_call_sid: str | None = None,
        telnyx_account_sid: str | None = None,
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
                return await self._transfer_telnyx(
                    call_control_id=telnyx_call_control_id,
                    call_sid=telnyx_call_sid,
                    account_sid=telnyx_account_sid,
                    reason=reason,
                )
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

    async def _transfer_telnyx(
        self,
        *,
        call_control_id: str | None,
        call_sid: str | None,
        account_sid: str | None,
        reason: str,
    ) -> bool:
        # TeXML Applications should be updated with new TeXML instructions.
        # Keep Call Control mode available only as an explicit fallback.
        if self.telnyx_transfer_mode in {"call_control", "call-control", "voice_api", "voice-api"}:
            return await self._transfer_telnyx_call_control(call_control_id, reason)

        # Backward compatibility: older code may still pass the Telnyx CallSid
        # through the telnyx_call_control_id argument. For TeXML, that value is
        # treated as a CallSid if telnyx_call_sid is not provided.
        texml_call_sid = call_sid or call_control_id
        return await self._transfer_telnyx_texml(texml_call_sid, account_sid, reason)

    def _build_telnyx_transfer_texml(self, reason: str) -> str:
        timeout_secs = os.getenv("OPERATOR_DIAL_TIMEOUT_SECS", "30").strip() or "30"
        caller_id = os.getenv("TELNYX_FROM_NUMBER", "").strip()
        action_url = os.getenv("TELNYX_DIAL_ACTION_URL", "").strip()

        dial_attrs = [f'timeout="{_xml_escape(timeout_secs)}"']
        if caller_id:
            dial_attrs.append(f'callerId="{_xml_escape(caller_id)}"')
        if action_url:
            dial_attrs.append(f'action="{_xml_escape(action_url)}"')
            dial_attrs.append('method="POST"')

        attrs = " ".join(dial_attrs)
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Response>'
            f'<Dial {attrs}>'
            f'<Number>{_xml_escape(self.operator_number)}</Number>'
            '</Dial>'
            '</Response>'
        )

    async def _transfer_telnyx_texml(
        self,
        call_sid: str | None,
        account_sid: str | None,
        reason: str,
    ) -> bool:
        if not call_sid:
            print("[Escalation][Telnyx][TeXML] Missing CallSid; cannot update live TeXML call.")
            return False

        account_sid = (account_sid or os.getenv("TELNYX_ACCOUNT_SID", "")).strip()
        if not account_sid:
            print("[Escalation][Telnyx][TeXML] TELNYX_ACCOUNT_SID missing; cannot update TeXML call.")
            return False

        api_key = _clean_bearer_token(os.getenv("TELNYX_API_KEY", ""))
        if not api_key:
            print("[Escalation][Telnyx][TeXML] TELNYX_API_KEY missing.")
            return False
        if any(ch.isspace() for ch in api_key):
            print("[Escalation][Telnyx][TeXML] TELNYX_API_KEY malformed: remove spaces/quotes/Bearer prefix.")
            return False

        url = (
            "https://api.telnyx.com/v2/texml/Accounts/"
            f"{quote(account_sid, safe='')}/Calls/{quote(call_sid, safe='')}"
        )
        texml = self._build_telnyx_transfer_texml(reason)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        data = {
            # Telnyx requires CamelCase body keys for TeXML update-call.
            "Texml": texml,
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, data=data, headers=headers)
            if response.status_code >= 300:
                print(f"[Escalation][Telnyx][TeXML] Update call HTTP {response.status_code}: {response.text[:300]}")
                return False

        print(f"[Escalation][Telnyx][TeXML] Live call updated to <Dial><Number>. reason={reason}")
        return True

    async def _transfer_telnyx_call_control(self, call_control_id: str | None, reason: str) -> bool:
        if not call_control_id:
            print("[Escalation][Telnyx][CallControl] Missing call_control_id; cannot transfer live call.")
            return False

        api_key = _clean_bearer_token(os.getenv("TELNYX_API_KEY", ""))
        if not api_key:
            print("[Escalation][Telnyx][CallControl] TELNYX_API_KEY missing.")
            return False
        if any(ch.isspace() for ch in api_key):
            print("[Escalation][Telnyx][CallControl] TELNYX_API_KEY malformed: remove spaces/quotes/Bearer prefix.")
            return False

        url = f"https://api.telnyx.com/v2/calls/{quote(call_control_id, safe='')}/actions/transfer"
        payload = {
            "to": self.operator_number,
            "from": os.getenv("TELNYX_FROM_NUMBER", "").strip() or None,
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
                print(f"[Escalation][Telnyx][CallControl] Transfer HTTP {response.status_code}: {response.text[:300]}")
                return False

        print(f"[Escalation][Telnyx][CallControl] Transfer command sent. reason={reason}")
        return True
