import asyncio
import os
import re
from typing import Any, cast

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from llmModule.LLMStrategy import LLMStrategy

# ---------------------------------------------------------------------------
# Module-level singleton — loaded ONCE on first initialization to preserve RAM
# and prevent concurrent disk I/O bottlenecks across active sessions.
# ---------------------------------------------------------------------------
_llm_instance: Llama | None = None


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_model_path(model_path: str | None) -> str:
    """Parses environment config to obtain a valid local model file path."""
    hf_model_env = os.getenv("GGUF_MODEL", "").strip()

    if hf_model_env:
        print(f"[GGUF] GGUF_MODEL target environment variable detected: {hf_model_env}")
        try:
            # 1. Match direct Hugging Face browser URLs
            if "huggingface.co/" in hf_model_env:
                url_pattern = r"huggingface\.co/([^/]+/[^/]+)/(?:blob|resolve)/[^/]+/(.+)"
                match = re.search(url_pattern, hf_model_env)
                if match:
                    repo_id, filename = match.group(1), match.group(2)
                    print(f"[GGUF] Parsed via URL structure -> Repo: {repo_id} | File: {filename}")
                    return hf_hub_download(repo_id=repo_id, filename=filename)
                raise ValueError("Malformed Hugging Face URL structure provided.")

            # 2. Match your space-separated format: "repo_id filename.gguf"
            if " " in hf_model_env:
                repo_id, filename = hf_model_env.split(" ", 1)
                repo_id = repo_id.strip()
                filename = filename.strip()
                print(f"[GGUF] Parsed via Space-Shorthand -> Repo: {repo_id} | File: {filename}")
                return hf_hub_download(repo_id=repo_id, filename=filename)

            # 3. Fallback match for traditional slash-shorthand: "repo_id/filename.gguf"
            parts = hf_model_env.split("/")
            if len(parts) >= 3:
                repo_id = f"{parts[0]}/{parts[1]}"
                filename = "/".join(parts[2:])
                print(f"[GGUF] Parsed via Slash-Shorthand -> Repo: {repo_id} | File: {filename}")
                return hf_hub_download(repo_id=repo_id, filename=filename)

            raise ValueError(
                "Shorthand unrecognised. Expected formats: "
                "'repo_id filename.gguf', 'user/repo/file.gguf', or a Hugging Face URL."
            )

        except Exception as e:
            print(f"[GGUF] ❌ Failed parsing or downloading Hugging Face asset: {e}")
            if model_path:
                print(f"[GGUF] Falling back to standard fallback directory: {model_path}")
                return model_path
            raise

    if model_path:
        return model_path

    raise ValueError(
        "No model source found! Please configure either LOCAL_MODEL_PATH or GGUF_MODEL."
    )


def _get_llm(model_path: str | None = None) -> Llama:
    """Retrieves or instantiates the global Llama inference engine singleton."""
    global _llm_instance
    if _llm_instance is None:
        resolved_path = _resolve_model_path(model_path)
        gpu_layers = int(os.getenv("N_GPU_LAYERS", "0"))
        ctx_size = int(os.getenv("LLM_N_CTX", "768"))

        print(f"[GGUF] Allocating engine instance (happens once): {resolved_path}")
        print(f"[GGUF] Strategy constraints -> N_CTX: {ctx_size} | N_GPU_LAYERS: {gpu_layers}")

        _llm_instance = Llama(
            model_path=resolved_path,
            n_ctx=ctx_size,
            n_gpu_layers=gpu_layers,
            n_batch=512,
            n_threads=os.cpu_count() or 4,
            verbose=False,
        )
        print("[GGUF] Engine compiled successfully and cached for active runtime sessions.")
    else:
        print("[GGUF] Reusing warm cached model instance.")
    return _llm_instance


def _force_no_think_prompt(prompt: str) -> str:
    """
    Qwen3 GGUF models often obey /no_think at prompt level.
    This is a fallback for llama.cpp builds that do not support chat_template_kwargs.
    """
    prompt = (prompt or "").strip()
    lower_prompt = prompt.lower()

    if "/no_think" in lower_prompt:
        return prompt

    no_think_rules = (
        "/no_think\n"
        "Do not output hidden reasoning. "
        "Do not output <think> or </think>. "
        "Give only the final spoken answer.\n\n"
    )
    return no_think_rules + prompt


class ThinkTagStreamFilter:
    """
    Removes streamed <think>...</think> blocks without leaking partial tags to TTS.

    Streaming chunks can split tags like '<thi' + 'nk>', so regex on each chunk is
    not enough. This filter keeps a small tail buffer to catch split tags safely.
    """

    def __init__(self, tail_size: int = 32):
        self.tail_size = tail_size
        self.buffer = ""
        self.inside_think = False

    def feed(self, text: str) -> str:
        if not text:
            return ""

        self.buffer += text
        output_parts: list[str] = []

        while self.buffer:
            lower_buffer = self.buffer.lower()

            if self.inside_think:
                end_idx = lower_buffer.find("</think>")
                if end_idx == -1:
                    # Keep only enough tail to detect a split closing tag later.
                    self.buffer = self.buffer[-self.tail_size:]
                    return "".join(output_parts)

                self.buffer = self.buffer[end_idx + len("</think>"):]
                self.inside_think = False
                continue

            start_idx = lower_buffer.find("<think>")
            if start_idx == -1:
                # Emit all except the tail because a future chunk may complete '<think>'.
                safe_len = max(0, len(self.buffer) - self.tail_size)
                if safe_len:
                    output_parts.append(self.buffer[:safe_len])
                    self.buffer = self.buffer[safe_len:]
                return "".join(output_parts)

            output_parts.append(self.buffer[:start_idx])
            self.buffer = self.buffer[start_idx + len("<think>"):]
            self.inside_think = True

        return "".join(output_parts)

    def flush(self) -> str:
        if self.inside_think:
            self.buffer = ""
            return ""

        remaining = self.buffer
        self.buffer = ""
        # Remove broken tags just in case generation stopped midway.
        remaining = re.sub(r"</?think[^>]*>", "", remaining, flags=re.IGNORECASE)
        return remaining.strip()


class GGUFStrategy(LLMStrategy):
    """Local quantized model strategy implementing the asynchronous stream interface."""

    def __init__(self, model_path: str | None = None, system_prompt: str = ""):
        super().__init__(system_prompt)
        self.llm = _get_llm(model_path)
        self.disable_thinking = _env_bool("QWEN_DISABLE_THINKING", True)

    async def generate_stream(self, user_input: str):
        system_prompt = self.system_prompt
        if self.disable_thinking:
            system_prompt = _force_no_think_prompt(system_prompt)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
            # Pre-fill empty think block so the model skips thinking entirely
            # and starts generating real content immediately.
            # Required because chat_template_kwargs is unsupported on this build.
            {"role": "assistant", "content": "<think>\n\n</think>\n\n"},
        ]

        generation_kwargs: dict[str, Any] = {
            "messages": cast(Any, messages),
            "stream": True,
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "120")),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.35")),
            "top_p": float(os.getenv("LLM_TOP_P", "0.85")),
            "top_k": int(os.getenv("LLM_TOP_K", "20")),
            "repeat_penalty": float(os.getenv("LLM_REPEAT_PENALTY", "1.10")),
            "stop": [
                "<|im_end|>",
                "<think>",  # hard stop if thinking starts
                "User:",
                "user:",
                "\nUser:",
                "\nuser:",
                "Assistant:",
                "assistant:",
            ],
        }

        if self.disable_thinking:
            # Supported by newer llama-cpp-python / llama.cpp Qwen chat templates.
            generation_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

        def get_stream():
            try:
                return self.llm.create_chat_completion(**generation_kwargs)
            except TypeError as e:
                # Older llama-cpp-python may not accept chat_template_kwargs.
                if "chat_template_kwargs" not in str(e):
                    raise
                generation_kwargs.pop("chat_template_kwargs", None)
                print("[GGUF] chat_template_kwargs unsupported; falling back to /no_think prompt only.")
                return self.llm.create_chat_completion(**generation_kwargs)

        # Offload the blocking synchronous create_chat_completion call safely.
        stream = await asyncio.to_thread(get_stream)

        print("AI (Local): ", end="", flush=True)
        think_filter = ThinkTagStreamFilter()

        for chunk in stream:
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            if not content:
                continue

            clean_content = think_filter.feed(content) if self.disable_thinking else content
            if clean_content:
                print(clean_content, end="", flush=True)
                yield clean_content

        final_content = think_filter.flush() if self.disable_thinking else ""
        if final_content:
            print(final_content, end="", flush=True)
            yield final_content

        print("\n")