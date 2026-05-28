import asyncio
import os
import re
from typing import cast, Any
from llama_cpp import Llama
from llmModule.LLMStrategy import LLMStrategy
from huggingface_hub import hf_hub_download


# ---------------------------------------------------------------------------
# Module-level singleton — the model is loaded ONCE when this module is first
# imported and reused across every WebSocket session.  Loading 2+ GiB on every
# call wastes ~30-60 s of startup time and doubles RAM usage.
# ---------------------------------------------------------------------------
_llm_instance: Llama | None = None


def _resolve_model_path(model_path: str | None) -> str:
    hf_model_env = os.getenv("GGUF_MODEL", "").strip()

    if hf_model_env:
        print(f"[GGUF] HF_MODEL variable detected: {hf_model_env}")
        try:
            if "huggingface.co/" in hf_model_env:
                url_pattern = r"huggingface\.co/([^/]+/[^/]+)/(?:blob|resolve)/[^/]+/(.+)"
                match = re.search(url_pattern, hf_model_env)
                if match:
                    repo_id, filename = match.group(1), match.group(2)
                    print(f"[GGUF] Parsed HF URL -> Repo: {repo_id} | File: {filename}")
                    return hf_hub_download(repo_id=repo_id, filename=filename)
                raise ValueError("Could not parse Hugging Face URL structure.")
            else:
                parts = hf_model_env.split("/")
                if len(parts) >= 3:
                    repo_id = f"{parts[0]}/{parts[1]}"
                    filename = "/".join(parts[2:])
                    print(f"[GGUF] Parsed HF Shorthand -> Repo: {repo_id} | File: {filename}")
                    return hf_hub_download(repo_id=repo_id, filename=filename)
                raise ValueError("Shorthand must match 'username/repo_name/filename.gguf' format.")
        except Exception as e:
            print(f"[GGUF] ❌ Error downloading from Hugging Face: {e}")
            if model_path:
                print("[GGUF] Falling back to local model path...")
                return model_path
            raise

    if model_path:
        return model_path

    raise ValueError(
        "No model specified! Please configure the 'GGUF_MODEL' environment variable/secret "
        "in your Hugging Face Space settings."
    )


def _get_llm(model_path: str | None = None) -> Llama:
    """Return the shared Llama instance, creating it on the first call only."""
    global _llm_instance
    if _llm_instance is None:
        resolved_path = _resolve_model_path(model_path)
        gpu_layers = int(os.getenv("N_GPU_LAYERS", "0"))
        ctx_size = int(os.getenv("N_CTX", "2048"))
        print(f"[GGUF] Loading model (this happens ONCE): {resolved_path} | GPU layers: {gpu_layers}")
        _llm_instance = Llama(
            model_path=resolved_path,
            n_ctx=ctx_size,
            n_gpu_layers=gpu_layers,
        )
        print("[GGUF] Model loaded and cached — will be reused for all sessions.")
    else:
        print("[GGUF] Reusing cached model instance.")
    return _llm_instance


class GGUFStrategy(LLMStrategy):
    def __init__(self, model_path: str = None, system_prompt: str = ""):
        super().__init__(system_prompt)
        # Grab (or create) the shared singleton — no per-call disk I/O
        self.llm = _get_llm(model_path)

    async def generate_stream(self, user_input: str):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]

        def get_stream():
            return self.llm.create_chat_completion(
                messages=cast(Any, messages),  # Suppresses type checker warning cleanly
                stream=True
            )

        stream = await asyncio.to_thread(get_stream)

        print("AI (Local): ", end="", flush=True)

        for chunk in stream:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                content = delta['content']
                print(content, end="", flush=True)
                yield content

        print("\n")