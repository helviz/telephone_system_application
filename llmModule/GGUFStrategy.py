import asyncio
import os
import re
from typing import cast, Any
from llama_cpp import Llama
from llmModule.LLMStrategy import LLMStrategy
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Module-level singleton — loaded ONCE on first initialization to preserve RAM
# and prevent concurrent disk I/O bottlenecks across active sessions.
# ---------------------------------------------------------------------------
_llm_instance: Llama | None = None


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

            # 2. Match your space-separated format ("repo_id filename.gguf")
            elif " " in hf_model_env:
                repo_id, filename = hf_model_env.split(" ", 1)
                repo_id = repo_id.strip()
                filename = filename.strip()
                print(f"[GGUF] Parsed via Space-Shorthand -> Repo: {repo_id} | File: {filename}")
                return hf_hub_download(repo_id=repo_id, filename=filename)

            # 3. Fallback match for traditional slash-shorthand ("repo_id/filename.gguf")
            else:
                parts = hf_model_env.split("/")
                if len(parts) >= 3:
                    repo_id = f"{parts[0]}/{parts[1]}"
                    filename = "/".join(parts[2:])
                    print(f"[GGUF] Parsed via Slash-Shorthand -> Repo: {repo_id} | File: {filename}")
                    return hf_hub_download(repo_id=repo_id, filename=filename)
                raise ValueError(
                    "Shorthand unrecognised. Expected formats: 'repo_id filename.gguf' or 'user/repo/file.gguf'"
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
        "No model source found! Please configure either 'LOCAL_MODEL_PATH' or 'GGUF_MODEL' env profiles."
    )


def _get_llm(model_path: str | None = None) -> Llama:
    """Retrieves or instantiates the global Llama inference engine singleton."""
    global _llm_instance
    if _llm_instance is None:
        resolved_path = _resolve_model_path(model_path)
        gpu_layers = int(os.getenv("N_GPU_LAYERS", "0"))
        ctx_size = int(os.getenv("N_CTX", "2048"))

        print(f"[GGUF] Allocating engine instance (Happens ONCE): {resolved_path}")
        print(f"[GGUF] Strategy constraints -> N_CTX: {ctx_size} | N_GPU_LAYERS: {gpu_layers}")

        _llm_instance = Llama(
            model_path=resolved_path,
            n_ctx=ctx_size,
            n_gpu_layers=gpu_layers,
        )
        print("[GGUF] Engine compiled successfully and cached for active runtime sessions.")
    else:
        print("[GGUF] Reusing warm cached model instance.")
    return _llm_instance


class GGUFStrategy(LLMStrategy):
    """Local quantized model strategy implementation implementing the asynchronous stream interface."""

    def __init__(self, model_path: str = None, system_prompt: str = ""):
        super().__init__(system_prompt)
        self.llm = _get_llm(model_path)

    async def generate_stream(self, user_input: str):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]

        def get_stream():
            return self.llm.create_chat_completion(
                messages=cast(Any, messages),
                stream=True,
                max_tokens=50,
                temperature=0.4,
                top_p=0.85,
                repeat_penalty=1.30,
                stop=["<|im_end|>", "User:", "Assistant:"],
            )

        # Offloads the blocking synchronous create_chat_completion generator generator loops safely
        stream = await asyncio.to_thread(get_stream)

        print("AI (Local): ", end="", flush=True)

        for chunk in stream:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                content = delta['content']
                print(content, end="", flush=True)
                yield content

        print("\n")