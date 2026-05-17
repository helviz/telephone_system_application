import asyncio
import os
import re
from typing import cast, Any
from llama_cpp import Llama
from llmModule.LLMStrategy import LLMStrategy
from huggingface_hub import hf_hub_download


class GGUFStrategy(LLMStrategy):
    def __init__(self, model_path: str = None, system_prompt: str = ""):
        super().__init__(system_prompt)

        # 1. Fetch configurations directly from Hugging Face Space Variables/Secrets
        hf_model_env = os.getenv("GGUF_MODEL", "").strip()
        gpu_layers = int(os.getenv("N_GPU_LAYERS", "0"))  # 0 for CPU, -1 for GPU
        ctx_size = int(os.getenv("N_CTX", "2048"))

        resolved_path = None

        # 2. Determine model source: HF link or local fallback
        if hf_model_env:
            print(f"HF_MODEL variable detected: {hf_model_env}")
            try:
                # Option A: Full browser URL (blob or resolve links)
                if "huggingface.co/" in hf_model_env:
                    url_pattern = r"huggingface\.co/([^/]+/[^/]+)/(?:blob|resolve)/[^/]+/(.+)"
                    match = re.search(url_pattern, hf_model_env)
                    if match:
                        repo_id = match.group(1)
                        filename = match.group(2)
                        print(f"Parsed HF URL -> Repo: {repo_id} | File: {filename}")
                        resolved_path = hf_hub_download(repo_id=repo_id, filename=filename)
                    else:
                        raise ValueError("Could not parse Hugging Face URL structure.")

                # Option B: Shorthand notation (e.g., 'username/repo/file.gguf')
                else:
                    parts = hf_model_env.split("/")
                    if len(parts) >= 3:
                        repo_id = f"{parts[0]}/{parts[1]}"
                        filename = "/".join(parts[2:])
                        print(f"Parsed HF Shorthand -> Repo: {repo_id} | File: {filename}")
                        resolved_path = hf_hub_download(repo_id=repo_id, filename=filename)
                    else:
                        raise ValueError("Shorthand must match 'username/repo_name/filename.gguf' format.")
            except Exception as e:
                print(f"❌ Error downloading from Hugging Face: {e}")
                if model_path:
                    print("Falling back to local model path...")
                    resolved_path = model_path
                else:
                    raise e
        else:
            resolved_path = model_path

        if not resolved_path:
            raise ValueError(
                "No model specified! Please configure the 'HF_MODEL' environment variable/secret "
                "in your Hugging Face Space settings."
            )

        # 3. Initialize the model
        print(f"Loading GGUF model from: {resolved_path} (GPU layers: {gpu_layers})")
        self.llm = Llama(
            model_path=resolved_path,
            n_ctx=ctx_size,
            n_gpu_layers=gpu_layers
        )

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