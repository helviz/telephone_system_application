import asyncio
from llama_cpp import Llama
from llmModule.LLMStrategy import LLMStrategy


class GGUFStrategy(LLMStrategy):
    def __init__(self, model_path: str, system_prompt: str):
        super().__init__(system_prompt)
        # n_ctx is the context window. Qwen models usually support large windows.
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=0  # Set to -1 if you have a GPU, 0 for CPU-only
        )

    async def generate_stream(self, user_input: str):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]

        def get_stream():
            return self.llm.create_chat_completion(
                messages=messages,
                stream=True
            )

        stream = await asyncio.to_thread(get_stream)

        print("AI (Local): ", end="", flush=True)

        for chunk in stream:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                content = delta['content']
                # 1. Immediate terminal feedback
                print(content, end="", flush=True)

                # 2. Yield to the TTS sentence buffer
                yield content

        print("\n")