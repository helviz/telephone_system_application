import asyncio
from google import genai
from google.genai import types
from llmModule.LLMStrategy import LLMStrategy


class GeminiStrategy(LLMStrategy):
    def __init__(self, api_key: str, system_prompt: str, model_name="gemma-4-26b-a4b-it"):
        super().__init__(system_prompt)

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.system_prompt = system_prompt

        self.chat = self.client.chats.create(
            model=model_name,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt
            )
        )

    async def generate_stream(self, user_input: str):
        """
        FIX 1: send_message_stream() is a blocking call that iterates a
        synchronous generator. Running it directly inside an async generator
        blocks the event loop for the entire duration of the response,
        freezing all other coroutines (STT, WebSocket receiver, etc.).

        Solution: collect all chunks in a thread via run_in_executor, then
        yield them back on the event loop. This keeps the loop free during
        the blocking network I/O.
        """
        print("AI: ", end="", flush=True)

        loop = asyncio.get_running_loop()

        def _collect_chunks() -> list[str]:
            chunks = []
            for chunk in self.chat.send_message_stream(user_input):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    chunks.append(chunk.text)
            return chunks

        chunks = await loop.run_in_executor(None, _collect_chunks)

        for chunk in chunks:
            yield chunk

        print("\n")