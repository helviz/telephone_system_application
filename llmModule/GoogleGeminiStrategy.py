from google import genai
from google.genai import types
from llmModule.LLMStrategy import LLMStrategy


class GeminiStrategy(LLMStrategy):
    def __init__(self, api_key: str, system_prompt: str, model_name="gemini-2.0-flash-lite"):
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
        print("AI: ", end="", flush=True)

        response = self.chat.send_message_stream(user_input)

        for chunk in response:
            if chunk.text:
                print(chunk.text, end="", flush=True)
                yield chunk.text

        print("\n")