from abc import ABC, abstractmethod
class LLMStrategy(ABC):
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt

    @abstractmethod
    async def generate_stream(self, user_input: str):
        pass