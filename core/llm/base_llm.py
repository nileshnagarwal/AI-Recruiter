# Description: This file contains the abstract class for the LLM model.

from abc import ABC, abstractmethod

# core/llm/base_llm.py
class BaseLLM(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, history: list) -> str:
        pass
