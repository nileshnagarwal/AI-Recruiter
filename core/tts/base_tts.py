# Description: This file contains the abstract class for TTS (Text to Speech) module.

from abc import ABC, abstractmethod

class BaseTTS(ABC):
    @abstractmethod
    def synthesize(self, text: str, language: str = "hi-IN") -> str:
        pass
