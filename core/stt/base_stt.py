# Description: This file contains the abstract class for the Speech to Text (STT) module.

from abc import ABC, abstractmethod

class BaseSTT(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        pass
