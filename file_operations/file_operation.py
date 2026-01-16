from abc import ABC, abstractmethod
from typing import Union


class FileOperation(ABC):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def run(self):
        pass
