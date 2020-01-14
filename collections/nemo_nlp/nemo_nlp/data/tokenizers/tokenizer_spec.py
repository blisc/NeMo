from abc import abstractmethod, ABC
from typing import List


class TokenizerSpec(ABC):
    """
    Text should be a string. Tokens should be a list of strings where each
    element represents a wordpiece. Ids should be a list of ints where each
    element represents a wordpiece.
    """
    @abstractmethod
    def text_to_tokens(self, text):
        pass

    @abstractmethod
    def tokens_to_text(self, tokens):
        pass

    @abstractmethod
    def tokens_to_ids(self, tokens):
        pass

    @abstractmethod
    def ids_to_tokens(self, ids):
        pass

    @abstractmethod
    def text_to_ids(self, text):
        pass

    @abstractmethod
    def ids_to_text(self, ids):
        pass

    def add_special_tokens(self, special_tokens: List[str]):
        pass
