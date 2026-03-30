import abc
from collections.abc import Iterator


class LLMClient(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def chat(self, messages: list[dict[str, str]], *args, **kwargs) -> str:
        """Send messages and return the full response."""

    def stream(self, messages: list[dict[str, str]], *args, **kwargs) -> Iterator[str]:
        """Yield response token-by-token. Override for real streaming.

        Default falls back to chat() and yields the whole thing at once.
        """
        yield self.chat(messages, *args, **kwargs)
