from __future__ import annotations

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


class OpenAIClient(LLMClient):
    """OpenAI-compatible client. Requires `pip install openai`."""

    def __init__(self, model: str = "gpt-4o", **client_kwargs) -> None:
        from openai import OpenAI

        self.client = OpenAI(**client_kwargs)
        self.model = model

    def chat(self, messages: list[dict[str, str]], *args, **kwargs) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return resp.choices[0].message.content or ""

    def stream(self, messages: list[dict[str, str]], *args, **kwargs) -> Iterator[str]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicClient(LLMClient):
    """Anthropic client. Requires `pip install anthropic`."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        **client_kwargs,
    ) -> None:
        import anthropic

        self.client = anthropic.Anthropic(**client_kwargs)
        self.model = model
        self.max_tokens = max_tokens

    def _split_messages(self, messages: list[dict[str, str]]) -> tuple[str, list[dict]]:
        system = ""
        chat_msgs = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_msgs.append(m)
        return system, chat_msgs

    def chat(self, messages: list[dict[str, str]], *args, **kwargs) -> str:
        system, chat_msgs = self._split_messages(messages)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=chat_msgs,
        )
        return resp.content[0].text

    def stream(self, messages: list[dict[str, str]], *args, **kwargs) -> Iterator[str]:
        system, chat_msgs = self._split_messages(messages)
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=chat_msgs,
        ) as s:
            yield from s.text_stream
