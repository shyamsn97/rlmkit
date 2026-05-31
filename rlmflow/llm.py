from __future__ import annotations

import abc
from collections.abc import Iterator
from dataclasses import dataclass

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

# Transient HTTP / streaming faults from the OpenAI / Anthropic client
# stack. Matched by class name so this module doesn't have to import
# httpx, httpcore, openai, or anthropic at module load.
RETRYABLE_EXC_NAMES = frozenset(
    {
        "APIConnectionError",
        "APIError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
        "RemoteProtocolError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "ReadError",
    }
)


def is_retryable(exc: BaseException) -> bool:
    if type(exc).__name__ in RETRYABLE_EXC_NAMES:
        return True
    cause = exc.__cause__
    return cause is not None and type(cause).__name__ in RETRYABLE_EXC_NAMES


retry_transient = retry(
    retry=retry_if_exception(is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    reraise=True,
)


@dataclass
class LLMUsage:
    """Token counts from a single LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(metaclass=abc.ABCMeta):
    last_usage: LLMUsage | None = None
    thread_safe: bool = False

    @abc.abstractmethod
    def chat(self, messages: list[dict[str, str]], *args, **kwargs) -> str:
        """Send messages and return the full response."""

    def stream(self, messages: list[dict[str, str]], *args, **kwargs) -> Iterator[str]:
        """Yield response token-by-token. Override for real streaming.

        Default falls back to chat() and yields the whole thing at once.
        """
        yield self.chat(messages, *args, **kwargs)

    def completion(
        self, messages: list[dict[str, str]], *args, **kwargs
    ) -> tuple[str, LLMUsage]:
        """Return response text and usage for one request.

        The default adapter preserves the existing ``stream`` / ``last_usage``
        contract. Shared schedulers should guard this method for clients that
        do not override it, because ``last_usage`` is mutable client state.
        """
        text = "".join(self.stream(messages, *args, **kwargs))
        return text, self.last_usage or LLMUsage()


class OpenAIClient(LLMClient):
    """OpenAI-compatible client. Requires `pip install openai`."""

    thread_safe = True

    def __init__(self, model: str = "gpt-4o", **client_kwargs) -> None:
        from openai import OpenAI

        self.client = OpenAI(**client_kwargs)
        self.model = model

    def chat(self, messages: list[dict[str, str]], *args, **kwargs) -> str:
        text, _usage = self.completion(messages, *args, **kwargs)
        return text

    @retry_transient
    def completion(
        self, messages: list[dict[str, str]], *args, **kwargs
    ) -> tuple[str, LLMUsage]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        usage = LLMUsage()
        if resp.usage:
            usage = LLMUsage(
                input_tokens=resp.usage.prompt_tokens or 0,
                output_tokens=resp.usage.completion_tokens or 0,
            )
            self.last_usage = usage
        return resp.choices[0].message.content or "", usage

    def stream(self, messages: list[dict[str, str]], *args, **kwargs) -> Iterator[str]:
        # Buffer until the stream is fully consumed before yielding any
        # tokens, so tenacity can safely retry transient mid-stream
        # drops without double-emitting partial output. Real-time
        # streaming is sacrificed for correctness on retry.
        yield from self.collect_stream(messages)

    @retry_transient
    def collect_stream(self, messages: list[dict[str, str]]) -> list[str]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks: list[str] = []
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
            if getattr(chunk, "usage", None):
                self.last_usage = LLMUsage(
                    input_tokens=chunk.usage.prompt_tokens or 0,
                    output_tokens=chunk.usage.completion_tokens or 0,
                )
        return chunks


class AnthropicClient(LLMClient):
    """Anthropic client. Requires `pip install anthropic`."""

    thread_safe = True

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 8192,
        **client_kwargs,
    ) -> None:
        import anthropic

        self.client = anthropic.Anthropic(**client_kwargs)
        self.model = model
        self.max_tokens = max_tokens

    def split_messages(self, messages: list[dict[str, str]]) -> tuple[str, list[dict]]:
        system = ""
        chat_msgs = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_msgs.append(m)
        return system, chat_msgs

    def chat(self, messages: list[dict[str, str]], *args, **kwargs) -> str:
        text, _usage = self.completion(messages, *args, **kwargs)
        return text

    @retry_transient
    def completion(
        self, messages: list[dict[str, str]], *args, **kwargs
    ) -> tuple[str, LLMUsage]:
        system, chat_msgs = self.split_messages(messages)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=chat_msgs,
        )
        usage = LLMUsage(
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )
        self.last_usage = usage
        return resp.content[0].text, usage

    def stream(self, messages: list[dict[str, str]], *args, **kwargs) -> Iterator[str]:
        yield from self.collect_stream(messages)

    @retry_transient
    def collect_stream(self, messages: list[dict[str, str]]) -> list[str]:
        system, chat_msgs = self.split_messages(messages)
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=chat_msgs,
        ) as s:
            chunks = list(s.text_stream)
            msg = s.get_final_message()
            self.last_usage = LLMUsage(
                input_tokens=msg.usage.input_tokens,
                output_tokens=msg.usage.output_tokens,
            )
            return chunks
