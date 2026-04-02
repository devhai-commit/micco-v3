"""
LLM Provider Types
==================
Shared data classes for the multi-provider LLM abstraction.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LLMResult:
    """Result from an LLM call, optionally including thinking text."""
    content: str
    thinking: str = ""


@dataclass
class LLMImagePart:
    """An image attachment for an LLM message."""
    data: bytes
    mime_type: str = "image/png"


@dataclass
class LLMMessage:
    """A single message in a conversation."""
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    images: list[LLMImagePart] = field(default_factory=list)
    # For OpenAI native tool calling: the tool_call_id when role="tool"
    tool_call_id: str | None = field(default=None, repr=False)
    # For OpenAI native tool calling: the tool_calls list when role="assistant"
    tool_calls: list[dict] | None = field(default=None, repr=False)
    # Opaque provider-specific content (e.g. Gemini Content with
    # thought_signature).  When set, providers should use this directly
    # instead of building from ``content``/``images``.
    _raw_provider_content: object | None = field(default=None, repr=False)


@dataclass
class StreamChunk:
    """A single chunk from streaming LLM output."""
    type: str  # "text" | "thinking" | "function_call"
    text: str = ""
    function_call: dict | None = None  # {"name": str, "args": dict}
