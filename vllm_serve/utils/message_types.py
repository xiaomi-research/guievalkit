from typing import Literal, Any
from typing_extensions import TypedDict

ROLE = Literal["user", "system", "assistant"] | str


class TextContent(TypedDict):
    type: Literal["text"]
    text: str


class ImageUrlContent(TypedDict):
    type: Literal["image_url"]
    image_url: dict[Literal["url"], str | bytes]


class ImageContent(TypedDict):
    type: Literal["image"]
    image: str | Any


class StrictMessage(TypedDict):
    role: ROLE
    content: str | list[TextContent | ImageUrlContent]


class Message(TypedDict):
    role: ROLE
    content: str | list[TextContent | ImageUrlContent | ImageContent]


StrictMessages = list[StrictMessage]
Messages = list[Message]


__all__ = [
    "ROLE",
    "TextContent",
    "ImageUrlContent",
    "ImageContent",
    "StrictMessage",
    "Message",
    "Messages",
    "StrictMessages"
]
