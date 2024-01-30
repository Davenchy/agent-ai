from typing import Any, Dict, List, Optional, Union, cast

from openai.types.chat import (ChatCompletionAssistantMessageParam,
                               ChatCompletionMessageParam,
                               ChatCompletionMessageToolCallParam,
                               ChatCompletionToolMessageParam)


class Message:
    """Represent the OpenAI API message objects.

    Inherit from this class to create your own message objects.

    Use + operator to concatenate content with other messages or string.

    Use predefined classes like: SystemMessage, UserMessage, ToolMessage
        and AssistantMessage

    Use MessagesContainer to create a collection of messages

    Examples:
        >>> container = MessagesContainer([UserMessage("Hello")])
        >>> container << AssistantMessage("Hi")
        >>> container.add(UserMessage("How are you?"))
    """

    def __init__(
        self,
        role: str,
        content: str,
        name: Optional[str] = None,
        extras: Dict[str, Any] = dict(),
    ):
        self._name = name
        self._role = role
        self._content = content
        self.extras = extras

    @property
    def has_name(self) -> bool:
        return self._name is not None

    @property
    def name(self) -> str:
        return cast(str, self._name)

    @property
    def role(self) -> str:
        return self._role

    @property
    def content(self) -> str:
        return self._content

    def as_chat_message(self, **extras: str) -> ChatCompletionMessageParam:
        """Returns JSON object to send to OpenAI API."""

        extras.update(self.extras)
        if self.has_name and 'name' not in extras:
            extras['name'] = self.name

        return cast(
            ChatCompletionMessageParam,
            {
                "role": self._role,
                "content": self._content,
                **extras,
            },
        )

    def __repr__(self):
        return f"{self.__class__.__name__}{{name: {self._name}, \
role: {self._role}, content: {self._content}}}"

    def __str__(self):
        name = f"{self.name}({self._role})" if self.has_name else self._role
        return f"{name}: {self._content}"

    def __add__(self, other):
        if type(other) is str:
            self._content += other
        elif isinstance(other, Message):
            self._content += other.content
        else:
            raise TypeError(f"Unsupported adding for type: {type(other)}")
        return self


class SystemMessage(Message):
    def __init__(self, content: str, name: Optional[str] = None,
                 extras: Dict[str, Any] = dict()):
        super().__init__("system", content, name=name, extras=extras)


class UserMessage(Message):
    def __init__(self, content: str, name: Optional[str] = None,
                 extras: Dict[str, Any] = dict()):
        super().__init__("user", content, name=name, extras=extras)


class AssistantMessage(Message):
    def __init__(self, content: str, name: Optional[str] = None,
                 tool_calls: List[ChatCompletionMessageToolCallParam] = [],
                 extras: Dict[str, Any] = dict()):
        super().__init__("assistant", content, name=name, extras=extras)
        self._tool_calls = tool_calls

    def as_chat_message(self,
                        **extras: Any) -> ChatCompletionAssistantMessageParam:
        if len(self._tool_calls) > 0:
            extras['tool_calls'] = self._tool_calls
        else:
            extras['content'] = self._content

        return cast(ChatCompletionAssistantMessageParam, {
            **super().as_chat_message(),
            'role': 'assistant',
            **extras
        })


class ToolMessage(Message):
    def __init__(self, call_id: str, content: str, name: Optional[str] = None,
                 extras: Dict[str, Any] = dict()):
        super().__init__("tool", content, name=name, extras=extras)
        self._call_id = call_id

    @ property
    def call_id(self) -> str:
        return self._call_id

    def as_chat_message(self, **extras: str) -> ChatCompletionToolMessageParam:
        return cast(
            ChatCompletionToolMessageParam,
            {
                "tool_call_id": self._call_id,
                **super().as_chat_message(**extras),
            },
        )


class MessagesContainer:
    def __init__(self, messages: List[Message]):
        self.messages = messages

    @ property
    def length(self) -> int:
        return len(self)

    def __len__(self) -> int:
        return len(self.messages)

    def add(
        self,
        *messages: Union[Message, "MessagesContainer"],
    ) -> "MessagesContainer":
        for message in messages:
            if isinstance(message, Message):
                self.messages.append(message)
            elif isinstance(message, MessagesContainer):
                self.messages.extend(message.messages)
            else:
                raise TypeError(
                    f"Unsupported adding for type: {type(message)}")
        return self

    def insert(self, index: int, *messages: Message):
        self.messages.insert(index, *messages)

    def drop(self, k: int, skip: int = 0):
        """Skip first 'skip' messages then drop the next 'k' messages"""
        self.messages = self.messages[skip: skip + k]

    def clear(self):
        """Clear all messages"""
        self.messages.clear()

    def as_chat_messages(self) -> List[ChatCompletionMessageParam]:
        return [msg.as_chat_message() for msg in self.messages]

    def get_conversation(self) -> str:
        """Use all messages to write a full conversation and return as str"""
        return "\n\n".join([str(msg) for msg in self.messages])

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.length}]({self.messages})"

    def __lshift__(self, other):
        if isinstance(other, Message):
            self.messages.append(other)
        elif isinstance(other, MessagesContainer):
            self.messages.extend(other.messages)
        else:
            raise TypeError(f"Unsupported adding for type: {type(other)}")
        return self
