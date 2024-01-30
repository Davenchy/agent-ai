import json
from typing import Any, Dict, List

from .message import AssistantMessage

from openai.types.chat import ChatCompletionMessageToolCallParam
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from openai.types.chat.chat_completion_message_tool_call_param import (
    Function as OpenAIFunction)


class ToolFunctionCall:
    """This object represents a tool function call request from the API
    contains important information like the callId, name and arguments as
    a stringified json object

    Use to generate the function object according to the API scheme.
    Use to generate the tool_call object according to the API scheme."""

    def __init__(self, call_id: str, name: str, arguments: str):
        self._call_id = call_id
        self._name = name
        self._arguments_str = arguments

    @property
    def call_id(self) -> str:
        return self._call_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def args(self) -> Dict[str, Any]:
        return json.loads(self._arguments_str)

    @property
    def args_str(self) -> str:
        return self._arguments_str

    @property
    def to_function(self) -> OpenAIFunction:
        return {
            'name': self._name,
            'arguments': self._arguments_str,
        }

    def generate_tool_call(self) -> ChatCompletionMessageToolCallParam:
        return {
            'id': self._call_id,
            'function': self.to_function,
            'type': 'function',
        }

    def __repr__(self) -> str:
        return "ToolFunctionCall{{call_id: {}, name: {}, arguments: {}".format(
            self._call_id,
            self._name,
            self._arguments_str,
        )


class ToolFunctionCallBuilder:
    """A builder object for ToolFunctionCall object.
    Used to collect tool function: callId, name and arguments in a single
    object between streamed chunks"""

    call_id: str = ""
    """The ID of the tool call."""

    name: str = ""
    """The name of the function to call."""

    arguments: str = ""
    """The stringified json object represents the function call arguments"""

    def append_delta(self, delta: ChoiceDeltaToolCall):
        """Appends the chunk delta data to already collected data"""
        if delta.id:
            self.call_id += delta.id
        if delta.function:
            fn = delta.function
            if fn.name:
                self.name += fn.name
            if fn.arguments:
                self.arguments += fn.arguments

    def build(self) -> ToolFunctionCall:
        """Use the current configuration to build a ToolFunctionCall object"""
        return ToolFunctionCall(self.call_id, self.name, self.arguments)

    def __repr__(self) -> str:
        return "ToolFunctionCallBuilder(call_id: {}, name: {}, arguments: {})\
".format(
            self.call_id,
            self.name,
            self.arguments)


class ToolChunksContainer:
    """Used to trace tool function calls data between chunks for streams API
    requests then generates a ToolFunctionCall object for each call to ease
    tool calls management.

    Also could generate repeated AssistantMessage for tool_calls which stored
    in the messages list and send as a message before the tool response message
    also generates the response Message for each call.

    Usage:
        - Use `collect_tool_calls` method between chunks to collect delta data
        - Use `build_tool_calls` method to generate ToolFunctionCall objects
        - Get the generated ToolFunctionCall objects from the attribute `calls`
        - Use `generate_assistant_message` to generate assistant message as
            a response for the tool_calls request.
            This method uses the `calls` attribute, so you need to build them
            first (if you didn't as mentioned in the previous steps)."""

    def __init__(self):
        self.builders: List[ToolFunctionCallBuilder] = list()
        self.calls: List[ToolFunctionCall] = list()

    @property
    def has_calls(self) -> bool:
        """Check if any tool calls were build"""
        return len(self.calls) > 0

    @property
    def has_builders(self) -> bool:
        """Check if any tool calls where received"""
        return len(self.builders) > 0

    def _request_builders(
            self, call: ChoiceDeltaToolCall) -> ToolFunctionCallBuilder:
        # if no builder for call index then create one
        if len(self.builders) <= call.index:
            self.builders.append(ToolFunctionCallBuilder())
        # return builder by index
        return self.builders[call.index]

    def collect_tool_calls(self, *tool_calls: ChoiceDeltaToolCall):
        """Use between chunks to collect delta data"""
        for call in tool_calls:
            # request builder
            builder = self._request_builders(call)
            # append received pieces
            builder.append_delta(call)

    def build_tool_calls(self):
        """Clear the `calls` attribute and use collected builders which stored
        in `builders` attribute to generate ToolFunctionCall objects and store
        them in `calls`"""
        self.calls = [builder.build() for builder in self.builders]

    def generate_assistant_message(self) -> AssistantMessage:
        """Use the generated ToolFunctionCall objects by `build_tool_calls`
        which stored in `calls` attribute to generate AssistantMessage object
        """
        return AssistantMessage("", tool_calls=[
            call.generate_tool_call()
            for call in self.calls
        ])
