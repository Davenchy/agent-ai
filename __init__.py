import sys
from io import TextIOWrapper
from typing import (Dict, Generator, List, Literal, Optional, TextIO, Union,
                    cast)


from .message import (AssistantMessage, Message, MessagesContainer,
                      SystemMessage, ToolMessage, UserMessage)
from .ability import Ability
from ._tool_chunks_container import ToolChunksContainer

from openai import chat, Stream
from openai._types import NotGiven
from openai.types.chat import ChatCompletionChunk, ChatCompletionToolParam


AgentModelType = Literal[
    'gpt-4-1106-preview',
    "gpt-4-vision-preview",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-16k-0613",
]


class Agent:
    """AI agent that will automate your OpenAI text generation tasks

    Use << operator to inject messages and abilities

    Execute the Agent instance as an alias for the `generate` method

    Examples:
        # create agent with instructions, agent name is optional
        >>> agent = Agent("You are a helpful assistant.")

        # add user input
        >>> agent << "How are you?"
        # OR use a UserMessage, name is optional
        >>> agent << UserMessage("How are you?")

        # stream output to a file
        >>> agent >> sys.stdout
        # OR, stream_to file parameter is optional, it is stdout by default
        >>> agent.stream_to()

        # write response to a file
        >>> with open('file', 'w') as f:
        ...    agent >> f
        ...    # OR
        ...    agent.stream_to(f)


        # stream_to uses generate under the hood
        # to manually stream response use the `generate` method
        >>> for chunk in agent.generate():
        ...     print(chunk, end="", flush=True)



        # Let's define a basic ability
        >>> @Ability.create()
        >>> def ability() -> str:
        ...     '''My ability'''
        ...     return "Hello"

        # assign the ability to the agent
        >>> agent << ability
        # OR
        >>> agent.add_abilities(ability)
    """

    def __init__(self, instructions: str,
                 model: Optional[AgentModelType] = None,
                 temperature: float = 1,
                 name: Optional[str] = None,
                 messages: Optional[MessagesContainer] = None,
                 **instructions_variables: str):
        self.messages = MessagesContainer([
            SystemMessage(instructions.format(**instructions_variables))
        ])
        if messages is not None:
            self.messages.add(messages)

        self._abilities: Dict[str, Ability] = dict()
        self._name = name
        self._model: AgentModelType = model or "gpt-3.5-turbo-1106"
        self.temperature = temperature

    @property
    def model(self) -> str:
        return self._model

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def name(self) -> str:
        return cast(str, self._name)

    @property
    def has_name(self) -> bool:
        return self._name is not None

    @property
    def abilities(self) -> List[str]:
        return list(self._abilities.keys())

    @temperature.setter
    def temperature(self, value: float):
        self._temperature = max(min(value, 2), 0)

    def add_abilities(self, *abilities: Ability):
        """Add abilities to this agent"""
        for ability in abilities:
            self._abilities[ability.name] = ability

    def add_messages(self, *messages: Message):
        """Add messages to this agent"""
        self.messages.add(*messages)

    def _execute_ability(self, name: str, *args, **kwargs) -> str:
        """Execute registered ability, with error handling"""
        # check if ability is registered otherwise return failed message
        if name not in self._abilities:
            return f"Unknown ability: {name}"

        # get the ability and execute then return the result
        ability = self._abilities[name]
        try:
            return ability(*args, **kwargs)
        except Exception as e:
            # handle any kind of errors and return as a fail message
            return f"Error: {e}"

    def _stream_chunks(self, stream: Stream[ChatCompletionChunk],
                       ) -> Generator[str, None, None]:
        """Load and parse OpenAI API streamed data also execute
        abilities(tools) on need"""

        # memorize received function calls between chunks
        container = ToolChunksContainer()
        assistant_response = str()

        # check each chunk and yield it if needed
        for chunk in stream:
            delta = chunk.choices[0].delta

            # if any assistant response was received
            if delta.content is not None:
                yield delta.content
                assistant_response += delta.content

            # if any tool call was received
            if delta.tool_calls is not None:
                container.collect_tool_calls(*delta.tool_calls)

        # if any assistant message was received append it to messages
        if assistant_response:
            msg = AssistantMessage(assistant_response, name=self._name)
            self.messages.add(msg)

        # check if container collected any builders then build them
        if container.has_builders:
            container.build_tool_calls()

        # if container has collected builders and built them then generate
        # responses
        if container.has_calls:
            # generate assistant response for calls
            msg = container.generate_assistant_message()
            self.messages.add(msg)

            # generate response for each tool call
            for call in container.calls:
                # get the response for the call using specified ability
                content = self._execute_ability(call.name, **call.args)
                # add the assistant response for that call
                msg = ToolMessage(call.call_id, content)
                self.messages.add(msg)

            # send responses and enter a new round of generation
            yield from self.generate()

    def stream(self, file: Union[TextIO, str, None] = None):
        """Generate and stream chunks to a file.
        The file could be a TextIO object, a string (file path) or None.
        If the file is None, then the stream will be written to stdout.
        """

        # if file path was passed, then open it for writing
        is_opened = False
        if type(file) is str:
            file = open(file, 'w')
            is_opened = True

        # if no file or file path was passed use stdout as a default file
        if file is None:
            file = sys.stdout

        # generate chunks
        for chunk in self.generate():
            file.write(chunk)
            file.flush()
        file.write("\n")
        file.flush()

        # if file was opened by the function then close it
        if is_opened:
            file.close()

    def get_output_text(self) -> str:
        """Request generation and write all chunks into a string then return"""
        return "".join(self.generate())

    def __call__(self,
                 user_message: Optional[str] = None
                 ) -> Generator[str, None, None]:
        """Alias for the generate method"""
        if user_message is not None:
            self.messages.add(UserMessage(user_message))
        return self.generate()

    def generate(self) -> Generator[str, None, None]:
        """Send request to OpenAI API and get response as a chunks of text"""
        response = chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=self.messages.as_chat_messages(),
            stream=True,
            tools=[
                cast(ChatCompletionToolParam, tool.to_json())
                for tool in self._abilities.values()
            ] if len(self._abilities) > 0 else NotGiven(),
        )

        yield from self._stream_chunks(response)

    def __lshift__(self, other):
        if isinstance(other, Message) or isinstance(other, MessagesContainer):
            self.messages.add(other)
        elif isinstance(other, Ability):
            self._abilities[other.name] = other
        elif Ability.has_ability(other):
            ability = Ability.get_ability(other)
            self._abilities[ability.name] = ability
        elif type(other) is str:
            self.messages.add(UserMessage(other))
        else:
            raise TypeError(f"Unsupported requesting for type: {type(other)}")
        return self

    def __rshift__(self, other):
        if isinstance(other, TextIOWrapper):
            self.stream(other)
        else:
            raise TypeError(f"Unsupported requesting for type: {type(other)}")
        return self
