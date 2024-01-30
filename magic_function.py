from functools import wraps
from typing import Callable, Optional, ParamSpec, TypeAlias, cast

from agent import Agent, AgentModelType
from agent.message import MessagesContainer

Params = ParamSpec("Params")
MagicFn: TypeAlias = Callable[Params, str]


def magic(
    model: Optional[AgentModelType] = None,
    temperature: float = 1.0,
    name: Optional[str] = None,
    messages: Optional[MessagesContainer] = None,
) -> Callable[[MagicFn[Params]], MagicFn[Params]]:
    """Define a magic function

    This decorator returns another decorator that converts any function into
    a magic function.

    Use this decorator on a function that returns agent instructions as a
    string.

    The generated magic function will take the arguments, execute the agent
    and returns the generated text as a string.

    Example:
        >>> @magic()
        >>> def rap_for(name: str) -> str:
        ...     return f'''You are a rap song writer
        ... Write a song for a person called {}!'''.format(name)

        >>> rap_for("Dylan")
        The generated rap song...
    """
    def decorator(func: MagicFn[Params]) -> MagicFn[Params]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> str:
            instructions = func(*args, **kwargs)
            agent = Agent(instructions, model, temperature, name, messages)
            return agent.get_output_text()

        return cast(MagicFn[Params], wrapper)

    return decorator
