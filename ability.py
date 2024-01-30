import inspect
import importlib
from traceback import extract_stack, FrameSummary
from typing import Callable, List, Optional, Union, Dict, Any, NoReturn, cast


AbilityFn = Callable[..., str]


class AbilityCreationError(Exception):
    def __init__(self, func_name: str, frame: FrameSummary, *args: object):
        super().__init__(*args)
        self._func_name = func_name
        self._frame = frame

    def __str__(self):
        frame, name = self._frame, self._func_name
        reason = "\n".join(self.args)
        return f"""Ability Creation Error: Failed to create ability by name \
'{name}':
In File: {frame.filename}: {name}: {frame.lineno}
Reason: {reason}"""

    @staticmethod
    def factory(func_name: str, frame: FrameSummary):
        def raiser(*args: object) -> NoReturn:
            raise AbilityCreationError(func_name, frame, *args)
        return raiser


class AbilityArgument:
    def __init__(self, name: str, type: str, description: str,
                 is_required: bool = False,
                 values: Optional[List[str]] = None):
        self._name = name
        self._type = type
        self._description = description
        self._is_required = is_required
        self._values = values

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> str:
        return self._type

    @property
    def description(self) -> str:
        return self._description

    @property
    def is_required(self) -> bool:
        return self._is_required

    @property
    def values(self) -> Optional[List[str]]:
        return self._values

    def to_json(self) -> Dict[str, Any]:
        json: Dict[str, Any] = {
            "type": self._type,
            "description": self._description,
        }

        if self._values is not None:
            json["enum"] = self._values
        return json


class Ability:
    def __init__(self,
                 func: AbilityFn,
                 name: str, description: str,
                 *arguments: AbilityArgument):
        self._func = func
        self._name = name
        self._description = description
        self._arguments: Dict[str, AbilityArgument] = {
            arg.name: arg
            for arg in arguments
        }

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def arguments(self) -> Dict[str, AbilityArgument]:
        return self._arguments

    def __call__(self, *args, **kwargs) -> str:
        return self._func(*args, **kwargs)

    def to_json(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self._name,
                "description": self._description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        arg.name: arg.to_json()
                        for arg in self._arguments.values()
                    },
                    "required": [
                        arg.name
                        for arg in self._arguments.values()
                        if arg.is_required
                    ]
                },
            }
        }

    @staticmethod
    def create(**arguments: Union[str, List[str]]):
        """Use to define a new ability from function declaration"""
        def wrapper(func: AbilityFn):
            # create custom error raiser function
            raiser = AbilityCreationError.factory(
                func.__name__, extract_stack()[-2])

            # validate doc-string existence
            if func.__doc__ is None:
                raiser("Ability must have a doc-string")

            # create ability object
            ability = Ability(func, func.__name__, func.__doc__)
            # inject the object to the function
            setattr(func, "__ability__", ability)

            # validate the existence of the return annotation and its type
            return_type = func.__annotations__.get("return", None)
            if return_type is None or return_type != str:
                raiser("Ability must has a return annotation of str")

            # get the function specifications
            spec = inspect.getfullargspec(func)

            # validate each function argument
            for name in spec.args:
                # check if argument has a description
                if name not in arguments:
                    raiser(
                        f"Ability create function must has a description for \
'{name}' argument.\nExample: @Ability.create(\
{name}=\"some description\")")

                # validate argument annotation existence
                if name not in spec.annotations:
                    raiser(
                        f"""Missing annotation for '{name}' argument
                        Example:
                        @Ability.create({name}=\"some description\")
                        def {func.__name__}({name}: TYPE):""")

                # get argument annotation
                annotation = spec.annotations[name]

                # get argument description
                description: Union[str, List[str]] = arguments[name]
                values: Union[List[str], None] = None

                # check if description is a list of description and choices
                if type(description) is list:
                    values = description[1:]
                    description = description[0]

                # get annotation json type name
                ann = annotation_to_str(annotation)
                if ann is None:
                    raiser(f"Unknown annotation type: {type(annotation)}")

                # if arg has default value then it is not required
                is_required = (spec.defaults is None
                               or name not in spec.defaults)

                # create AbilityArgument object and add to the ability object
                ability.arguments[name] = AbilityArgument(
                    name, ann, cast(str, description), is_required,
                    values)

            return func
        return wrapper

    @staticmethod
    def has_ability(func: Callable) -> bool:
        ability = getattr(func, "__ability__", None)
        return ability is not None and isinstance(ability, Ability)

    @staticmethod
    def get_ability(func: Callable) -> "Ability":
        return cast(Ability, getattr(func, "__ability__"))


def annotation_to_str(annotation: Any,) -> Optional[str]:
    """returns the json type name for the annotation
    returns None if no cross type annotation"""
    t = annotation

    if t is str:
        return "string"
    elif t is int or t is float:
        return "number"
    elif t is bool:
        return "boolean"
    elif t is list:
        return "array"
    elif t is dict:
        return "object"
    elif t is None:
        return "null"
    else:
        return None


def import_abilities_module(path: str) -> List[Ability]:
    """Loads all abilities from a module"""
    abilities: List[Ability] = list()

    module = importlib.import_module(path)
    members = inspect.getmembers(module)

    for _, member in members:
        if isinstance(member, Ability):
            abilities.append(member)
        elif Ability.has_ability(member):
            abilities.append(Ability.get_ability(member))

    return abilities
