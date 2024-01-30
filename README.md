# Agent

A collection of classes as a wrapper around OpenAI package.

I created this project to build other projects and learn more about OpenAI API.

## Features

- **Agent**: A wrapper around OpenAI ChatCompletion API.
- **Ability**: A class that defined the agent abilities also it is a wrapper
  around the tools feature in OpenAI API.
- **Message**: My custom definition of ChatCompletion API messages.
- **magic**: A decorator that creates an Agent alias for any function.
- **audio**: A wrapper functions around OpenAI Whisper and TTS engine APIs.

## Usage:

### Agent

**Agent** class takes:

- **instructions**: str, required = The agent system instructions
- **model**: str or **AgentModelName**, optional = The model name to use, defaults to "gpt-3.5-turbo-1106".
- **temperature**: float, optional = The model temperature to use, defaults to 1.0 (0.0 - 2.0), for more details check the OpenAI API documentation.
- **name**: str, optional = The name of the agent.
- **messages**: MessagesContainer, optional = A container that holds a list of Message objects which represents the history of the conversation.

To set the **OpenAI** access token:

- Set **OPENAI_API_KEY** in your environment.
- Use the **Agent** staticmethod **set_openai_key**, to set the access token.

```python
from agent import Agent

Agent.set_openai_key("...")
# OR set the environment variable OPENAI_API_KEY
```

```python
from agent import Agent

# create agent with instructions, agent name is optional
agent = Agent("You are a helpful assistant.")

# add user input
agent << "How are you?"
# OR use a UserMessage, name is optional
agent << UserMessage("How are you?")

# stream output to a file
agent >> sys.stdout
# OR, stream file parameter is optional, it is stdout by default
agent.stream()

# write response to a file
with open('file', 'w') as f:
   agent >> f
   # OR
   agent.stream(f)


# stream_to uses generate under the hood
# to manually stream response use the `generate` method
for chunk in agent.generate():
    print(chunk, end="", flush=True)
```

### Message

This class represents a message in the conversation.

Recommended to use the inherited predefined message roles:

- UserMessage
- AssistantMessage
- SystemMessage
- ToolMessage

Use the + operator to concatenate content with other messages or string.

```python
from agent.message import UserMessage

msg = UserMessage("Hello") + ", World!"
msg.content  # "Hello, World!"

msg = UserMessage("Hello") + UserMessage(", World!")
msg.content  # "Hello, World!"
```

> To work with many message objects, it is recommended to use **MessagesContainer**

### MessagesContainer

This object is used to manager many **Message** objects.
It also represents the chat history for an **Agent**.

It takes an optional list of **Message** objects.

```python
from agent.message import MessagesContainer, UserMessage, AssistantMessage

container = MessagesContainer([UserMessage("Hello")])
container.add(UserMessage("How are you?"))
container << AssistantMessage("Hi")  # OR use the << operator to add messages
container << [AssistantMessage("Hello, World!")]

container[0].content  # "Hello"
container[-1].content  # "Hello, World!"

[m.content for m in container[0:2]]  # ["Hello", "How are you?"]
```

### Ability

This object is used to define the abilities of an **Agent**.
It is a wrapper around the tools feature in OpenAI API.

Ability takes:

- **func**: **AbilityFn**, required = The ability logic which is a function that returns a string as ability response(output).
- **name**: str, required = The ability name.
- **description**: str, required = The ability description, it should describe what the ability does and why to use it.
- **arguments**: **AbilityArgument**[], optional = A list of ability arguments.

**AbilityArgument** takes:

- **name**: str, required = the ability argument name
- **type**: str, required = the ability argument type, it should describe what the argument data type is: `string, number, boolean, object, array, null`.
- **description**: str, required = the ability argument description, it should describe what the argument is.
- **is_required**: bool, optional = if this ability argument is required, default is `False`.
- **values**: str[], optional = a list of possible values, if you want to define an argument that takes a specific values (like `enums`).

The ability logic function

```python
from os import getenv


def get_environment_variable(name: str) -> str:
    """Get an environment variable by its name."""
    value = os.getenv(name)

    if value is not None:
        return f"The value of the variable {name} is {value}"
    else:
        return f"The variable {name} is not set"
```

Let's define the ability:

```python
from agent.ability import Ability, AbilityArgument

ability = Ability(
    func=get_environment_variable,
    name="get_environment_variable",
    description="Get an environment variable by its name.",
    arguments=[
        AbilityArgument(
            name="name",
            type="string",
            description="The name of the environment variable.",
            is_required=True
        )
    ])
```

Also you could do it like this:

```python
ability = Ability(
    func=get_environment_variable,
    name="get_environment_variable",
    description="Get an environment variable by its name.")

ability["name"] = AbilityArgument(
                    name="name",
                    type="string",
                    description="The name of the environment variable.",
                    is_required=True)
```

To use **Ability**:

```python
ability("EDITOR")  # nvim
```

To add an **Ability** to an **Agent**:

```python
agent << ability
# OR
agent.add_abilities(ability)
# OR
agent["get_environment_variable"] = ability
```

### Magic Function

**magic** is a decorator that creates an **Agent** alias from instruction functions.

You can pass some arguments to the **Agent** object like:

- model: AgentModelName, optional = gpt-3.5-turbo-1106
- temperature: float, optional = 1.0
- name: str, optional = None

```python
from agent.magic import magic
from enum import Enum


class Gender(Enum):
    MALE = "male"
    FEMALE = "female"


@magic()
def generate_name(max_length: int = 8, min_length: int = 4,
                  gender: Gender = Gender.MALE,
                  language: str = "English") -> str:
    """A name generator."""
    return f"""You are a name generator.
Generate a name of a {gender.value} that its characters length between \
{min_length} and {max_length} characters in {language} language."""


generate_name(language="French", gender=Gender.FEMALE)  # Aurélie
```

### Audio

This module contains some wrapper functions around OpenAI Whisper and TTS engine APIs.

- **play_audio**: Play an audio file.

  - **path**: str, required: The audio file path to play.

- **audio_to_text**: Convert an audio file to text. **Returns a string**.

  - **path**: str, required: The audio file path to convert.
  - **language**: str, optional: The language of the audio file.
    **Recommended** to use [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)
  - **prompt**: str, optional: The prompt to use.

- **text_to_audio**: Convert text to an audio file.
  - **text**: str, required: The text to convert.
  - **path**: str, required: The audio file path to save.
  - **model**: TTS_MODEL, optional: The TTS model to use.
    `[tts-1, tts-1-hd]`
  - **voice**: TTS_VOICE, optional: The TTS voice to use.
    `[alloy, echo, fable, onyx, nova, shimmer]`
  - **format**: FILE_FORMAT, optional: The format of the audio file.
    `[mp3, opus, aac, flac]`
  - **speed**: float, optional: The speed of the audio file.
    **Value Range**: `[0.25, 4.0]`

Generate speech from text:

```python
from agent.audio import play_audio, text_to_audio

text_to_audio("Hello, how are you?", "hello.mp3")
play_audio("hello.mp3")
```

## Examples:

### Storyteller

This example will show you how to use **magic functions** to create a
storyteller.

Let's define some magic functions

```python
from agent.magic import magic


@magic()
def summarize_story_description(story: str) -> str:
    """Summarize a story description into lines of the most important points."""
    return f"""Your task is to summarize a story description into lines of \
the most important points and main events.

Summarize the following story description:

{story}""".format(story=story)


@magic()
def write_core_stroy(summary: str) -> str:
    """Writes the core of the story from the main story points."""
    return f"""Write a story of detailed events that happened in the story.

    Write the story by following the next main points and events:

    {summary}""".format(summary=summary)


@magic()
def write_story(core_story: str, language: str = "English",
                chapters: int = 5) -> str:
    """Writes a full story from the core story."""
    return f"""You are a novelist.

    The novel must follow the following rules:

    The novel must have a title.
    The novel must have {chapters} chapters with at least 800 words for each.
    The novel must have a genre.
    The novel characters must have names and details like age, gender,
    interests, etr...
    The novel must have a detailed events.
    The novel must have interesting conversations between characters.

    Write the novel in {language} language.

    Write a novel based on the following core story:

    {story}""".format(language=language, story=core_story, chapters=chapters)
```

This is a story description generated by ChatGPT:

```
1. **Introduction:**
   - The story begins with a teenage girl named Emma who lives in a bustling city.
   - Emma is curious and often daydreams about a world beyond her urban surroundings.

2. **Discovery of a Mysterious Portal:**
   - While exploring an old bookstore, Emma stumbles upon an ancient book that mentions a hidden portal to a magical realm.
   - The book provides cryptic clues and a map leading to the portal's location.

3. **Journey into the Magical World:**
   - Intrigued, Emma embarks on a journey to find the portal, following the clues and the enchanted map.
   - She discovers a concealed entrance in a forgotten park that transports her into a world filled with mystical creatures and enchanting landscapes.

4. **Meeting Magical Beings:**
   - In the magical realm, Emma encounters talking animals, fairies, and wizards who guide her on her quest.
   - Each magical being imparts wisdom and unique abilities to help her navigate the challenges of this new world.

5. **Quest for a Lost Artifact:**
   - Emma learns that the magical realm is in danger, and she must retrieve a lost artifact to restore balance.
   - Her journey involves solving riddles, overcoming obstacles, and forming alliances with magical creatures.

6. **Friendship and Betrayal:**
   - Emma forms bonds with magical beings who become her friends and allies.
   - However, she also faces challenges and betrayal from unexpected quarters, adding a layer of complexity to her quest.

7. **Epic Showdown and Redemption:**
   - The climax involves an epic showdown where Emma must confront the antagonist threatening the magical world.
   - Through courage and newfound abilities, Emma overcomes challenges, leading to redemption and restoration of the realm.

8. **Return to the City:**
   - After saving the magical world, Emma returns to her city with a newfound appreciation for the ordinary.
   - The experiences in the magical realm shape her character and perspective, making her cherish the extraordinary in the everyday.

This is just a set of points to provide a framework for a potential story. Feel free to expand upon these ideas and develop the narrative further based on your preferences and creative direction.
```

Now let's use all of the above:

```python

with open("story_description.txt", "r") as f:
    story_description = f.read()

summary = summarize_story_description(story_description)
core_story = write_core_stroy(summary)
story = write_story(core_story, chapters=10)

with open("novel.txt", "w") as f:
    f.write(story)
```
