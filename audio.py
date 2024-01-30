from typing import Literal, Optional

from openai import audio
from openai._types import NotGiven
from pydub import AudioSegment
from pydub.playback import play

TTS_MODEL = Literal["tts-1", "tts-1-hd"]
TTS_VOICE = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
FILE_FORMAT = Literal["mp3", "opus", "aac", "flac"]


def text_to_audio(
    text: str,
    path: str,
    model: TTS_MODEL = "tts-1-hd",
    voice: TTS_VOICE = "nova",
    format: FILE_FORMAT = "mp3",
    speed: float = 1.0,
):
    """Convert text into audio and save it to path in `format` default is mp3.
    By default it uses the `tts-1-hd` model with the  `voice` nova.

    Args:
        input: The text to be converted.
        path: The path to save the audio file.
        model: The TTS model to use.
        voice: The voice to use.
        format: The format of the audio file.
        speed: The speech speed in range [0.24, 4]
    """
    response = audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        response_format=format,
        speed=min(max(speed, 0.25), 4.0),
    )
    response.write_to_file(path)


def audio_to_text(
    path: str, language: Optional[str] = None, prompt: Optional[str] = None
) -> str:
    """Uses Whisper-1 model to convert audio to text.

    Args:
        path: Path to audio file
        language: Language of audio, recommended to use [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).
        prompt: An optional text to guide the model's style.
    """

    with open(path, "rb") as audio_file:
        response = audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            prompt=prompt or NotGiven(),
            language=language or NotGiven(),
        )
        return response.text


def play_audio(path: str):
    """Play audio file by its path"""
    sound = AudioSegment.from_mp3(path)
    play(sound)
