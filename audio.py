from openai import audio
from pydub import AudioSegment
from pydub.playback import play


def text_to_audio(text: str, path: str):
    response = audio.speech.create(
        model="tts-1-hd",
        voice="nova",
        input=text,
    )
    response.write_to_file(path)


def audio_to_text(path: str, is_english: bool = True) -> str:
    create = (
        audio.transcriptions.create if is_english else
        audio.translations.create)
    with open(path, "rb") as audio_file:
        return create(model="whisper-1", file=audio_file).text


def play_audio(path: str):
    sound = AudioSegment.from_mp3(path)
    play(sound)
