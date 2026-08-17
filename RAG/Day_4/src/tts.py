import os
import tempfile

from kokoro import KPipeline
import soundfile as sf

# Create pipeline only once
pipeline = KPipeline(lang_code="a")   # American English


def text_to_speech(
    text: str,
    voice: str = "af_heart",
) -> str:
    """
    Converts text to speech.

    Returns
    -------
    str
        Path to generated wav file.
    """

    generator = pipeline(
        text,
        voice=voice,
        speed=1.0,
    )

    _, _, audio = next(generator)

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    sf.write(path, audio, 24000)

    return path