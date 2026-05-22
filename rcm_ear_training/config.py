"""Application settings and asset paths."""

from pathlib import Path


SAMPLE_RATE = 44100
NOTE_DURATION = 1.2
PAUSE_DURATION = 0.25
CONNECTED_NOTE_SPACING = 0.08
VOLUME = 0.35
ATTACK_THRESHOLD_RATIO = 0.01
ATTACK_PREROLL_SECONDS = 0.005

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_FOLDER = PROJECT_ROOT / "static"
AUDIO_FOLDER = STATIC_FOLDER / "audio"
PIANO_SAMPLE_FOLDER = STATIC_FOLDER / "piano_samples"


def ensure_audio_folder():
    """Create the generated audio folder if it does not exist."""

    AUDIO_FOLDER.mkdir(parents=True, exist_ok=True)
