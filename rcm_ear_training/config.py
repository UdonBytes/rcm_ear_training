"""Application settings and asset paths."""

from pathlib import Path


SAMPLE_RATE = 44100
NOTE_DURATION = 1.2
PAUSE_DURATION = 0.25
CONNECTED_NOTE_SPACING = 0.08
BROKEN_CHORD_NOTE_DURATION = 0.55
CHORD_EVENT_DURATION = 1.4
VOLUME = 0.35
ATTACK_THRESHOLD_RATIO = 0.05
ATTACK_PREROLL_SECONDS = 0.005

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_FOLDER = PROJECT_ROOT / "static"
AUDIO_FOLDER = STATIC_FOLDER / "audio"
INTERVAL_AUDIO_FOLDER = AUDIO_FOLDER / "intervals"
CHORD_AUDIO_FOLDER = AUDIO_FOLDER / "chords"
CLAPBACK_AUDIO_FOLDER = AUDIO_FOLDER / "clapback"
PIANO_SAMPLE_FOLDER = STATIC_FOLDER / "piano_samples"
PERCUSSION_FOLDER = STATIC_FOLDER / "percussion"
CLAPBACK_EXAMPLE_FOLDER = STATIC_FOLDER / "clapback_examples"


def ensure_audio_folder(folder=AUDIO_FOLDER):
    """Create a generated audio folder if it does not exist."""

    folder.mkdir(parents=True, exist_ok=True)
