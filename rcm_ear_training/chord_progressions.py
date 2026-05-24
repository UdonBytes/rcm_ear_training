"""Curated chord-progression ear-test examples."""

from dataclasses import dataclass
import random

import numpy as np
import soundfile as sf

from rcm_ear_training.audio import add_fade_out, create_silence
from rcm_ear_training import config
from rcm_ear_training.questions import make_safe_filename
from rcm_ear_training.samples import find_sample_path, load_piano_sample
from rcm_ear_training.theory import note_to_midi


CHORD_PROGRESSION_AUDIO_VERSION = "progressions_v17"
CHORD_PROGRESSION_CHORD_DURATION = 1.0
CHORD_PROGRESSION_CHORD_GAP = 0.0
CHORD_PROGRESSION_REPEAT_PAUSE = 1.0
CHORD_PROGRESSION_BASS_VOLUME = 1.15
CHORD_PROGRESSION_INNER_VOICE_VOLUME = 0.7
CHORD_PROGRESSION_TOP_VOICE_VOLUME = 1.75
CHORD_PROGRESSION_BASS_FUNDAMENTAL_VOLUME = 0.35
LEVEL_5_CHORD_PROGRESSION_CHOICES = ("I-IV-I", "I-V-I")
LEVEL_6_CHORD_PROGRESSION_CHOICES = ("I-IV-I / i-iv-i", "I-V-I / i-V-i")

# These examples were easy to mis-hear as A2 because of masking in the piano sample.
# The validation keeps octave-specific bass lines from silently drifting.
STRICT_D_TO_V_BASS_LINES = {
    "prog5-d-v-1": ("D3", "A3", "D3"),
    "prog5-d-v-2": ("D3", "A3", "D3"),
    "prog6-d-minor-v-1": ("D3", "A3", "D3"),
}


@dataclass(frozen=True)
class ChordProgressionEvent:
    label: str
    top_notes: tuple[str, ...]
    bass_note: str


@dataclass(frozen=True)
class ChordProgressionExample:
    id: str
    level: int
    key: str
    progression: str
    inversion_pattern: str
    events: tuple[ChordProgressionEvent, ...]
    status: str = "approved"


@dataclass(frozen=True)
class ChordProgressionQuestion:
    level: int
    example_id: str
    key: str
    progression: str
    inversion_pattern: str
    answer: str
    choices: tuple[str, ...]
    audio_file: str


def event(label, top_notes, bass_note):
    """Create a chord-progression event."""

    return ChordProgressionEvent(
        label=label,
        top_notes=tuple(top_notes),
        bass_note=bass_note,
    )


LEVEL_5_CHORD_PROGRESSION_EXAMPLES = (
    ChordProgressionExample(
        id="prog5-iv-1",
        level=5,
        key="C major",
        progression="I IV I",
        inversion_pattern="I6, IV, I6",
        events=(
            event("I6", ("E4", "G4", "C5"), "C3"),
            event("IV", ("F4", "A4", "C5"), "F3"),
            event("I6", ("E4", "G4", "C5"), "C3"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-iv-2",
        level=5,
        key="C major",
        progression="I IV I",
        inversion_pattern="I, IV6/4, I",
        events=(
            event("I", ("C4", "E4", "G4"), "C3"),
            event("IV6/4", ("C4", "F4", "A4"), "F3"),
            event("I", ("C4", "E4", "G4"), "C3"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-iv-3",
        level=5,
        key="C major",
        progression="I IV I",
        inversion_pattern="I6/4, IV6, I6/4",
        events=(
            event("I6/4", ("G3", "C4", "E4"), "C3"),
            event("IV6", ("A3", "C4", "F4"), "F3"),
            event("I6/4", ("G3", "C4", "E4"), "C3"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-iv-4",
        level=5,
        key="C major",
        progression="I IV I",
        inversion_pattern="I, IV6, I6/4",
        events=(
            event("I", ("C4", "E4", "G4"), "C3"),
            event("IV6", ("A3", "C4", "F4"), "F3"),
            event("I6/4", ("G3", "C4", "E4"), "C3"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-iv-5",
        level=5,
        key="C major",
        progression="I IV I",
        inversion_pattern="I6/4, IV6, I",
        events=(
            event("I6/4", ("G3", "C4", "E4"), "C3"),
            event("IV6", ("A3", "C4", "F4"), "F3"),
            event("I", ("C4", "E4", "G4"), "C3"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-v-1",
        level=5,
        key="C major",
        progression="I V I",
        inversion_pattern="I, V6, I",
        events=(
            event("I", ("C4", "E4", "G4"), "C3"),
            event("V6", ("B3", "D4", "G4"), "G3"),
            event("I", ("C4", "E4", "G4"), "C3"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-v-2",
        level=5,
        key="C major",
        progression="I V I",
        inversion_pattern="I6, V6/4, I6",
        events=(
            event("I6", ("E4", "G4", "C5"), "C3"),
            event("V6/4", ("D4", "G4", "B4"), "G3"),
            event("I6", ("E4", "G4", "C5"), "C3"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-v-3",
        level=5,
        key="C major",
        progression="I V I",
        inversion_pattern="I6/4, V, I6",
        events=(
            event("I6/4", ("G4", "C5", "E5"), "C3"),
            event("V", ("G4", "B4", "D5"), "G3"),
            event("I6", ("E4", "G4", "C5"), "C3"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-v-4",
        level=5,
        key="C major",
        progression="I V I",
        inversion_pattern="I6, V, I6/4",
        events=(
            event("I6", ("E4", "G4", "C5"), "C3"),
            event("V", ("G4", "B4", "D5"), "G3"),
            event("I6/4", ("G4", "C5", "E5"), "C3"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-v-5",
        level=5,
        key="C major",
        progression="I V I",
        inversion_pattern="I6/4, V, I6/4",
        events=(
            event("I6/4", ("G4", "C5", "E5"), "C3"),
            event("V", ("G4", "B4", "D5"), "G3"),
            event("I6/4", ("G4", "C5", "E5"), "C3"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-g-iv-1",
        level=5,
        key="G major",
        progression="I IV I",
        inversion_pattern="I6, IV, I6",
        events=(
            event("I6", ("B3", "D4", "G4"), "G2"),
            event("IV", ("C4", "E4", "G4"), "C3"),
            event("I6", ("B3", "D4", "G4"), "G2"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-g-iv-2",
        level=5,
        key="G major",
        progression="I IV I",
        inversion_pattern="I, IV6/4, I",
        events=(
            event("I", ("G3", "B3", "D4"), "G2"),
            event("IV6/4", ("G3", "C4", "E4"), "C3"),
            event("I", ("G3", "B3", "D4"), "G2"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-g-iv-3",
        level=5,
        key="G major",
        progression="I IV I",
        inversion_pattern="I6/4, IV6, I6/4",
        events=(
            event("I6/4", ("D4", "G4", "B4"), "G2"),
            event("IV6", ("E4", "G4", "C5"), "C3"),
            event("I6/4", ("D4", "G4", "B4"), "G2"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-g-iv-4",
        level=5,
        key="G major",
        progression="I IV I",
        inversion_pattern="I, IV6, I6/4",
        events=(
            event("I", ("G4", "B4", "D5"), "G2"),
            event("IV6", ("E4", "G4", "C5"), "C3"),
            event("I6/4", ("D4", "G4", "B4"), "G2"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-g-iv-5",
        level=5,
        key="G major",
        progression="I IV I",
        inversion_pattern="I6/4, IV6, I",
        events=(
            event("I6/4", ("D4", "G4", "B4"), "G2"),
            event("IV6", ("E4", "G4", "C5"), "C3"),
            event("I", ("G4", "B4", "D5"), "G2"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-g-v-1",
        level=5,
        key="G major",
        progression="I V I",
        inversion_pattern="I, V6, I",
        events=(
            event("I", ("G4", "B4", "D5"), "G2"),
            event("V6", ("F#4", "A4", "D5"), "D3"),
            event("I", ("G4", "B4", "D5"), "G2"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-g-v-2",
        level=5,
        key="G major",
        progression="I V I",
        inversion_pattern="I6, V6/4, I6",
        events=(
            event("I6", ("B3", "D4", "G4"), "G2"),
            event("V6/4", ("A3", "D4", "F#4"), "D3"),
            event("I6", ("B3", "D4", "G4"), "G2"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-g-v-3",
        level=5,
        key="G major",
        progression="I V I",
        inversion_pattern="I6/4, V, I6",
        events=(
            event("I6/4", ("D4", "G4", "B4"), "G2"),
            event("V", ("D4", "F#4", "A4"), "D3"),
            event("I6", ("B3", "D4", "G4"), "G2"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-g-v-4",
        level=5,
        key="G major",
        progression="I V I",
        inversion_pattern="I6, V, I6/4",
        events=(
            event("I6", ("B3", "D4", "G4"), "G2"),
            event("V", ("D4", "F#4", "A4"), "D3"),
            event("I6/4", ("D4", "G4", "B4"), "G2"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-g-v-5",
        level=5,
        key="G major",
        progression="I V I",
        inversion_pattern="I6/4, V, I6/4",
        events=(
            event("I6/4", ("D4", "G4", "B4"), "G2"),
            event("V", ("D4", "F#4", "A4"), "D3"),
            event("I6/4", ("D4", "G4", "B4"), "G2"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-a-iv-1",
        level=5,
        key="A major",
        progression="I IV I",
        inversion_pattern="I6, IV, I6",
        events=(
            event("I6", ("C#4", "E4", "A4"), "A2"),
            event("IV", ("D4", "F#4", "A4"), "D3"),
            event("I6", ("C#4", "E4", "A4"), "A2"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-a-iv-2",
        level=5,
        key="A major",
        progression="I IV I",
        inversion_pattern="I, IV6/4, I",
        events=(
            event("I", ("A3", "C#4", "E4"), "A2"),
            event("IV6/4", ("A3", "D4", "F#4"), "D3"),
            event("I", ("A3", "C#4", "E4"), "A2"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-a-iv-3",
        level=5,
        key="A major",
        progression="I IV I",
        inversion_pattern="I6/4, IV6, I6/4",
        events=(
            event("I6/4", ("E4", "A4", "C#5"), "A2"),
            event("IV6", ("F#4", "A4", "D5"), "D3"),
            event("I6/4", ("E4", "A4", "C#5"), "A2"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-a-iv-4",
        level=5,
        key="A major",
        progression="I IV I",
        inversion_pattern="I, IV6, I6/4",
        events=(
            event("I", ("A4", "C#5", "E5"), "A2"),
            event("IV6", ("F#4", "A4", "D5"), "D3"),
            event("I6/4", ("E4", "A4", "C#5"), "A2"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-a-iv-5",
        level=5,
        key="A major",
        progression="I IV I",
        inversion_pattern="I6/4, IV6, I",
        events=(
            event("I6/4", ("E4", "A4", "C#5"), "A2"),
            event("IV6", ("F#4", "A4", "D5"), "D3"),
            event("I", ("A4", "C#5", "E5"), "A2"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-a-v-1",
        level=5,
        key="A major",
        progression="I V I",
        inversion_pattern="I, V6, I",
        events=(
            event("I", ("A4", "C#5", "E5"), "A2"),
            event("V6", ("G#4", "B4", "E5"), "E3"),
            event("I", ("A4", "C#5", "E5"), "A2"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-a-v-2",
        level=5,
        key="A major",
        progression="I V I",
        inversion_pattern="I6, V6/4, I6",
        events=(
            event("I6", ("C#4", "E4", "A4"), "A2"),
            event("V6/4", ("B3", "E4", "G#4"), "E3"),
            event("I6", ("C#4", "E4", "A4"), "A2"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-a-v-3",
        level=5,
        key="A major",
        progression="I V I",
        inversion_pattern="I6/4, V, I6",
        events=(
            event("I6/4", ("E4", "A4", "C#5"), "A2"),
            event("V", ("E4", "G#4", "B4"), "E3"),
            event("I6", ("C#4", "E4", "A4"), "A2"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-a-v-4",
        level=5,
        key="A major",
        progression="I V I",
        inversion_pattern="I6, V, I6/4",
        events=(
            event("I6", ("C#4", "E4", "A4"), "A2"),
            event("V", ("E4", "G#4", "B4"), "E3"),
            event("I6/4", ("E4", "A4", "C#5"), "A2"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-a-v-5",
        level=5,
        key="A major",
        progression="I V I",
        inversion_pattern="I6/4, V, I6/4",
        events=(
            event("I6/4", ("E4", "A4", "C#5"), "A2"),
            event("V", ("E4", "G#4", "B4"), "E3"),
            event("I6/4", ("E4", "A4", "C#5"), "A2"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-d-iv-1",
        level=5,
        key="D major",
        progression="I IV I",
        inversion_pattern="I6, IV, I6",
        events=(
            event("I6", ("F#4", "A4", "D5"), "D3"),
            event("IV", ("G4", "B4", "D5"), "G3"),
            event("I6", ("F#4", "A4", "D5"), "D3"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-d-iv-2",
        level=5,
        key="D major",
        progression="I IV I",
        inversion_pattern="I, IV6/4, I",
        events=(
            event("I", ("D4", "F#4", "A4"), "D3"),
            event("IV6/4", ("D4", "G4", "B4"), "G3"),
            event("I", ("D4", "F#4", "A4"), "D3"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-d-iv-3",
        level=5,
        key="D major",
        progression="I IV I",
        inversion_pattern="I6/4, IV6, I6/4",
        events=(
            event("I6/4", ("A3", "D4", "F#4"), "D3"),
            event("IV6", ("B3", "D4", "G4"), "G3"),
            event("I6/4", ("A3", "D4", "F#4"), "D3"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-d-iv-4",
        level=5,
        key="D major",
        progression="I IV I",
        inversion_pattern="I, IV6, I6/4",
        events=(
            event("I", ("D4", "F#4", "A4"), "D3"),
            event("IV6", ("B3", "D4", "G4"), "G3"),
            event("I6/4", ("A3", "D4", "F#4"), "D3"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-d-iv-5",
        level=5,
        key="D major",
        progression="I IV I",
        inversion_pattern="I6/4, IV6, I",
        events=(
            event("I6/4", ("A3", "D4", "F#4"), "D3"),
            event("IV6", ("B3", "D4", "G4"), "G3"),
            event("I", ("D4", "F#4", "A4"), "D3"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-d-v-1",
        level=5,
        key="D major",
        progression="I V I",
        inversion_pattern="I, V6, I",
        events=(
            event("I", ("D4", "F#4", "A4"), "D3"),
            event("V6", ("C#4", "E4", "A4"), "A3"),
            event("I", ("D4", "F#4", "A4"), "D3"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-d-v-2",
        level=5,
        key="D major",
        progression="I V I",
        inversion_pattern="I6, V6/4, I6",
        events=(
            event("I6", ("F#4", "A4", "D5"), "D3"),
            event("V6/4", ("E4", "A4", "C#5"), "A3"),
            event("I6", ("F#4", "A4", "D5"), "D3"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-d-v-3",
        level=5,
        key="D major",
        progression="I V I",
        inversion_pattern="I6/4, V, I6",
        events=(
            event("I6/4", ("A4", "D5", "F#5"), "D3"),
            event("V", ("A4", "C#5", "E5"), "A3"),
            event("I6", ("F#4", "A4", "D5"), "D3"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-d-v-4",
        level=5,
        key="D major",
        progression="I V I",
        inversion_pattern="I6, V, I6/4",
        events=(
            event("I6", ("F#4", "A4", "D5"), "D3"),
            event("V", ("A4", "C#5", "E5"), "A3"),
            event("I6/4", ("A4", "D5", "F#5"), "D3"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-d-v-5",
        level=5,
        key="D major",
        progression="I V I",
        inversion_pattern="I6/4, V, I6/4",
        events=(
            event("I6/4", ("A4", "D5", "F#5"), "D3"),
            event("V", ("A4", "C#5", "E5"), "A3"),
            event("I6/4", ("A4", "D5", "F#5"), "D3"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-e-iv-1",
        level=5,
        key="E major",
        progression="I IV I",
        inversion_pattern="I6, IV, I6",
        events=(
            event("I6", ("G#4", "B4", "E5"), "E3"),
            event("IV", ("A4", "C#5", "E5"), "A3"),
            event("I6", ("G#4", "B4", "E5"), "E3"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-e-iv-2",
        level=5,
        key="E major",
        progression="I IV I",
        inversion_pattern="I, IV6/4, I",
        events=(
            event("I", ("E4", "G#4", "B4"), "E3"),
            event("IV6/4", ("E4", "A4", "C#5"), "A3"),
            event("I", ("E4", "G#4", "B4"), "E3"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-e-iv-3",
        level=5,
        key="E major",
        progression="I IV I",
        inversion_pattern="I6/4, IV6, I6/4",
        events=(
            event("I6/4", ("B3", "E4", "G#4"), "E3"),
            event("IV6", ("C#4", "E4", "A4"), "A3"),
            event("I6/4", ("B3", "E4", "G#4"), "E3"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-e-iv-4",
        level=5,
        key="E major",
        progression="I IV I",
        inversion_pattern="I, IV6, I6/4",
        events=(
            event("I", ("E4", "G#4", "B4"), "E3"),
            event("IV6", ("C#4", "E4", "A4"), "A3"),
            event("I6/4", ("B3", "E4", "G#4"), "E3"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-e-iv-5",
        level=5,
        key="E major",
        progression="I IV I",
        inversion_pattern="I6/4, IV6, I",
        events=(
            event("I6/4", ("B3", "E4", "G#4"), "E3"),
            event("IV6", ("C#4", "E4", "A4"), "A3"),
            event("I", ("E4", "G#4", "B4"), "E3"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-e-v-1",
        level=5,
        key="E major",
        progression="I V I",
        inversion_pattern="I, V6, I",
        events=(
            event("I", ("E4", "G#4", "B4"), "E3"),
            event("V6", ("D#4", "F#4", "B4"), "B3"),
            event("I", ("E4", "G#4", "B4"), "E3"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-e-v-2",
        level=5,
        key="E major",
        progression="I V I",
        inversion_pattern="I6, V6/4, I6",
        events=(
            event("I6", ("G#4", "B4", "E5"), "E3"),
            event("V6/4", ("F#4", "B4", "D#5"), "B3"),
            event("I6", ("G#4", "B4", "E5"), "E3"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-e-v-3",
        level=5,
        key="E major",
        progression="I V I",
        inversion_pattern="I6/4, V, I6",
        events=(
            event("I6/4", ("B4", "E5", "G#5"), "E3"),
            event("V", ("B4", "D#5", "F#5"), "B3"),
            event("I6", ("G#4", "B4", "E5"), "E3"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-e-v-4",
        level=5,
        key="E major",
        progression="I V I",
        inversion_pattern="I6, V, I6/4",
        events=(
            event("I6", ("G#4", "B4", "E5"), "E3"),
            event("V", ("B4", "D#5", "F#5"), "B3"),
            event("I6/4", ("B4", "E5", "G#5"), "E3"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-e-v-5",
        level=5,
        key="E major",
        progression="I V I",
        inversion_pattern="I6/4, V, I6/4",
        events=(
            event("I6/4", ("B4", "E5", "G#5"), "E3"),
            event("V", ("B4", "D#5", "F#5"), "B3"),
            event("I6/4", ("B4", "E5", "G#5"), "E3"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-f-iv-1",
        level=5,
        key="F major",
        progression="I IV I",
        inversion_pattern="I6, IV, I6",
        events=(
            event("I6", ("A3", "C4", "F4"), "F2"),
            event("IV", ("Bb3", "D4", "F4"), "Bb2"),
            event("I6", ("A3", "C4", "F4"), "F2"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-f-iv-2",
        level=5,
        key="F major",
        progression="I IV I",
        inversion_pattern="I, IV6/4, I",
        events=(
            event("I", ("F3", "A3", "C4"), "F2"),
            event("IV6/4", ("F3", "Bb3", "D4"), "Bb2"),
            event("I", ("F3", "A3", "C4"), "F2"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-f-iv-3",
        level=5,
        key="F major",
        progression="I IV I",
        inversion_pattern="I6/4, IV6, I6/4",
        events=(
            event("I6/4", ("C4", "F4", "A4"), "F2"),
            event("IV6", ("D4", "F4", "Bb4"), "Bb2"),
            event("I6/4", ("C4", "F4", "A4"), "F2"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-f-iv-4",
        level=5,
        key="F major",
        progression="I IV I",
        inversion_pattern="I, IV6, I6/4",
        events=(
            event("I", ("F4", "A4", "C5"), "F2"),
            event("IV6", ("D4", "F4", "Bb4"), "Bb2"),
            event("I6/4", ("C4", "F4", "A4"), "F2"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-f-iv-5",
        level=5,
        key="F major",
        progression="I IV I",
        inversion_pattern="I6/4, IV6, I",
        events=(
            event("I6/4", ("C4", "F4", "A4"), "F2"),
            event("IV6", ("D4", "F4", "Bb4"), "Bb2"),
            event("I", ("F4", "A4", "C5"), "F2"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-f-v-1",
        level=5,
        key="F major",
        progression="I V I",
        inversion_pattern="I, V6, I",
        events=(
            event("I", ("F4", "A4", "C5"), "F2"),
            event("V6", ("E4", "G4", "C5"), "C3"),
            event("I", ("F4", "A4", "C5"), "F2"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-f-v-2",
        level=5,
        key="F major",
        progression="I V I",
        inversion_pattern="I6, V6/4, I6",
        events=(
            event("I6", ("A3", "C4", "F4"), "F2"),
            event("V6/4", ("G3", "C4", "E4"), "C3"),
            event("I6", ("A3", "C4", "F4"), "F2"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-f-v-3",
        level=5,
        key="F major",
        progression="I V I",
        inversion_pattern="I6/4, V, I6",
        events=(
            event("I6/4", ("C4", "F4", "A4"), "F2"),
            event("V", ("C4", "E4", "G4"), "C3"),
            event("I6", ("A3", "C4", "F4"), "F2"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-f-v-4",
        level=5,
        key="F major",
        progression="I V I",
        inversion_pattern="I6, V, I6/4",
        events=(
            event("I6", ("A3", "C4", "F4"), "F2"),
            event("V", ("C4", "E4", "G4"), "C3"),
            event("I6/4", ("C4", "F4", "A4"), "F2"),
        ),
    ),
    ChordProgressionExample(
        id="prog5-f-v-5",
        level=5,
        key="F major",
        progression="I V I",
        inversion_pattern="I6/4, V, I6/4",
        events=(
            event("I6/4", ("C4", "F4", "A4"), "F2"),
            event("V", ("C4", "E4", "G4"), "C3"),
            event("I6/4", ("C4", "F4", "A4"), "F2"),
        ),
    ),
)


LEVEL_6_MINOR_CHORD_PROGRESSION_EXAMPLES = (
    ChordProgressionExample(
        id="prog6-f-minor-iv-1",
        level=6,
        key="F minor",
        progression="i iv i",
        inversion_pattern="i6, iv, i6",
        events=(
            event("i6", ("Ab3", "C4", "F4"), "F2"),
            event("iv", ("Bb3", "Db4", "F4"), "Bb2"),
            event("i6", ("Ab3", "C4", "F4"), "F2"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-f-minor-iv-2",
        level=6,
        key="F minor",
        progression="i iv i",
        inversion_pattern="i, iv6/4, i",
        events=(
            event("i", ("F3", "Ab3", "C4"), "F2"),
            event("iv6/4", ("F3", "Bb3", "Db4"), "Bb2"),
            event("i", ("F3", "Ab3", "C4"), "F2"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-f-minor-iv-3",
        level=6,
        key="F minor",
        progression="i iv i",
        inversion_pattern="i6/4, iv6, i6/4",
        events=(
            event("i6/4", ("C4", "F4", "Ab4"), "F2"),
            event("iv6", ("Db4", "F4", "Bb4"), "Bb2"),
            event("i6/4", ("C4", "F4", "Ab4"), "F2"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-f-minor-iv-4",
        level=6,
        key="F minor",
        progression="i iv i",
        inversion_pattern="i, iv6, i6/4",
        events=(
            event("i", ("F4", "Ab4", "C5"), "F2"),
            event("iv6", ("Db4", "F4", "Bb4"), "Bb2"),
            event("i6/4", ("C4", "F4", "Ab4"), "F2"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-f-minor-iv-5",
        level=6,
        key="F minor",
        progression="i iv i",
        inversion_pattern="i6/4, iv6, i",
        events=(
            event("i6/4", ("C4", "F4", "Ab4"), "F2"),
            event("iv6", ("Db4", "F4", "Bb4"), "Bb2"),
            event("i", ("F4", "Ab4", "C5"), "F2"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-f-minor-v-1",
        level=6,
        key="F minor",
        progression="i V i",
        inversion_pattern="i, V6, i",
        events=(
            event("i", ("F4", "Ab4", "C5"), "F2"),
            event("V6", ("E4", "G4", "C5"), "C3"),
            event("i", ("F4", "Ab4", "C5"), "F2"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-f-minor-v-2",
        level=6,
        key="F minor",
        progression="i V i",
        inversion_pattern="i6, V6/4, i6",
        events=(
            event("i6", ("Ab3", "C4", "F4"), "F2"),
            event("V6/4", ("G3", "C4", "E4"), "C3"),
            event("i6", ("Ab3", "C4", "F4"), "F2"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-f-minor-v-3",
        level=6,
        key="F minor",
        progression="i V i",
        inversion_pattern="i6/4, V, i6",
        events=(
            event("i6/4", ("C4", "F4", "Ab4"), "F2"),
            event("V", ("C4", "E4", "G4"), "C3"),
            event("i6", ("Ab3", "C4", "F4"), "F2"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-f-minor-v-4",
        level=6,
        key="F minor",
        progression="i V i",
        inversion_pattern="i6, V, i6/4",
        events=(
            event("i6", ("Ab3", "C4", "F4"), "F2"),
            event("V", ("C4", "E4", "G4"), "C3"),
            event("i6/4", ("C4", "F4", "Ab4"), "F2"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-f-minor-v-5",
        level=6,
        key="F minor",
        progression="i V i",
        inversion_pattern="i6/4, V, i6/4",
        events=(
            event("i6/4", ("C4", "F4", "Ab4"), "F2"),
            event("V", ("C4", "E4", "G4"), "C3"),
            event("i6/4", ("C4", "F4", "Ab4"), "F2"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-g-minor-iv-1",
        level=6,
        key="G minor",
        progression="i iv i",
        inversion_pattern="i6, iv, i6",
        events=(
            event("i6", ("Bb3", "D4", "G4"), "G2"),
            event("iv", ("C4", "Eb4", "G4"), "C3"),
            event("i6", ("Bb3", "D4", "G4"), "G2"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-g-minor-iv-2",
        level=6,
        key="G minor",
        progression="i iv i",
        inversion_pattern="i, iv6/4, i",
        events=(
            event("i", ("G3", "Bb3", "D4"), "G2"),
            event("iv6/4", ("G3", "C4", "Eb4"), "C3"),
            event("i", ("G3", "Bb3", "D4"), "G2"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-g-minor-iv-3",
        level=6,
        key="G minor",
        progression="i iv i",
        inversion_pattern="i6/4, iv6, i6/4",
        events=(
            event("i6/4", ("D4", "G4", "Bb4"), "G2"),
            event("iv6", ("Eb4", "G4", "C5"), "C3"),
            event("i6/4", ("D4", "G4", "Bb4"), "G2"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-g-minor-iv-4",
        level=6,
        key="G minor",
        progression="i iv i",
        inversion_pattern="i, iv6, i6/4",
        events=(
            event("i", ("G4", "Bb4", "D5"), "G2"),
            event("iv6", ("Eb4", "G4", "C5"), "C3"),
            event("i6/4", ("D4", "G4", "Bb4"), "G2"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-g-minor-iv-5",
        level=6,
        key="G minor",
        progression="i iv i",
        inversion_pattern="i6/4, iv6, i",
        events=(
            event("i6/4", ("D4", "G4", "Bb4"), "G2"),
            event("iv6", ("Eb4", "G4", "C5"), "C3"),
            event("i", ("G4", "Bb4", "D5"), "G2"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-g-minor-v-1",
        level=6,
        key="G minor",
        progression="i V i",
        inversion_pattern="i, V6, i",
        events=(
            event("i", ("G4", "Bb4", "D5"), "G2"),
            event("V6", ("F#4", "A4", "D5"), "D3"),
            event("i", ("G4", "Bb4", "D5"), "G2"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-g-minor-v-2",
        level=6,
        key="G minor",
        progression="i V i",
        inversion_pattern="i6, V6/4, i6",
        events=(
            event("i6", ("Bb3", "D4", "G4"), "G2"),
            event("V6/4", ("A3", "D4", "F#4"), "D3"),
            event("i6", ("Bb3", "D4", "G4"), "G2"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-g-minor-v-3",
        level=6,
        key="G minor",
        progression="i V i",
        inversion_pattern="i6/4, V, i6",
        events=(
            event("i6/4", ("D4", "G4", "Bb4"), "G2"),
            event("V", ("D4", "F#4", "A4"), "D3"),
            event("i6", ("Bb3", "D4", "G4"), "G2"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-g-minor-v-4",
        level=6,
        key="G minor",
        progression="i V i",
        inversion_pattern="i6, V, i6/4",
        events=(
            event("i6", ("Bb3", "D4", "G4"), "G2"),
            event("V", ("D4", "F#4", "A4"), "D3"),
            event("i6/4", ("D4", "G4", "Bb4"), "G2"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-g-minor-v-5",
        level=6,
        key="G minor",
        progression="i V i",
        inversion_pattern="i6/4, V, i6/4",
        events=(
            event("i6/4", ("D4", "G4", "Bb4"), "G2"),
            event("V", ("D4", "F#4", "A4"), "D3"),
            event("i6/4", ("D4", "G4", "Bb4"), "G2"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-a-minor-iv-1",
        level=6,
        key="A minor",
        progression="i iv i",
        inversion_pattern="i6, iv, i6",
        events=(
            event("i6", ("C4", "E4", "A4"), "A2"),
            event("iv", ("D4", "F4", "A4"), "D3"),
            event("i6", ("C4", "E4", "A4"), "A2"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-a-minor-iv-2",
        level=6,
        key="A minor",
        progression="i iv i",
        inversion_pattern="i, iv6/4, i",
        events=(
            event("i", ("A3", "C4", "E4"), "A2"),
            event("iv6/4", ("A3", "D4", "F4"), "D3"),
            event("i", ("A3", "C4", "E4"), "A2"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-a-minor-iv-3",
        level=6,
        key="A minor",
        progression="i iv i",
        inversion_pattern="i6/4, iv6, i6/4",
        events=(
            event("i6/4", ("E4", "A4", "C5"), "A2"),
            event("iv6", ("F4", "A4", "D5"), "D3"),
            event("i6/4", ("E4", "A4", "C5"), "A2"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-a-minor-iv-4",
        level=6,
        key="A minor",
        progression="i iv i",
        inversion_pattern="i, iv6, i6/4",
        events=(
            event("i", ("A4", "C5", "E5"), "A2"),
            event("iv6", ("F4", "A4", "D5"), "D3"),
            event("i6/4", ("E4", "A4", "C5"), "A2"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-a-minor-iv-5",
        level=6,
        key="A minor",
        progression="i iv i",
        inversion_pattern="i6/4, iv6, i",
        events=(
            event("i6/4", ("E4", "A4", "C5"), "A2"),
            event("iv6", ("F4", "A4", "D5"), "D3"),
            event("i", ("A4", "C5", "E5"), "A2"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-a-minor-v-1",
        level=6,
        key="A minor",
        progression="i V i",
        inversion_pattern="i, V6, i",
        events=(
            event("i", ("A4", "C5", "E5"), "A2"),
            event("V6", ("G#4", "B4", "E5"), "E3"),
            event("i", ("A4", "C5", "E5"), "A2"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-a-minor-v-2",
        level=6,
        key="A minor",
        progression="i V i",
        inversion_pattern="i6, V6/4, i6",
        events=(
            event("i6", ("C4", "E4", "A4"), "A2"),
            event("V6/4", ("B3", "E4", "G#4"), "E3"),
            event("i6", ("C4", "E4", "A4"), "A2"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-a-minor-v-3",
        level=6,
        key="A minor",
        progression="i V i",
        inversion_pattern="i6/4, V, i6",
        events=(
            event("i6/4", ("E4", "A4", "C5"), "A2"),
            event("V", ("E4", "G#4", "B4"), "E3"),
            event("i6", ("C4", "E4", "A4"), "A2"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-a-minor-v-4",
        level=6,
        key="A minor",
        progression="i V i",
        inversion_pattern="i6, V, i6/4",
        events=(
            event("i6", ("C4", "E4", "A4"), "A2"),
            event("V", ("E4", "G#4", "B4"), "E3"),
            event("i6/4", ("E4", "A4", "C5"), "A2"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-a-minor-v-5",
        level=6,
        key="A minor",
        progression="i V i",
        inversion_pattern="i6/4, V, i6/4",
        events=(
            event("i6/4", ("E4", "A4", "C5"), "A2"),
            event("V", ("E4", "G#4", "B4"), "E3"),
            event("i6/4", ("E4", "A4", "C5"), "A2"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-c-minor-iv-1",
        level=6,
        key="C minor",
        progression="i iv i",
        inversion_pattern="i6, iv, i6",
        events=(
            event("i6", ("Eb4", "G4", "C5"), "C3"),
            event("iv", ("F4", "Ab4", "C5"), "F3"),
            event("i6", ("Eb4", "G4", "C5"), "C3"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-c-minor-iv-2",
        level=6,
        key="C minor",
        progression="i iv i",
        inversion_pattern="i, iv6/4, i",
        events=(
            event("i", ("C4", "Eb4", "G4"), "C3"),
            event("iv6/4", ("C4", "F4", "Ab4"), "F3"),
            event("i", ("C4", "Eb4", "G4"), "C3"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-c-minor-iv-3",
        level=6,
        key="C minor",
        progression="i iv i",
        inversion_pattern="i6/4, iv6, i6/4",
        events=(
            event("i6/4", ("G3", "C4", "Eb4"), "C3"),
            event("iv6", ("Ab3", "C4", "F4"), "F3"),
            event("i6/4", ("G3", "C4", "Eb4"), "C3"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-c-minor-iv-4",
        level=6,
        key="C minor",
        progression="i iv i",
        inversion_pattern="i, iv6, i6/4",
        events=(
            event("i", ("C4", "Eb4", "G4"), "C3"),
            event("iv6", ("Ab3", "C4", "F4"), "F3"),
            event("i6/4", ("G3", "C4", "Eb4"), "C3"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-c-minor-iv-5",
        level=6,
        key="C minor",
        progression="i iv i",
        inversion_pattern="i6/4, iv6, i",
        events=(
            event("i6/4", ("G3", "C4", "Eb4"), "C3"),
            event("iv6", ("Ab3", "C4", "F4"), "F3"),
            event("i", ("C4", "Eb4", "G4"), "C3"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-c-minor-v-1",
        level=6,
        key="C minor",
        progression="i V i",
        inversion_pattern="i, V6, i",
        events=(
            event("i", ("C4", "Eb4", "G4"), "C3"),
            event("V6", ("B3", "D4", "G4"), "G3"),
            event("i", ("C4", "Eb4", "G4"), "C3"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-c-minor-v-2",
        level=6,
        key="C minor",
        progression="i V i",
        inversion_pattern="i6, V6/4, i6",
        events=(
            event("i6", ("Eb4", "G4", "C5"), "C3"),
            event("V6/4", ("D4", "G4", "B4"), "G3"),
            event("i6", ("Eb4", "G4", "C5"), "C3"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-c-minor-v-3",
        level=6,
        key="C minor",
        progression="i V i",
        inversion_pattern="i6/4, V, i6",
        events=(
            event("i6/4", ("G4", "C5", "Eb5"), "C3"),
            event("V", ("G4", "B4", "D5"), "G3"),
            event("i6", ("Eb4", "G4", "C5"), "C3"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-c-minor-v-4",
        level=6,
        key="C minor",
        progression="i V i",
        inversion_pattern="i6, V, i6/4",
        events=(
            event("i6", ("Eb4", "G4", "C5"), "C3"),
            event("V", ("G4", "B4", "D5"), "G3"),
            event("i6/4", ("G4", "C5", "Eb5"), "C3"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-c-minor-v-5",
        level=6,
        key="C minor",
        progression="i V i",
        inversion_pattern="i6/4, V, i6/4",
        events=(
            event("i6/4", ("G4", "C5", "Eb5"), "C3"),
            event("V", ("G4", "B4", "D5"), "G3"),
            event("i6/4", ("G4", "C5", "Eb5"), "C3"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-d-minor-iv-1",
        level=6,
        key="D minor",
        progression="i iv i",
        inversion_pattern="i6, iv, i6",
        events=(
            event("i6", ("F4", "A4", "D5"), "D3"),
            event("iv", ("G4", "Bb4", "D5"), "G3"),
            event("i6", ("F4", "A4", "D5"), "D3"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-d-minor-iv-2",
        level=6,
        key="D minor",
        progression="i iv i",
        inversion_pattern="i, iv6/4, i",
        events=(
            event("i", ("D4", "F4", "A4"), "D3"),
            event("iv6/4", ("D4", "G4", "Bb4"), "G3"),
            event("i", ("D4", "F4", "A4"), "D3"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-d-minor-iv-3",
        level=6,
        key="D minor",
        progression="i iv i",
        inversion_pattern="i6/4, iv6, i6/4",
        events=(
            event("i6/4", ("A3", "D4", "F4"), "D3"),
            event("iv6", ("Bb3", "D4", "G4"), "G3"),
            event("i6/4", ("A3", "D4", "F4"), "D3"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-d-minor-iv-4",
        level=6,
        key="D minor",
        progression="i iv i",
        inversion_pattern="i, iv6, i6/4",
        events=(
            event("i", ("D4", "F4", "A4"), "D3"),
            event("iv6", ("Bb3", "D4", "G4"), "G3"),
            event("i6/4", ("A3", "D4", "F4"), "D3"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-d-minor-iv-5",
        level=6,
        key="D minor",
        progression="i iv i",
        inversion_pattern="i6/4, iv6, i",
        events=(
            event("i6/4", ("A3", "D4", "F4"), "D3"),
            event("iv6", ("Bb3", "D4", "G4"), "G3"),
            event("i", ("D4", "F4", "A4"), "D3"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-d-minor-v-1",
        level=6,
        key="D minor",
        progression="i V i",
        inversion_pattern="i, V6, i",
        events=(
            event("i", ("D4", "F4", "A4"), "D3"),
            event("V6", ("C#4", "E4", "A4"), "A3"),
            event("i", ("D4", "F4", "A4"), "D3"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-d-minor-v-2",
        level=6,
        key="D minor",
        progression="i V i",
        inversion_pattern="i6, V6/4, i6",
        events=(
            event("i6", ("F4", "A4", "D5"), "D3"),
            event("V6/4", ("E4", "A4", "C#5"), "A3"),
            event("i6", ("F4", "A4", "D5"), "D3"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-d-minor-v-3",
        level=6,
        key="D minor",
        progression="i V i",
        inversion_pattern="i6/4, V, i6",
        events=(
            event("i6/4", ("A4", "D5", "F5"), "D3"),
            event("V", ("A4", "C#5", "E5"), "A3"),
            event("i6", ("F4", "A4", "D5"), "D3"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-d-minor-v-4",
        level=6,
        key="D minor",
        progression="i V i",
        inversion_pattern="i6, V, i6/4",
        events=(
            event("i6", ("F4", "A4", "D5"), "D3"),
            event("V", ("A4", "C#5", "E5"), "A3"),
            event("i6/4", ("A4", "D5", "F5"), "D3"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-d-minor-v-5",
        level=6,
        key="D minor",
        progression="i V i",
        inversion_pattern="i6/4, V, i6/4",
        events=(
            event("i6/4", ("A4", "D5", "F5"), "D3"),
            event("V", ("A4", "C#5", "E5"), "A3"),
            event("i6/4", ("A4", "D5", "F5"), "D3"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-e-minor-iv-1",
        level=6,
        key="E minor",
        progression="i iv i",
        inversion_pattern="i6, iv, i6",
        events=(
            event("i6", ("G4", "B4", "E5"), "E3"),
            event("iv", ("A4", "C5", "E5"), "A3"),
            event("i6", ("G4", "B4", "E5"), "E3"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-e-minor-iv-2",
        level=6,
        key="E minor",
        progression="i iv i",
        inversion_pattern="i, iv6/4, i",
        events=(
            event("i", ("E4", "G4", "B4"), "E3"),
            event("iv6/4", ("E4", "A4", "C5"), "A3"),
            event("i", ("E4", "G4", "B4"), "E3"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-e-minor-iv-3",
        level=6,
        key="E minor",
        progression="i iv i",
        inversion_pattern="i6/4, iv6, i6/4",
        events=(
            event("i6/4", ("B3", "E4", "G4"), "E3"),
            event("iv6", ("C4", "E4", "A4"), "A3"),
            event("i6/4", ("B3", "E4", "G4"), "E3"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-e-minor-iv-4",
        level=6,
        key="E minor",
        progression="i iv i",
        inversion_pattern="i, iv6, i6/4",
        events=(
            event("i", ("E4", "G4", "B4"), "E3"),
            event("iv6", ("C4", "E4", "A4"), "A3"),
            event("i6/4", ("B3", "E4", "G4"), "E3"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-e-minor-iv-5",
        level=6,
        key="E minor",
        progression="i iv i",
        inversion_pattern="i6/4, iv6, i",
        events=(
            event("i6/4", ("B3", "E4", "G4"), "E3"),
            event("iv6", ("C4", "E4", "A4"), "A3"),
            event("i", ("E4", "G4", "B4"), "E3"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-e-minor-v-1",
        level=6,
        key="E minor",
        progression="i V i",
        inversion_pattern="i, V6, i",
        events=(
            event("i", ("E4", "G4", "B4"), "E3"),
            event("V6", ("D#4", "F#4", "B4"), "B3"),
            event("i", ("E4", "G4", "B4"), "E3"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-e-minor-v-2",
        level=6,
        key="E minor",
        progression="i V i",
        inversion_pattern="i6, V6/4, i6",
        events=(
            event("i6", ("G4", "B4", "E5"), "E3"),
            event("V6/4", ("F#4", "B4", "D#5"), "B3"),
            event("i6", ("G4", "B4", "E5"), "E3"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-e-minor-v-3",
        level=6,
        key="E minor",
        progression="i V i",
        inversion_pattern="i6/4, V, i6",
        events=(
            event("i6/4", ("B4", "E5", "G5"), "E3"),
            event("V", ("B4", "D#5", "F#5"), "B3"),
            event("i6", ("G4", "B4", "E5"), "E3"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-e-minor-v-4",
        level=6,
        key="E minor",
        progression="i V i",
        inversion_pattern="i6, V, i6/4",
        events=(
            event("i6", ("G4", "B4", "E5"), "E3"),
            event("V", ("B4", "D#5", "F#5"), "B3"),
            event("i6/4", ("B4", "E5", "G5"), "E3"),
        ),
    ),
    ChordProgressionExample(
        id="prog6-e-minor-v-5",
        level=6,
        key="E minor",
        progression="i V i",
        inversion_pattern="i6/4, V, i6/4",
        events=(
            event("i6/4", ("B4", "E5", "G5"), "E3"),
            event("V", ("B4", "D#5", "F#5"), "B3"),
            event("i6/4", ("B4", "E5", "G5"), "E3"),
        ),
    ),
)


def get_chord_progression_examples(level):
    """Return approved chord-progression examples for a level."""

    if level == 5:
        return LEVEL_5_CHORD_PROGRESSION_EXAMPLES

    if level == 6:
        return LEVEL_5_CHORD_PROGRESSION_EXAMPLES + LEVEL_6_MINOR_CHORD_PROGRESSION_EXAMPLES

    return ()


def get_chord_progression_choices(level):
    """Return answer choices for one chord-progression level."""

    if level == 6:
        return LEVEL_6_CHORD_PROGRESSION_CHOICES

    return LEVEL_5_CHORD_PROGRESSION_CHOICES


def chord_progression_answer_label(level, progression):
    """Return the public answer label for a progression."""

    compact_progression = progression.replace(" ", "-")

    if level == 6:
        if compact_progression.lower() == "i-iv-i":
            return "I-IV-I / i-iv-i"

        if compact_progression.lower() == "i-v-i":
            return "I-V-I / i-V-i"

    return compact_progression


def create_bass_fundamental_audio(note, duration):
    """Create a quiet fundamental-tone layer for a bass note."""

    frequency = 440 * (2 ** ((note_to_midi(note) - 69) / 12))
    sample_count = int(config.SAMPLE_RATE * duration)
    time = np.arange(sample_count) / config.SAMPLE_RATE
    tone = np.sin(2 * np.pi * frequency * time)
    fade_length = min(int(config.SAMPLE_RATE * 0.03), sample_count // 2)

    if fade_length > 0:
        fade_in = np.linspace(0, 1, fade_length)
        fade_out = np.linspace(1, 0, fade_length)
        tone[:fade_length] *= fade_in
        tone[-fade_length:] *= fade_out

    return tone * CHORD_PROGRESSION_BASS_FUNDAMENTAL_VOLUME


def create_solid_event_audio(
    notes,
    duration=CHORD_PROGRESSION_CHORD_DURATION,
    weights=None,
    reinforce_bass=False,
):
    """Create a blocked sonority from bass and upper chord notes."""

    chord_audio = np.zeros(int(config.SAMPLE_RATE * duration))
    note_weights = weights or tuple(1.0 for _ in notes)

    for note, weight in zip(notes, note_weights):
        note_audio = load_piano_sample(note)
        if len(note_audio) < len(chord_audio):
            note_audio = np.pad(note_audio, (0, len(chord_audio) - len(note_audio)))

        chord_audio += note_audio[: len(chord_audio)] * weight

    if reinforce_bass:
        chord_audio += create_bass_fundamental_audio(notes[0], duration)

    peak = np.max(np.abs(chord_audio))

    if peak > 0:
        chord_audio = chord_audio / peak

    return add_fade_out(chord_audio * config.VOLUME)


def create_progression_event_audio(event_item, reinforce_bass=False):
    """Create one chord with the soprano voice clearly above the texture."""

    notes = (event_item.bass_note, *event_item.top_notes)
    weights = (
        CHORD_PROGRESSION_BASS_VOLUME,
        *(CHORD_PROGRESSION_INNER_VOICE_VOLUME for _ in event_item.top_notes[:-1]),
        CHORD_PROGRESSION_TOP_VOICE_VOLUME,
    )

    return create_solid_event_audio(
        notes,
        weights=weights,
        reinforce_bass=reinforce_bass,
    )


def create_progression_pass(example):
    """Create one pass through a three-chord progression."""

    parts = []
    reinforce_bass = example.id in STRICT_D_TO_V_BASS_LINES

    for index, event_item in enumerate(example.events):
        parts.append(
            create_progression_event_audio(
                event_item,
                reinforce_bass=reinforce_bass,
            )
        )

        if CHORD_PROGRESSION_CHORD_GAP > 0 and index < len(example.events) - 1:
            parts.append(create_silence(CHORD_PROGRESSION_CHORD_GAP))

    return np.concatenate(parts)


def resolved_bass_notes(example):
    """Return final bass note names resolved before audio rendering."""

    return tuple(event_item.bass_note for event_item in example.events)


def resolved_bass_sample_names(example):
    """Return sample file names used by the final bass notes."""

    return tuple(find_sample_path(note).name for note in resolved_bass_notes(example))


def validate_strict_bass_line(example):
    """Fail if a strict chord-progression example has the wrong bass octaves."""

    expected_bass_line = STRICT_D_TO_V_BASS_LINES.get(example.id)

    if expected_bass_line is None:
        return

    actual_bass_line = resolved_bass_notes(example)

    if actual_bass_line != expected_bass_line:
        raise ValueError(
            f"{example.id} bass line must be {expected_bass_line}, "
            f"not {actual_bass_line}. Octave matters."
        )

    middle_bass = actual_bass_line[1]

    if middle_bass != "A3" or note_to_midi(middle_bass) != note_to_midi("A3"):
        raise ValueError(f"{example.id} middle bass must resolve to A3, not {middle_bass}.")


def debug_bass_resolution(example):
    """Print final resolved bass notes and sample paths before rendering."""

    bass_notes = resolved_bass_notes(example)
    sample_names = resolved_bass_sample_names(example)
    print(
        "[chord-progressions] rendering "
        f"{example.id}: bass={list(bass_notes)} samples={list(sample_names)}"
    )


def chord_progression_filename(level, example):
    """Return the generated WAV filename for one chord-progression example."""

    key_label = make_safe_filename(example.key)
    return make_safe_filename(
        f"level_{level}_{key_label}_{example.id}_{CHORD_PROGRESSION_AUDIO_VERSION}.wav"
    )


def chord_progression_audio_path(level, example):
    """Return the generated WAV path for one chord-progression example."""

    return config.CHORD_PROGRESSION_AUDIO_FOLDER / chord_progression_filename(level, example)


def choose_chord_progression_example(level, example_index=None):
    """Return a random or indexed chord-progression example for a level."""

    examples = get_chord_progression_examples(level)

    if not examples:
        raise ValueError(f"No chord-progression examples are implemented for level {level}.")

    if example_index is None:
        return random.choice(examples)

    return examples[example_index % len(examples)]


def create_chord_progression_waveform(example):
    """Create a progression played twice with a pause between passes."""

    progression_pass = create_progression_pass(example)

    return np.concatenate(
        [
            progression_pass,
            create_silence(CHORD_PROGRESSION_REPEAT_PAUSE),
            progression_pass,
        ]
    )


def create_chord_progression_audio(example, file_path):
    """Write a chord-progression WAV file."""

    validate_strict_bass_line(example)
    debug_bass_resolution(example)

    sf.write(
        file_path,
        create_chord_progression_waveform(example),
        config.SAMPLE_RATE,
    )


def create_chord_progression_question(level, example_index=None):
    """Create one chord-progression question and generated audio file."""

    example = choose_chord_progression_example(level, example_index)

    config.ensure_audio_folder(config.CHORD_PROGRESSION_AUDIO_FOLDER)

    filename = chord_progression_filename(level, example)
    file_path = chord_progression_audio_path(level, example)

    if not file_path.exists():
        create_chord_progression_audio(example, file_path)

    return ChordProgressionQuestion(
        level=level,
        example_id=example.id,
        key=example.key,
        progression=example.progression,
        inversion_pattern=example.inversion_pattern,
        answer=chord_progression_answer_label(level, example.progression),
        choices=get_chord_progression_choices(level),
        audio_file=f"chord_progressions/{filename}",
    )
