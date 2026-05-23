"""Curated chord-progression ear-test examples."""

from dataclasses import dataclass
import random

import numpy as np
import soundfile as sf

from rcm_ear_training.audio import add_fade_out, create_silence
from rcm_ear_training import config
from rcm_ear_training.questions import make_safe_filename
from rcm_ear_training.samples import load_piano_sample


CHORD_PROGRESSION_AUDIO_VERSION = "progressions_v16"
CHORD_PROGRESSION_CHORD_DURATION = 1.0
CHORD_PROGRESSION_CHORD_GAP = 0.18
CHORD_PROGRESSION_REPEAT_PAUSE = 1.0
CHORD_PROGRESSION_BASS_VOLUME = 1.15
CHORD_PROGRESSION_INNER_VOICE_VOLUME = 0.7
CHORD_PROGRESSION_TOP_VOICE_VOLUME = 1.75
CHORD_PROGRESSION_CHOICES = ("I-IV-I", "I-V-I")


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


def get_chord_progression_examples(level):
    """Return approved chord-progression examples for a level."""

    if level == 5:
        return LEVEL_5_CHORD_PROGRESSION_EXAMPLES

    return ()


def create_solid_event_audio(
    notes,
    duration=CHORD_PROGRESSION_CHORD_DURATION,
    weights=None,
):
    """Create a blocked sonority from bass and upper chord notes."""

    chord_audio = np.zeros(int(config.SAMPLE_RATE * duration))
    note_weights = weights or tuple(1.0 for _ in notes)

    for note, weight in zip(notes, note_weights):
        note_audio = load_piano_sample(note)
        if len(note_audio) < len(chord_audio):
            note_audio = np.pad(note_audio, (0, len(chord_audio) - len(note_audio)))

        chord_audio += note_audio[: len(chord_audio)] * weight

    peak = np.max(np.abs(chord_audio))

    if peak > 0:
        chord_audio = chord_audio / peak

    return add_fade_out(chord_audio * config.VOLUME)


def create_progression_event_audio(event_item):
    """Create one chord with the soprano voice clearly above the texture."""

    notes = (event_item.bass_note, *event_item.top_notes)
    weights = (
        CHORD_PROGRESSION_BASS_VOLUME,
        *(CHORD_PROGRESSION_INNER_VOICE_VOLUME for _ in event_item.top_notes[:-1]),
        CHORD_PROGRESSION_TOP_VOICE_VOLUME,
    )

    return create_solid_event_audio(notes, weights=weights)


def create_progression_pass(example):
    """Create one pass through a three-chord progression."""

    parts = []

    for event_item in example.events:
        parts.append(create_progression_event_audio(event_item))
        parts.append(create_silence(CHORD_PROGRESSION_CHORD_GAP))

    return np.concatenate(parts)


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

    sf.write(
        file_path,
        create_chord_progression_waveform(example),
        config.SAMPLE_RATE,
    )


def create_chord_progression_question(level, example_index=None):
    """Create one chord-progression question and generated audio file."""

    examples = get_chord_progression_examples(level)

    if not examples:
        raise ValueError(f"No chord-progression examples are implemented for level {level}.")

    if example_index is None:
        example = random.choice(examples)
    else:
        example = examples[example_index % len(examples)]

    config.ensure_audio_folder(config.CHORD_PROGRESSION_AUDIO_FOLDER)

    key_label = make_safe_filename(example.key)
    filename = make_safe_filename(
        f"level_{level}_{key_label}_{example.id}_{CHORD_PROGRESSION_AUDIO_VERSION}.wav"
    )
    file_path = config.CHORD_PROGRESSION_AUDIO_FOLDER / filename

    if not file_path.exists():
        create_chord_progression_audio(example, file_path)

    return ChordProgressionQuestion(
        level=level,
        example_id=example.id,
        key=example.key,
        progression=example.progression,
        inversion_pattern=example.inversion_pattern,
        answer=example.progression.replace(" ", "-"),
        choices=CHORD_PROGRESSION_CHOICES,
        audio_file=f"chord_progressions/{filename}",
    )
