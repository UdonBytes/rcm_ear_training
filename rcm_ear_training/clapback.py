"""Approved-source clapback examples and audio generation."""

from dataclasses import dataclass
import random

import numpy as np
import soundfile as sf

from rcm_ear_training import config
from rcm_ear_training.audio import create_silence
from rcm_ear_training.questions import make_safe_filename
from rcm_ear_training.samples import load_piano_sample, resample_audio


CLAPBACK_TEMPO = 96
BEAT_SECONDS = 60 / CLAPBACK_TEMPO
CLAPBACK_AUDIO_VERSION = "approved_v1"
WOODBLOCK_FILE = config.PERCUSSION_FOLDER / "woodblock_low.wav"


@dataclass(frozen=True)
class MelodyEvent:
    note: str | None
    beats: float


@dataclass(frozen=True)
class ClapbackExample:
    id: str
    level: int
    time_signature: str
    key: str
    image_file: str
    draft_events: tuple[MelodyEvent, ...] = ()
    starting_chord_label: str = ""
    starting_chord_notes: tuple[str, ...] = ()
    playback_repetitions: int = 1
    status: str = "approved"
    measures: int = 2
    audio_version: str = CLAPBACK_AUDIO_VERSION


@dataclass(frozen=True)
class ClapbackQuestion:
    level: int
    example_id: str
    time_signature: str
    key: str
    image_file: str
    status: str
    audio_file: str
    playback_audio_file: str = ""
    starting_chord_label: str = ""


def create_events(notes, beats):
    """Create melody events from matching note and beat sequences."""

    return tuple(
        MelodyEvent(note, beat)
        for note, beat in zip(notes, beats)
    )


def create_level_1_playback_example(identifier, key, chord_notes, notes):
    """Create a level 1 playback example with the shared rhythm."""

    return ClapbackExample(
        id=f"play1-{identifier}",
        level=1,
        time_signature="4/4",
        key=key,
        image_file="",
        starting_chord_label=f"{key.title()} chord",
        starting_chord_notes=chord_notes,
        playback_repetitions=2,
        draft_events=create_events(notes, (1, 1, 1, 1, 4)),
        measures=2,
    )


LEVEL_1_CLAPBACK_EXAMPLES = (
    ClapbackExample(
        id="clap1-1",
        level=1,
        time_signature="4/4",
        key="D major",
        image_file="level_1/clap1-1.png",
        draft_events=(
            MelodyEvent("D4", 0.5),
            MelodyEvent("E4", 0.5),
            MelodyEvent("F#4", 1),
            MelodyEvent("F#4", 1),
            MelodyEvent("G4", 0.5),
            MelodyEvent("A4", 0.5),
            MelodyEvent("B4", 0.5),
            MelodyEvent("G4", 0.5),
            MelodyEvent("E4", 1),
            MelodyEvent("D4", 2),
        ),
    ),
    ClapbackExample(
        id="clap1-2",
        level=1,
        time_signature="3/4",
        key="F major",
        image_file="level_1/clap1-2.png",
        draft_events=(
            MelodyEvent("C5", 2),
            MelodyEvent("F5", 1),
            MelodyEvent("D5", 1.5),
            MelodyEvent("C5", 0.5),
            MelodyEvent("Bb4", 1),
            MelodyEvent("A4", 3),
        ),
        status="approved",
        measures=3,
    ),
    ClapbackExample(
        id="clap1-3",
        level=1,
        time_signature="3/4",
        key="C major",
        image_file="level_1/clap1-3.png",
        draft_events=(
            MelodyEvent("E4", 1.5),
            MelodyEvent("C4", 0.5),
            MelodyEvent("D4", 1),
            MelodyEvent("E4", 1),
            MelodyEvent("E4", 0.5),
            MelodyEvent("F4", 0.5),
            MelodyEvent("G4", 1),
            MelodyEvent("C4", 3),
        ),
        status="approved",
        measures=3,
    ),
    ClapbackExample(
        id="clap1-4",
        level=1,
        time_signature="4/4",
        key="D major",
        image_file="",
        draft_events=(
            MelodyEvent("D5", 1),
            MelodyEvent("B4", 0.5),
            MelodyEvent("A4", 0.5),
            MelodyEvent("B4", 1.5),
            MelodyEvent("C5", 0.5),
            MelodyEvent("A4", 1),
            MelodyEvent("F#4", 1),
            MelodyEvent("G4", 2),
        ),
        status="approved",
        measures=2,
    ),
    ClapbackExample(
        id="clap1-5",
        level=1,
        time_signature="3/4",
        key="F major",
        image_file="level_1/clap1-5.png",
        draft_events=(
            MelodyEvent("F4", 1),
            MelodyEvent("F4", 1),
            MelodyEvent("A4", 0.5),
            MelodyEvent("F4", 0.5),
            MelodyEvent("C4", 1),
            MelodyEvent("C4", 1.5),
            MelodyEvent("E4", 0.5),
            MelodyEvent("F4", 3),
        ),
        status="approved",
        measures=3,
    ),
    ClapbackExample(
        id="clap1-6",
        level=1,
        time_signature="4/4",
        key="F major",
        image_file="level_1/clap1-6-to-10.png",
        draft_events=(
            MelodyEvent("C4", 0.5),
            MelodyEvent("D4", 0.5),
            MelodyEvent("E4", 0.5),
            MelodyEvent("C4", 0.5),
            MelodyEvent("F4", 1.5),
            MelodyEvent("F4", 0.5),
            MelodyEvent("G4", 1),
            MelodyEvent("C5", 1),
            MelodyEvent("F4", 2),
        ),
        status="approved",
        measures=2,
    ),
    ClapbackExample(
        id="clap1-7",
        level=1,
        time_signature="3/4",
        key="C major",
        image_file="level_1/clap1-6-to-10.png",
        draft_events=(
            MelodyEvent("C4", 0.5),
            MelodyEvent("D4", 0.5),
            MelodyEvent("E4", 0.5),
            MelodyEvent("F4", 0.5),
            MelodyEvent("G4", 1),
            MelodyEvent("F4", 1.5),
            MelodyEvent("E4", 0.5),
            MelodyEvent("D4", 1),
            MelodyEvent("C4", 3),
        ),
        status="approved",
        measures=3,
    ),
    ClapbackExample(
        id="clap1-8",
        level=1,
        time_signature="4/4",
        key="F major",
        image_file="level_1/clap1-6-to-10.png",
        draft_events=(
            MelodyEvent("C5", 0.5),
            MelodyEvent("Bb4", 0.5),
            MelodyEvent("A4", 1),
            MelodyEvent("Bb4", 1.5),
            MelodyEvent("A4", 0.5),
            MelodyEvent("G4", 1),
            MelodyEvent("E4", 1),
            MelodyEvent("F4", 2),
        ),
        status="approved",
        measures=2,
    ),
    ClapbackExample(
        id="clap1-9",
        level=1,
        time_signature="3/4",
        key="F major",
        image_file="level_1/clap1-6-to-10.png",
        draft_events=(
            MelodyEvent("C4", 1),
            MelodyEvent("F4", 0.5),
            MelodyEvent("G4", 0.5),
            MelodyEvent("A4", 1),
            MelodyEvent("C4", 1.5),
            MelodyEvent("A4", 0.5),
            MelodyEvent("G4", 1),
            MelodyEvent("F4", 3),
        ),
        status="approved",
        measures=3,
    ),
    ClapbackExample(
        id="clap1-10",
        level=1,
        time_signature="4/4",
        key="D major",
        image_file="level_1/clap1-6-to-10.png",
        draft_events=(
            MelodyEvent("A4", 1.5),
            MelodyEvent("A4", 0.5),
            MelodyEvent("G4", 0.5),
            MelodyEvent("F#4", 0.5),
            MelodyEvent("E4", 0.5),
            MelodyEvent("D4", 0.5),
            MelodyEvent("C#4", 1),
            MelodyEvent("E4", 1),
            MelodyEvent("D4", 2),
        ),
        status="approved",
        measures=2,
    ),
    ClapbackExample(
        id="clap1-11",
        level=1,
        time_signature="3/4",
        key="D major",
        image_file="level_1/clap1-11-to-20.png",
        draft_events=(
            MelodyEvent("D5", 1),
            MelodyEvent("C5", 0.5),
            MelodyEvent("B4", 0.5),
            MelodyEvent("A4", 0.5),
            MelodyEvent("G4", 0.5),
            MelodyEvent("F#4", 1),
            MelodyEvent("E4", 0.5),
            MelodyEvent("D4", 0.5),
            MelodyEvent("F#4", 1),
            MelodyEvent("G4", 3),
        ),
        measures=3,
    ),
    ClapbackExample(
        id="clap1-12",
        level=1,
        time_signature="4/4",
        key="C major",
        image_file="level_1/clap1-11-to-20.png",
        draft_events=(
            MelodyEvent("C4", 1.5),
            MelodyEvent("D4", 0.5),
            MelodyEvent("E4", 1),
            MelodyEvent("G4", 1),
            MelodyEvent("F4", 1),
            MelodyEvent("D4", 1),
            MelodyEvent("C4", 2),
        ),
        measures=2,
    ),
    ClapbackExample(
        id="clap1-13",
        level=1,
        time_signature="4/4",
        key="D major",
        image_file="level_1/clap1-11-to-20.png",
        draft_events=(
            MelodyEvent("G4", 1),
            MelodyEvent("A4", 1),
            MelodyEvent("B4", 1),
            MelodyEvent("C5", 0.5),
            MelodyEvent("D5", 0.5),
            MelodyEvent("B4", 1.5),
            MelodyEvent("A4", 0.5),
            MelodyEvent("G4", 2),
        ),
        measures=2,
    ),
    ClapbackExample(
        id="clap1-14",
        level=1,
        time_signature="3/4",
        key="C major",
        image_file="level_1/clap1-11-to-20.png",
        draft_events=(
            MelodyEvent("G5", 0.5),
            MelodyEvent("A5", 0.5),
            MelodyEvent("G5", 1),
            MelodyEvent("G5", 1),
            MelodyEvent("E5", 1.5),
            MelodyEvent("E5", 0.5),
            MelodyEvent("D5", 1),
            MelodyEvent("C5", 3),
        ),
        measures=3,
    ),
    ClapbackExample(
        id="clap1-15",
        level=1,
        time_signature="4/4",
        key="F major",
        image_file="level_1/clap1-11-to-20.png",
        draft_events=(
            MelodyEvent("Bb3", 1.5),
            MelodyEvent("D4", 0.5),
            MelodyEvent("F4", 2),
            MelodyEvent("G4", 1),
            MelodyEvent("A4", 0.5),
            MelodyEvent("F4", 0.5),
            MelodyEvent("Bb4", 2),
        ),
        measures=2,
    ),
    ClapbackExample(
        id="clap1-16",
        level=1,
        time_signature="3/4",
        key="D major",
        image_file="level_1/clap1-11-to-20.png",
        draft_events=(
            MelodyEvent("D4", 0.5),
            MelodyEvent("F#4", 0.5),
            MelodyEvent("A4", 1),
            MelodyEvent("A4", 1),
            MelodyEvent("B4", 1.5),
            MelodyEvent("G4", 0.5),
            MelodyEvent("E4", 1),
            MelodyEvent("D4", 3),
        ),
        measures=3,
    ),
    ClapbackExample(
        id="clap1-17",
        level=1,
        time_signature="4/4",
        key="D major",
        image_file="level_1/clap1-11-to-20.png",
        draft_events=(
            MelodyEvent("B4", 0.5),
            MelodyEvent("C5", 0.5),
            MelodyEvent("D5", 1),
            MelodyEvent("G5", 1),
            MelodyEvent("D5", 1),
            MelodyEvent("C5", 1.5),
            MelodyEvent("A4", 0.5),
            MelodyEvent("B4", 2),
        ),
        measures=2,
    ),
    ClapbackExample(
        id="clap1-18",
        level=1,
        time_signature="3/4",
        key="C major",
        image_file="level_1/clap1-11-to-20.png",
        draft_events=(
            MelodyEvent("G4", 0.5),
            MelodyEvent("A4", 0.5),
            MelodyEvent("G4", 1),
            MelodyEvent("E4", 0.5),
            MelodyEvent("G4", 0.5),
            MelodyEvent("F4", 0.5),
            MelodyEvent("G4", 0.5),
            MelodyEvent("F4", 1),
            MelodyEvent("D4", 1),
            MelodyEvent("C4", 3),
        ),
        measures=3,
    ),
    ClapbackExample(
        id="clap1-19",
        level=1,
        time_signature="4/4",
        key="D major",
        image_file="level_1/clap1-11-to-20.png",
        draft_events=(
            MelodyEvent("D5", 1.5),
            MelodyEvent("C#5", 0.5),
            MelodyEvent("D5", 1),
            MelodyEvent("A4", 1),
            MelodyEvent("B4", 1),
            MelodyEvent("A4", 0.5),
            MelodyEvent("G4", 0.5),
            MelodyEvent("F#4", 2),
        ),
        measures=2,
    ),
    ClapbackExample(
        id="clap1-20",
        level=1,
        time_signature="3/4",
        key="D major",
        image_file="level_1/clap1-11-to-20.png",
        draft_events=(
            MelodyEvent("G4", 0.5),
            MelodyEvent("A4", 0.5),
            MelodyEvent("B4", 1),
            MelodyEvent("C5", 1),
            MelodyEvent("D5", 0.5),
            MelodyEvent("E5", 0.5),
            MelodyEvent("D5", 1.5),
            MelodyEvent("A4", 0.5),
            MelodyEvent("B4", 3),
        ),
        measures=3,
    ),
    ClapbackExample(
        id="clap1-21",
        level=1,
        time_signature="3/4",
        key="D major",
        image_file="level_1/clap1-21.png",
        draft_events=(
            MelodyEvent("D5", 1),
            MelodyEvent("G4", 0.5),
            MelodyEvent("A4", 0.5),
            MelodyEvent("B4", 1),
            MelodyEvent("C5", 1.5),
            MelodyEvent("F#4", 0.5),
            MelodyEvent("A4", 1),
            MelodyEvent("G4", 3),
        ),
        measures=3,
    ),
    ClapbackExample(
        id="clap1-22",
        level=1,
        time_signature="4/4",
        key="C major",
        image_file="level_1/clap1-22.png",
        draft_events=(
            MelodyEvent("C4", 1.5),
            MelodyEvent("E4", 0.5),
            MelodyEvent("G4", 1.5),
            MelodyEvent("E4", 0.5),
            MelodyEvent("F4", 0.5),
            MelodyEvent("E4", 0.5),
            MelodyEvent("D4", 0.5),
            MelodyEvent("E4", 0.5),
            MelodyEvent("C4", 2),
        ),
        measures=2,
    ),
    ClapbackExample(
        id="clap1-23",
        level=1,
        time_signature="4/4",
        key="F major",
        image_file="level_1/clap1-23.png",
        draft_events=(
            MelodyEvent("F5", 0.5),
            MelodyEvent("E5", 0.5),
            MelodyEvent("D5", 0.5),
            MelodyEvent("C5", 0.5),
            MelodyEvent("Bb4", 1.5),
            MelodyEvent("C5", 0.5),
            MelodyEvent("A4", 0.5),
            MelodyEvent("Bb4", 0.5),
            MelodyEvent("G4", 1),
            MelodyEvent("F4", 2),
        ),
        measures=2,
    ),
)

C_MAJOR_TRIAD_C4 = ("C4", "E4", "G4")
G_MAJOR_TRIAD_G4 = ("G4", "B4", "D5")
A_MINOR_TRIAD_A4 = ("A4", "C5", "E5")

LEVEL_1_PLAYBACK_EXAMPLES = (
    create_level_1_playback_example("1a", "C major", C_MAJOR_TRIAD_C4, ("C4", "D4", "E4", "G4", "E4")),
    create_level_1_playback_example("1b", "G major", G_MAJOR_TRIAD_G4, ("G4", "A4", "B4", "D5", "B4")),
    create_level_1_playback_example("1c", "A minor", A_MINOR_TRIAD_A4, ("A4", "B4", "C5", "E5", "C5")),
    create_level_1_playback_example("2a", "C major", C_MAJOR_TRIAD_C4, ("C4", "E4", "G4", "E4", "C4")),
    create_level_1_playback_example("2b", "G major", G_MAJOR_TRIAD_G4, ("G4", "B4", "D5", "B4", "G4")),
    create_level_1_playback_example("2c", "A minor", A_MINOR_TRIAD_A4, ("A4", "C5", "E5", "C5", "A4")),
    create_level_1_playback_example("3a", "C major", C_MAJOR_TRIAD_C4, ("C4", "E4", "E4", "F4", "G4")),
    create_level_1_playback_example("3b", "G major", G_MAJOR_TRIAD_G4, ("G4", "B4", "B4", "C5", "D5")),
    create_level_1_playback_example("3c", "A minor", A_MINOR_TRIAD_A4, ("A4", "C5", "C5", "D5", "E5")),
    create_level_1_playback_example("4a", "C major", C_MAJOR_TRIAD_C4, ("C4", "D4", "D4", "F4", "G4")),
    create_level_1_playback_example("4b", "G major", G_MAJOR_TRIAD_G4, ("G4", "A4", "A4", "C5", "D5")),
    create_level_1_playback_example("4c", "A minor", A_MINOR_TRIAD_A4, ("A4", "B4", "B4", "D5", "E5")),
    create_level_1_playback_example("5a", "C major", C_MAJOR_TRIAD_C4, ("C4", "E4", "F4", "G4", "E4")),
    create_level_1_playback_example("5b", "G major", G_MAJOR_TRIAD_G4, ("G4", "B4", "C5", "D5", "B4")),
    create_level_1_playback_example("5c", "A minor", A_MINOR_TRIAD_A4, ("A4", "C5", "D5", "E5", "C5")),
    create_level_1_playback_example("6a", "C major", C_MAJOR_TRIAD_C4, ("G4", "G4", "F4", "D4", "C4")),
    create_level_1_playback_example("6b", "G major", G_MAJOR_TRIAD_G4, ("D5", "D5", "C5", "A4", "G4")),
    create_level_1_playback_example("6c", "A minor", A_MINOR_TRIAD_A4, ("E5", "E5", "D5", "B4", "A4")),
    create_level_1_playback_example("7a", "C major", C_MAJOR_TRIAD_C4, ("C4", "D4", "D4", "E4", "G4")),
    create_level_1_playback_example("7b", "G major", G_MAJOR_TRIAD_G4, ("G4", "A4", "A4", "B4", "D5")),
    create_level_1_playback_example("7c", "A minor", A_MINOR_TRIAD_A4, ("A4", "B4", "B4", "C5", "E5")),
    create_level_1_playback_example("8a", "C major", C_MAJOR_TRIAD_C4, ("G4", "F4", "E4", "E4", "C4")),
    create_level_1_playback_example("8b", "G major", G_MAJOR_TRIAD_G4, ("D5", "C5", "B4", "B4", "G4")),
    create_level_1_playback_example("8c", "A minor", A_MINOR_TRIAD_A4, ("E5", "D5", "C5", "C5", "A4")),
    create_level_1_playback_example("9a", "C major", C_MAJOR_TRIAD_C4, ("C4", "E4", "F4", "E4", "C4")),
    create_level_1_playback_example("9b", "G major", G_MAJOR_TRIAD_G4, ("G4", "B4", "C5", "B4", "G4")),
    create_level_1_playback_example("9c", "A minor", A_MINOR_TRIAD_A4, ("A4", "C5", "D5", "C5", "A4")),
    create_level_1_playback_example("10a", "C major", C_MAJOR_TRIAD_C4, ("C4", "C4", "D4", "F4", "G4")),
    create_level_1_playback_example("10b", "G major", G_MAJOR_TRIAD_G4, ("G4", "G4", "A4", "C5", "D5")),
    create_level_1_playback_example("10c", "A minor", A_MINOR_TRIAD_A4, ("A4", "A4", "B4", "D5", "E5")),
    create_level_1_playback_example("11a", "C major", C_MAJOR_TRIAD_C4, ("G4", "E4", "D4", "E4", "C4")),
    create_level_1_playback_example("11b", "G major", G_MAJOR_TRIAD_G4, ("D5", "B4", "A4", "B4", "G4")),
    create_level_1_playback_example("11c", "A minor", A_MINOR_TRIAD_A4, ("E5", "C5", "B4", "C5", "A4")),
    create_level_1_playback_example("12a", "C major", C_MAJOR_TRIAD_C4, ("G4", "E4", "D4", "C4", "E4")),
    create_level_1_playback_example("12b", "G major", G_MAJOR_TRIAD_G4, ("D5", "B4", "A4", "G4", "B4")),
    create_level_1_playback_example("12c", "A minor", A_MINOR_TRIAD_A4, ("E5", "C5", "B4", "A4", "C5")),
    create_level_1_playback_example("13a", "C major", C_MAJOR_TRIAD_C4, ("G4", "E4", "C4", "D4", "E4")),
    create_level_1_playback_example("13b", "G major", G_MAJOR_TRIAD_G4, ("D5", "B4", "G4", "A4", "B4")),
    create_level_1_playback_example("13c", "A minor", A_MINOR_TRIAD_A4, ("E5", "C5", "A4", "B4", "C5")),
    create_level_1_playback_example("14a", "C major", C_MAJOR_TRIAD_C4, ("G4", "F4", "F4", "E4", "C4")),
    create_level_1_playback_example("14b", "G major", G_MAJOR_TRIAD_G4, ("D5", "C5", "C5", "B4", "G4")),
    create_level_1_playback_example("14c", "A minor", A_MINOR_TRIAD_A4, ("E5", "D5", "D5", "C5", "A4")),
    create_level_1_playback_example("15a", "C major", C_MAJOR_TRIAD_C4, ("C4", "D4", "G4", "F4", "E4")),
    create_level_1_playback_example("15b", "G major", G_MAJOR_TRIAD_G4, ("G4", "A4", "D5", "C5", "B4")),
    create_level_1_playback_example("15c", "A minor", A_MINOR_TRIAD_A4, ("A4", "B4", "E5", "D5", "C5")),
    create_level_1_playback_example("16a", "C major", C_MAJOR_TRIAD_C4, ("C4", "D4", "F4", "G4", "E4")),
    create_level_1_playback_example("16b", "G major", G_MAJOR_TRIAD_G4, ("G4", "A4", "C5", "D5", "B4")),
    create_level_1_playback_example("16c", "A minor", A_MINOR_TRIAD_A4, ("A4", "B4", "D5", "E5", "C5")),
    create_level_1_playback_example("17a", "C major", C_MAJOR_TRIAD_C4, ("C4", "E4", "E4", "G4", "E4")),
    create_level_1_playback_example("17b", "G major", G_MAJOR_TRIAD_G4, ("G4", "B4", "B4", "D5", "B4")),
    create_level_1_playback_example("17c", "A minor", A_MINOR_TRIAD_A4, ("A4", "C5", "C5", "E5", "C5")),
    create_level_1_playback_example("18a", "C major", C_MAJOR_TRIAD_C4, ("C4", "D4", "F4", "D4", "E4")),
    create_level_1_playback_example("18b", "G major", G_MAJOR_TRIAD_G4, ("G4", "A4", "C5", "A4", "B4")),
    create_level_1_playback_example("18c", "A minor", A_MINOR_TRIAD_A4, ("A4", "B4", "D5", "B4", "C5")),
    create_level_1_playback_example("19a", "C major", C_MAJOR_TRIAD_C4, ("G4", "G4", "F4", "E4", "C4")),
    create_level_1_playback_example("19b", "G major", G_MAJOR_TRIAD_G4, ("D5", "D5", "C5", "B4", "G4")),
    create_level_1_playback_example("19c", "A minor", A_MINOR_TRIAD_A4, ("E5", "E5", "D5", "C5", "A4")),
    create_level_1_playback_example("20a", "C major", C_MAJOR_TRIAD_C4, ("G4", "F4", "F4", "D4", "C4")),
    create_level_1_playback_example("20b", "G major", G_MAJOR_TRIAD_G4, ("D5", "C5", "C5", "A4", "G4")),
    create_level_1_playback_example("20c", "A minor", A_MINOR_TRIAD_A4, ("E5", "D5", "D5", "B4", "A4")),
    create_level_1_playback_example("21a", "C major", C_MAJOR_TRIAD_C4, ("G4", "F4", "D4", "F4", "E4")),
    create_level_1_playback_example("21b", "G major", G_MAJOR_TRIAD_G4, ("D5", "C5", "A4", "C5", "B4")),
    create_level_1_playback_example("21c", "A minor", A_MINOR_TRIAD_A4, ("E5", "D5", "B4", "D5", "C5")),
    create_level_1_playback_example("22a", "C major", C_MAJOR_TRIAD_C4, ("C4", "D4", "F4", "F4", "G4")),
    create_level_1_playback_example("22b", "G major", G_MAJOR_TRIAD_G4, ("G4", "A4", "C5", "C5", "D5")),
    create_level_1_playback_example("22c", "A minor", A_MINOR_TRIAD_A4, ("A4", "B4", "D5", "D5", "E5")),
)

LEVEL_5_CLAPBACK_PLAYBACK_EXAMPLES = (
    ClapbackExample(
        id="clap5-1",
        level=5,
        time_signature="4/4",
        key="E major",
        image_file="level_5/level5-clapback-playback-1-to-22.png",
        starting_chord_label="E major chord",
        starting_chord_notes=("E4", "G#4", "B4", "E5"),
        draft_events=(
            MelodyEvent("E4", 1.5), MelodyEvent("G#4", 0.5),
            MelodyEvent("B4", 0.5), MelodyEvent("A4", 0.5),
            MelodyEvent("G#4", 0.5), MelodyEvent("F#4", 0.5),
            MelodyEvent("G#4", 4),
        ),
        measures=2,
    ),
    ClapbackExample(
        id="clap5-2",
        level=5,
        time_signature="4/4",
        key="A major",
        image_file="level_5/level5-clapback-playback-1-to-22.png",
        starting_chord_label="A major chord",
        starting_chord_notes=("A3", "C#4", "E4", "A4"),
        draft_events=(
            MelodyEvent("C#4", 0.5), MelodyEvent("D4", 0.5),
            MelodyEvent("E4", 0.5), MelodyEvent("D4", 0.5),
            MelodyEvent("C#4", 1.5), MelodyEvent("E4", 0.5),
            MelodyEvent("A4", 4),
        ),
        measures=2,
    ),
    ClapbackExample(
        id="clap5-3",
        level=5,
        time_signature="4/4",
        key="E major",
        image_file="level_5/level5-clapback-playback-1-to-22.png",
        starting_chord_label="E major chord",
        starting_chord_notes=("E4", "G#4", "B4", "E5"),
        draft_events=(
            MelodyEvent("E5", 1), MelodyEvent("B4", 0.5),
            MelodyEvent("A4", 0.5), MelodyEvent("G#4", 0.5),
            MelodyEvent("A4", 0.5), MelodyEvent("B4", 1),
            MelodyEvent("E4", 4),
        ),
        measures=2,
    ),
    ClapbackExample(
        id="clap5-4",
        level=5,
        time_signature="4/4",
        key="E minor",
        image_file="level_5/level5-clapback-playback-1-to-22.png",
        starting_chord_label="E minor chord",
        starting_chord_notes=("E4", "G4", "B4", "E5"),
        draft_events=(
            MelodyEvent("G4", 1.5), MelodyEvent("B4", 0.5),
            MelodyEvent("E4", 0.5), MelodyEvent("F#4", 0.5),
            MelodyEvent("G4", 1), MelodyEvent("A4", 1),
            MelodyEvent("B4", 1), MelodyEvent("E5", 2),
        ),
        measures=2,
    ),
    ClapbackExample(
        id="clap5-5",
        level=5,
        time_signature="3/4",
        key="E major",
        image_file="level_5/level5-clapback-playback-1-to-22.png",
        starting_chord_label="E major chord",
        starting_chord_notes=("E4", "G#4", "B4", "E5"),
        draft_events=(
            MelodyEvent("G#4", 1), MelodyEvent("A4", 1),
            MelodyEvent("G#4", 0.5), MelodyEvent("F#4", 0.5),
            MelodyEvent("E4", 2), MelodyEvent("B4", 1),
            MelodyEvent("E5", 3),
        ),
        measures=3,
    ),
    ClapbackExample(
        id="clap5-6",
        level=5,
        time_signature="4/4",
        key="A major",
        image_file="level_5/level5-clapback-playback-1-to-22.png",
        starting_chord_label="A major chord",
        starting_chord_notes=("A4", "C#5", "E5", "A5"),
        draft_events=(
            MelodyEvent("E5", 0.5), MelodyEvent("D5", 0.5),
            MelodyEvent("C#5", 0.5), MelodyEvent("B4", 0.5),
            MelodyEvent("A4", 1), MelodyEvent("A5", 1),
            MelodyEvent("E5", 4),
        ),
        measures=2,
    ),
    ClapbackExample(
        id="clap5-7",
        level=5,
        time_signature="3/4",
        key="E major",
        image_file="level_5/level5-clapback-playback-1-to-22.png",
        starting_chord_label="E major chord",
        starting_chord_notes=("E4", "G#4", "B4", "E5"),
        draft_events=(
            MelodyEvent("B4", 1), MelodyEvent("E5", 1),
            MelodyEvent("G#4", 1), MelodyEvent("F#4", 1.5),
            MelodyEvent("G#4", 0.5), MelodyEvent("A4", 1),
            MelodyEvent("G#4", 3),
        ),
        measures=3,
    ),
    ClapbackExample(
        id="clap5-8",
        level=5,
        time_signature="3/4",
        key="E minor",
        image_file="level_5/level5-clapback-playback-1-to-22.png",
        starting_chord_label="E minor chord",
        starting_chord_notes=("E4", "G4", "B4", "E5"),
        draft_events=(
            MelodyEvent("G4", 1), MelodyEvent("G4", 0.5),
            MelodyEvent("A4", 0.5), MelodyEvent("B4", 1),
            MelodyEvent("F#4", 2), MelodyEvent("B4", 1),
            MelodyEvent("E4", 3),
        ),
        measures=3,
    ),
    ClapbackExample(
        id="clap5-9",
        level=5,
        time_signature="3/4",
        key="A minor",
        image_file="level_5/level5-clapback-playback-1-to-22.png",
        starting_chord_label="A minor chord",
        starting_chord_notes=("A4", "C5", "E5", "A5"),
        draft_events=(
            MelodyEvent("E5", 2), MelodyEvent("A4", 1),
            MelodyEvent("B4", 0.5), MelodyEvent("C5", 0.5),
            MelodyEvent("D5", 1.5), MelodyEvent("E5", 0.5),
            MelodyEvent("A5", 3),
        ),
        measures=3,
    ),
    ClapbackExample(
        id="clap5-10",
        level=5,
        time_signature="4/4",
        key="A major",
        image_file="level_5/level5-clapback-playback-1-to-22.png",
        starting_chord_label="A major chord",
        starting_chord_notes=("A3", "C#4", "E4", "A4"),
        draft_events=(
            MelodyEvent("A4", 1), MelodyEvent("D4", 0.5),
            MelodyEvent("C#4", 0.5), MelodyEvent("D4", 1),
            MelodyEvent("B3", 1), MelodyEvent("A3", 1.5),
            MelodyEvent("C#4", 0.5), MelodyEvent("E4", 2),
        ),
        measures=2,
    ),
    ClapbackExample(
        id="clap5-11",
        level=5,
        time_signature="3/4",
        key="A minor",
        image_file="level_5/level5-clapback-playback-1-to-22.png",
        starting_chord_label="A minor chord",
        starting_chord_notes=("A4", "C5", "E5", "A5"),
        draft_events=(
            MelodyEvent("A4", 2), MelodyEvent("E5", 1),
            MelodyEvent("D5", 0.5), MelodyEvent("C5", 0.5),
            MelodyEvent("B4", 1), MelodyEvent("C5", 1),
            MelodyEvent("A4", 3),
        ),
        measures=3,
    ),
    ClapbackExample(
        id="clap5-12",
        level=5,
        time_signature="4/4",
        key="E minor",
        image_file="level_5/level5-clapback-playback-1-to-22.png",
        starting_chord_label="E minor chord",
        starting_chord_notes=("E4", "G4", "B4", "E5"),
        draft_events=(
            MelodyEvent("B4", 1), MelodyEvent("E5", 1),
            MelodyEvent("B4", 0.5), MelodyEvent("A4", 0.5),
            MelodyEvent("F#4", 0.5), MelodyEvent("G4", 0.5),
            MelodyEvent("E4", 4),
        ),
        measures=2,
    ),
    ClapbackExample(
        id="clap5-13",
        level=5,
        time_signature="3/4",
        key="A minor",
        image_file="level_5/level5-clapback-playback-1-to-22.png",
        starting_chord_label="A minor chord",
        starting_chord_notes=("A4", "C5", "E5", "A5"),
        draft_events=(
            MelodyEvent("C5", 1), MelodyEvent("E5", 1),
            MelodyEvent("A4", 0.5), MelodyEvent("B4", 0.5),
            MelodyEvent("C5", 1.5), MelodyEvent("D5", 0.5),
            MelodyEvent("E5", 1), MelodyEvent("A5", 3),
        ),
        measures=3,
    ),
    ClapbackExample(
        id="clap5-14",
        level=5,
        time_signature="4/4",
        key="A major",
        image_file="level_5/level5-clapback-playback-1-to-22.png",
        starting_chord_label="A major chord",
        starting_chord_notes=("A3", "C#4", "E4", "A4"),
        draft_events=(
            MelodyEvent("C#4", 0.5), MelodyEvent("D4", 0.5),
            MelodyEvent("E4", 1), MelodyEvent("A4", 1),
            MelodyEvent("E4", 0.5), MelodyEvent("B3", 0.5),
            MelodyEvent("C#4", 4),
        ),
        measures=2,
    ),
    ClapbackExample(
        id="clap5-15",
        level=5,
        time_signature="4/4",
        key="A minor",
        image_file="level_5/level5-clapback-playback-1-to-22.png",
        starting_chord_label="A minor chord",
        starting_chord_notes=("A4", "C5", "E5", "A5"),
        draft_events=(
            MelodyEvent("A4", 1.5), MelodyEvent("C5", 0.5),
            MelodyEvent("B4", 0.5), MelodyEvent("D5", 0.5),
            MelodyEvent("C5", 0.5), MelodyEvent("E5", 0.5),
            MelodyEvent("A5", 4),
        ),
        measures=2,
    ),
    ClapbackExample(
        id="clap5-16",
        level=5,
        time_signature="4/4",
        key="E minor",
        image_file="level_5/level5-clapback-playback-1-to-22.png",
        starting_chord_label="E minor chord",
        starting_chord_notes=("E4", "G4", "B4", "E5"),
        draft_events=(
            MelodyEvent("G4", 1), MelodyEvent("B4", 1),
            MelodyEvent("E5", 1.5), MelodyEvent("B4", 0.5),
            MelodyEvent("A4", 0.5), MelodyEvent("B4", 0.5),
            MelodyEvent("F#4", 1), MelodyEvent("E4", 2),
        ),
        measures=2,
    ),
    ClapbackExample(
        id="clap5-17",
        level=5,
        time_signature="3/4",
        key="A minor",
        image_file="level_5/level5-clapback-playback-1-to-22.png",
        starting_chord_label="A minor chord",
        starting_chord_notes=("A4", "C5", "E5", "A5"),
        draft_events=(
            MelodyEvent("A5", 2), MelodyEvent("E5", 1),
            MelodyEvent("B4", 0.5), MelodyEvent("C5", 0.5),
            MelodyEvent("D5", 1), MelodyEvent("E5", 1),
            MelodyEvent("A4", 3),
        ),
        measures=3,
    ),
    ClapbackExample(
        id="clap5-18",
        level=5,
        time_signature="3/4",
        key="A major",
        image_file="level_5/level5-clapback-playback-1-to-22.png",
        starting_chord_label="A major chord",
        starting_chord_notes=("A3", "C#4", "E4", "A4"),
        draft_events=(
            MelodyEvent("E4", 1), MelodyEvent("E4", 0.5),
            MelodyEvent("A4", 0.5), MelodyEvent("E4", 1),
            MelodyEvent("D4", 1.5), MelodyEvent("B3", 0.5),
            MelodyEvent("C#4", 1), MelodyEvent("A3", 3),
        ),
        measures=3,
    ),
    ClapbackExample(
        id="clap5-19",
        level=5,
        time_signature="3/4",
        key="E major",
        image_file="level_5/level5-clapback-playback-1-to-22.png",
        starting_chord_label="E major chord",
        starting_chord_notes=("E4", "G#4", "B4", "E5"),
        draft_events=(
            MelodyEvent("E4", 2), MelodyEvent("G#4", 1),
            MelodyEvent("B4", 1.5), MelodyEvent("E5", 0.5),
            MelodyEvent("B4", 1), MelodyEvent("A4", 2),
            MelodyEvent("F#4", 1), MelodyEvent("G#4", 3),
        ),
        measures=4,
    ),
    ClapbackExample(
        id="clap5-20",
        level=5,
        time_signature="3/4",
        key="E minor",
        image_file="level_5/level5-clapback-playback-1-to-22.png",
        starting_chord_label="E minor chord",
        starting_chord_notes=("E4", "G4", "B4", "E5"),
        draft_events=(
            MelodyEvent("E4", 1), MelodyEvent("E4", 1.5),
            MelodyEvent("F#4", 0.5), MelodyEvent("G4", 0.5),
            MelodyEvent("A4", 0.5), MelodyEvent("B4", 1),
            MelodyEvent("D#5", 1), MelodyEvent("E5", 3),
        ),
        measures=3,
    ),
    ClapbackExample(
        id="clap5-21",
        level=5,
        time_signature="3/4",
        key="E major",
        image_file="level_5/level5-clapback-playback-1-to-22.png",
        starting_chord_label="E major chord",
        starting_chord_notes=("E4", "G#4", "B4", "E5"),
        draft_events=(
            MelodyEvent("G#4", 1.5), MelodyEvent("A4", 0.5),
            MelodyEvent("G#4", 1), MelodyEvent("E5", 2),
            MelodyEvent("B4", 0.5), MelodyEvent("A4", 0.5),
            MelodyEvent("G#4", 3),
        ),
        measures=3,
    ),
    ClapbackExample(
        id="clap5-22",
        level=5,
        time_signature="4/4",
        key="A minor",
        image_file="level_5/level5-clapback-playback-1-to-22.png",
        starting_chord_label="A minor chord",
        starting_chord_notes=("A4", "C5", "E5", "A5"),
        draft_events=(
            MelodyEvent("A5", 1), MelodyEvent("E5", 1),
            MelodyEvent("E5", 1.5), MelodyEvent("C5", 0.5),
            MelodyEvent("B4", 0.5), MelodyEvent("C5", 0.5),
            MelodyEvent("D5", 1), MelodyEvent("C5", 2),
        ),
        measures=2,
    ),
)


def get_clapback_examples(level):
    """Return source clapback examples for a level."""

    if level == 1:
        return LEVEL_1_CLAPBACK_EXAMPLES

    if level == 5:
        return LEVEL_5_CLAPBACK_PLAYBACK_EXAMPLES

    return ()


def get_playable_clapback_examples(level):
    """Return examples that have audio data available for review or practice."""

    return tuple(
        example
        for example in get_clapback_examples(level)
        if example.draft_events
    )


def get_playback_examples(level):
    """Return source playback examples for a level."""

    if level == 1:
        return LEVEL_1_PLAYBACK_EXAMPLES

    return ()


def get_playable_playback_examples(level):
    """Return playable playback examples for a level."""

    return tuple(
        example
        for example in get_playback_examples(level)
        if example.draft_events
    )


def choose_unplayed_example_index(examples, played_example_ids, previous_key=None):
    """Choose a random example index, preferring unplayed and a different key."""

    unplayed_indexes = [
        index
        for index, example in enumerate(examples)
        if example.id not in played_example_ids
    ]
    candidate_indexes = unplayed_indexes or list(range(len(examples)))
    different_key_indexes = [
        index
        for index in candidate_indexes
        if examples[index].key != previous_key
    ]

    if different_key_indexes:
        return random.choice(different_key_indexes)

    return random.choice(candidate_indexes)


def beats_per_measure(time_signature):
    """Return the number of beats in one measure."""

    numerator, _ = time_signature.split("/")
    return int(numerator)


def total_event_beats(events):
    """Return total event duration in beats."""

    return sum(event.beats for event in events)


def events_with_completed_length(example):
    """Add silence so playback occupies the written measure count."""

    written_beats = beats_per_measure(example.time_signature) * example.measures
    missing_beats = written_beats - total_event_beats(example.draft_events)

    if missing_beats < 0:
        raise ValueError(f"{example.id} is longer than its written measure count.")

    if missing_beats == 0:
        return example.draft_events

    return example.draft_events + (MelodyEvent(None, missing_beats),)


def trim_or_pad_audio(audio, duration):
    """Trim or pad audio to an exact duration."""

    target_length = int(config.SAMPLE_RATE * duration)

    if len(audio) > target_length:
        return audio[:target_length]

    return np.concatenate([audio, np.zeros(target_length - len(audio))])


def load_woodblock():
    """Load and normalize the count-in woodblock."""

    audio, sample_rate = sf.read(WOODBLOCK_FILE)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = resample_audio(audio, sample_rate, config.SAMPLE_RATE)
    peak = np.max(np.abs(audio))

    if peak > 0:
        audio = audio / peak

    return audio * 0.5


def create_count_in(time_signature):
    """Create a one-measure woodblock count-in."""

    hit = trim_or_pad_audio(load_woodblock(), 0.18)
    tail = create_silence(max(BEAT_SECONDS - 0.18, 0))
    beat = np.concatenate([hit, tail])
    return np.concatenate([beat for _ in range(beats_per_measure(time_signature))])


def create_melody_audio(example):
    """Create monophonic piano audio for one approved example."""

    segments = []

    for event in events_with_completed_length(example):
        duration = event.beats * BEAT_SECONDS

        if event.note is None:
            segments.append(create_silence(duration))
        else:
            segments.append(trim_or_pad_audio(load_piano_sample(event.note), duration))

    return np.concatenate(segments)


def create_solid_chord_audio(notes, duration=BEAT_SECONDS * 2):
    """Create a solid starting chord trimmed to a predictable duration."""

    if not notes:
        return np.array([], dtype=float)

    chord_audio = np.zeros(int(config.SAMPLE_RATE * duration))

    for note in notes:
        chord_audio += trim_or_pad_audio(load_piano_sample(note), duration)

    peak = np.max(np.abs(chord_audio))

    if peak > 0:
        chord_audio = chord_audio / peak

    return chord_audio * 0.75


def create_starting_chord_intro(example):
    """Create the starting chord with a small breath before the test begins."""

    if not example.starting_chord_notes:
        return np.array([], dtype=float)

    return np.concatenate(
        [
            create_solid_chord_audio(example.starting_chord_notes),
            create_silence(BEAT_SECONDS),
        ]
    )


def create_clapback_waveform(example):
    """Create count-in, melody, one-bar rest, then melody again."""

    intro = create_starting_chord_intro(example)
    count_in = create_count_in(example.time_signature)
    melody = create_melody_audio(example)
    rest = create_silence(beats_per_measure(example.time_signature) * BEAT_SECONDS)
    return np.concatenate([intro, count_in, melody, rest, melody])


def create_playback_waveform(example):
    """Create starting chord and the configured number of melody repetitions."""

    intro = create_starting_chord_intro(example)
    melody = create_melody_audio(example)
    melody_repetitions = []

    for repetition in range(example.playback_repetitions):
        if repetition > 0:
            melody_repetitions.append(
                create_silence(beats_per_measure(example.time_signature) * BEAT_SECONDS)
            )

        melody_repetitions.append(melody)

    return np.concatenate([intro, *melody_repetitions])


def create_clapback_audio(example, file_path):
    """Write a clapback WAV."""

    sf.write(file_path, create_clapback_waveform(example), config.SAMPLE_RATE)


def create_playback_audio(example, file_path):
    """Write a playback WAV."""

    sf.write(file_path, create_playback_waveform(example), config.SAMPLE_RATE)


def create_clapback_question(level, example_index=0):
    """Create one clapback question."""

    examples = get_playable_clapback_examples(level)

    if not examples:
        raise ValueError(f"No clapback examples for level {level}.")

    config.ensure_audio_folder(config.CLAPBACK_AUDIO_FOLDER)
    example = examples[example_index % len(examples)]
    filename = make_safe_filename(
        f"level_{level}_{example.id}_clapback_{example.audio_version}.wav"
    )
    file_path = config.CLAPBACK_AUDIO_FOLDER / filename
    playback_filename = make_safe_filename(
        f"level_{level}_{example.id}_playback_{example.audio_version}.wav"
    )
    playback_file_path = config.CLAPBACK_AUDIO_FOLDER / playback_filename

    if not file_path.exists():
        create_clapback_audio(example, file_path)

    if example.starting_chord_notes and not playback_file_path.exists():
        create_playback_audio(example, playback_file_path)

    return ClapbackQuestion(
        level=level,
        example_id=example.id,
        time_signature=example.time_signature,
        key=example.key,
        image_file=example.image_file,
        status=example.status,
        audio_file=f"clapback/{filename}",
        playback_audio_file=(
            f"clapback/{playback_filename}" if example.starting_chord_notes else ""
        ),
        starting_chord_label=example.starting_chord_label,
    )


def create_playback_question(level, example_index=0):
    """Create one playback question."""

    examples = get_playable_playback_examples(level)

    if not examples:
        raise ValueError(f"No playback examples for level {level}.")

    config.ensure_audio_folder(config.CLAPBACK_AUDIO_FOLDER)
    example = examples[example_index % len(examples)]
    filename = make_safe_filename(
        f"level_{level}_{example.id}_playback_{example.audio_version}.wav"
    )
    file_path = config.CLAPBACK_AUDIO_FOLDER / filename

    if not file_path.exists():
        create_playback_audio(example, file_path)

    return ClapbackQuestion(
        level=level,
        example_id=example.id,
        time_signature=example.time_signature,
        key=example.key,
        image_file=example.image_file,
        status=example.status,
        audio_file=f"clapback/{filename}",
        starting_chord_label=example.starting_chord_label,
    )
