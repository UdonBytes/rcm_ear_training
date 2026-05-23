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


def get_clapback_examples(level):
    """Return source clapback examples for a level."""

    if level != 1:
        return ()

    return LEVEL_1_CLAPBACK_EXAMPLES


def get_playable_clapback_examples(level):
    """Return examples that have audio data available for review or practice."""

    return tuple(
        example
        for example in get_clapback_examples(level)
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


def create_clapback_waveform(example):
    """Create count-in, melody, one-bar rest, then melody again."""

    count_in = create_count_in(example.time_signature)
    melody = create_melody_audio(example)
    rest = create_silence(beats_per_measure(example.time_signature) * BEAT_SECONDS)
    return np.concatenate([count_in, melody, rest, melody])


def create_clapback_audio(example, file_path):
    """Write a clapback WAV."""

    sf.write(file_path, create_clapback_waveform(example), config.SAMPLE_RATE)


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

    if not file_path.exists():
        create_clapback_audio(example, file_path)

    return ClapbackQuestion(
        level=level,
        example_id=example.id,
        time_signature=example.time_signature,
        key=example.key,
        image_file=example.image_file,
        status=example.status,
        audio_file=f"clapback/{filename}",
    )
