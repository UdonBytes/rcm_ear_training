"""Interval audio synthesis."""

import numpy as np
import soundfile as sf

from rcm_ear_training.config import (
    CONNECTED_NOTE_SPACING,
    PAUSE_DURATION,
    SAMPLE_RATE,
    VOLUME,
)
from rcm_ear_training.samples import load_piano_sample


CONNECTED_ASCENDING_DESCENDING = "connected_ascending_descending"
MELODIC_THEN_HARMONIC = "melodic_then_harmonic"


def create_silence(duration):
    """Create silence between notes."""

    return np.zeros(int(SAMPLE_RATE * duration))


def add_fade_out(audio):
    """Add a short fade out to a generated clip."""

    faded_audio = audio.copy()
    fade_length = int(SAMPLE_RATE * 0.04)

    if len(faded_audio) > fade_length:
        fade_out = np.linspace(1, 0, fade_length)
        faded_audio[-fade_length:] *= fade_out

    return faded_audio


def create_harmonic_audio(first_note, second_note):
    """Create one harmonic interval by playing both notes together."""

    first_audio = load_piano_sample(first_note)
    second_audio = load_piano_sample(second_note)
    harmonic_audio = first_audio + second_audio
    max_value = np.max(np.abs(harmonic_audio))

    if max_value > 0:
        harmonic_audio = harmonic_audio / max_value

    harmonic_audio = harmonic_audio * VOLUME
    return add_fade_out(harmonic_audio)


def create_interval_waveform(level, start_note, end_note, direction):
    """Create an interval waveform for a question."""

    pause = create_silence(PAUSE_DURATION)

    if level <= 4:
        transition = create_silence(CONNECTED_NOTE_SPACING)

        return np.concatenate(
            [
                load_piano_sample(start_note),
                transition,
                load_piano_sample(end_note),
                transition,
                load_piano_sample(start_note),
            ]
        )

    if direction == "ascending":
        melodic_first = load_piano_sample(start_note)
        melodic_second = load_piano_sample(end_note)
    else:
        melodic_first = load_piano_sample(end_note)
        melodic_second = load_piano_sample(start_note)

    harmonic_interval = create_harmonic_audio(start_note, end_note)
    return np.concatenate(
        [
            melodic_first,
            pause,
            melodic_second,
            pause,
            harmonic_interval,
        ]
    )


def create_interval_audio(level, start_note, end_note, direction, file_path):
    """Create the interval audio file."""

    sf.write(file_path, create_interval_waveform(level, start_note, end_note, direction), SAMPLE_RATE)
