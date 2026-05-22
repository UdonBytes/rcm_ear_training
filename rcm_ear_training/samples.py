"""Piano sample discovery and loading."""

import numpy as np
import soundfile as sf

from rcm_ear_training.config import (
    ATTACK_PREROLL_SECONDS,
    ATTACK_THRESHOLD_RATIO,
    NOTE_DURATION,
    PIANO_SAMPLE_FOLDER,
    SAMPLE_RATE,
    VOLUME,
)
from rcm_ear_training.theory import (
    INTERVALS,
    SAMPLE_HIGH_NOTE,
    SAMPLE_LOW_NOTE,
    make_sample_note_name,
    midi_to_note,
    note_to_midi,
)


def find_sample_path(note):
    """Find the matching piano sample file for an app note name."""

    sample_note = make_sample_note_name(note)
    possible_paths = [
        PIANO_SAMPLE_FOLDER / f"Piano.mf.{sample_note}.aiff",
        PIANO_SAMPLE_FOLDER / f"Piano.mf.{sample_note}.aif",
        PIANO_SAMPLE_FOLDER / f"piano.mf.{sample_note}.aiff",
        PIANO_SAMPLE_FOLDER / f"piano.mf.{sample_note}.aif",
    ]

    for sample_path in possible_paths:
        if sample_path.exists():
            return sample_path

    available_files = sorted(file.name for file in PIANO_SAMPLE_FOLDER.glob("*"))
    raise FileNotFoundError(
        f"Missing piano sample for {note}. Available files: {available_files}"
    )


def note_has_sample(note):
    """Return whether a sample exists for this note."""

    try:
        find_sample_path(note)
    except FileNotFoundError:
        return False

    return True


def get_possible_starting_notes(interval_name):
    """Find starting notes that fit inside the available sample range."""

    interval_distance = INTERVALS[interval_name]
    low_midi = note_to_midi(SAMPLE_LOW_NOTE)
    high_midi = note_to_midi(SAMPLE_HIGH_NOTE)
    possible_notes = []

    for midi_number in range(low_midi, high_midi + 1):
        start_note = midi_to_note(midi_number)
        end_midi = midi_number + interval_distance
        end_note = midi_to_note(end_midi)

        if (
            end_midi <= high_midi
            and note_has_sample(start_note)
            and note_has_sample(end_note)
        ):
            possible_notes.append(start_note)

    return possible_notes


def resample_audio(audio, original_rate, target_rate):
    """Resample audio if the piano sample rate does not match the app sample rate."""

    if original_rate == target_rate:
        return audio

    original_positions = np.linspace(0, 1, len(audio))
    target_length = int(len(audio) * target_rate / original_rate)
    target_positions = np.linspace(0, 1, target_length)
    return np.interp(target_positions, original_positions, audio)


def trim_or_pad_audio(audio, duration):
    """Trim or pad audio to a fixed duration."""

    target_length = int(SAMPLE_RATE * duration)

    if len(audio) > target_length:
        return audio[:target_length]

    padding = np.zeros(target_length - len(audio))
    return np.concatenate([audio, padding])


def trim_leading_silence(audio):
    """Align a sample so its audible attack starts near the beginning."""

    peak = np.max(np.abs(audio))

    if peak == 0:
        return audio

    threshold = peak * ATTACK_THRESHOLD_RATIO
    audible_indexes = np.where(np.abs(audio) >= threshold)[0]

    if len(audible_indexes) == 0:
        return audio

    preroll_samples = int(SAMPLE_RATE * ATTACK_PREROLL_SECONDS)
    start_index = max(int(audible_indexes[0]) - preroll_samples, 0)
    return audio[start_index:]


def add_fade_out(audio):
    """Add a short fade out so the note does not click at the end."""

    faded_audio = audio.copy()
    fade_length = int(SAMPLE_RATE * 0.04)

    if len(faded_audio) > fade_length:
        fade_out = np.linspace(1, 0, fade_length)
        faded_audio[-fade_length:] *= fade_out

    return faded_audio


def load_piano_sample(note):
    """Load and normalize one real piano sample."""

    sample_path = find_sample_path(note)
    audio, sample_rate = sf.read(sample_path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = resample_audio(audio, sample_rate, SAMPLE_RATE)
    audio = trim_leading_silence(audio)
    audio = trim_or_pad_audio(audio, NOTE_DURATION)

    max_value = np.max(np.abs(audio))

    if max_value > 0:
        audio = audio / max_value

    audio = audio * VOLUME
    return add_fade_out(audio)
