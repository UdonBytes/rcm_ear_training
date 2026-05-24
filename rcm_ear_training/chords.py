"""RCM chord ear-test requirements."""

from dataclasses import dataclass
import random

import numpy as np
import soundfile as sf

from rcm_ear_training.audio import add_fade_out, create_silence
from rcm_ear_training import config
from rcm_ear_training.questions import make_safe_filename
from rcm_ear_training.samples import load_piano_sample, note_has_sample
from rcm_ear_training.theory import (
    SAMPLE_HIGH_NOTE,
    SAMPLE_LOW_NOTE,
    midi_to_note,
    note_to_midi,
)


MAJOR_TRIAD = "Major triad"
MINOR_TRIAD = "Minor triad"
DOMINANT_7TH = "Dominant 7th"
DIMINISHED_7TH = "Diminished 7th"
AUGMENTED_TRIAD = "Augmented triad"

ROOT = "Root"
THIRD = "Third"
FIFTH = "Fifth"

CHORD_FORMULAS = {
    MAJOR_TRIAD: (0, 4, 7),
    MINOR_TRIAD: (0, 3, 7),
    DOMINANT_7TH: (0, 4, 7, 10),
    DIMINISHED_7TH: (0, 3, 6, 9),
    AUGMENTED_TRIAD: (0, 4, 8),
}

CHORD_DISPLAY_NAMES = {
    MAJOR_TRIAD: "Major",
    MINOR_TRIAD: "Minor",
    DOMINANT_7TH: "Dominant 7th",
    DIMINISHED_7TH: "Diminish 7th",
    AUGMENTED_TRIAD: "Augmented (Triad)",
}

ADVANCED_CHORD_DISPLAY_NAMES = {
    **CHORD_DISPLAY_NAMES,
    MAJOR_TRIAD: "Major (Triad)",
    MINOR_TRIAD: "Minor (Triad)",
}

TONE_CHOICES = (ROOT, THIRD, FIFTH)
BROKEN_CHORD_NOTE_DURATION = getattr(config, "BROKEN_CHORD_NOTE_DURATION", 0.55)
CHORD_EVENT_DURATION = getattr(config, "CHORD_EVENT_DURATION", 1.4)


@dataclass(frozen=True)
class ChordRequirement:
    level_id: str
    label: str
    question_types: tuple[str, ...]
    chord_qualities: tuple[str, ...]
    positions: tuple[str, ...]
    playback: str
    source_pages: tuple[int, ...]
    implementation_notes: tuple[str, ...]


@dataclass(frozen=True)
class ChordQuestion:
    level: int
    root_note: str
    chord_quality: str
    answer: str
    choices: list[str]
    audio_file: str
    prompt: str
    question_type: str
    tone_audio_file: str = ""
    target_tone: str | None = None
    quality_answer: str | None = None
    quality_choices: tuple[str, ...] = ()
    tone_answer: str | None = None
    tone_choices: tuple[str, ...] = ()


CHORD_REQUIREMENTS = {
    "prep_a": ChordRequirement(
        level_id="prep_a",
        label="Preparatory A",
        question_types=("Quality identification",),
        chord_qualities=("Major triad", "Minor triad"),
        positions=("Root position",),
        playback="First five notes of a major or minor scale, then tonic triad in solid form.",
        source_pages=(9,),
        implementation_notes=(
            "Generate a short scalar context before the blocked tonic triad.",
            "Ask students to identify major or minor.",
        ),
    ),
    "prep_b": ChordRequirement(
        level_id="prep_b",
        label="Preparatory B",
        question_types=("Quality identification",),
        chord_qualities=("Major triad", "Minor triad"),
        positions=("Root position",),
        playback="First five notes of a major or minor scale, then tonic triad in solid form.",
        source_pages=(14,),
        implementation_notes=(
            "Same chord-test format as Preparatory A, with broader key context later.",
        ),
    ),
    "level_1": ChordRequirement(
        level_id="level_1",
        label="Level 1",
        question_types=("Quality identification",),
        chord_qualities=("Major triad", "Minor triad"),
        positions=("Root position",),
        playback="Broken triad, then solid/blocked triad.",
        source_pages=(20,),
        implementation_notes=(
            "Generate both broken and blocked versions before asking major or minor.",
        ),
    ),
    "level_2": ChordRequirement(
        level_id="level_2",
        label="Level 2",
        question_types=("Quality identification",),
        chord_qualities=("Major triad", "Minor triad"),
        positions=("Root position",),
        playback="Solid/blocked triad.",
        source_pages=(26,),
        implementation_notes=(
            "Ask major or minor after one blocked root-position triad.",
        ),
    ),
    "level_3": ChordRequirement(
        level_id="level_3",
        label="Level 3",
        question_types=("Quality identification", "Chord-tone identification"),
        chord_qualities=("Major triad", "Minor triad"),
        positions=("Root position",),
        playback="Solid/blocked triad for quality; broken triad before single-note identification.",
        source_pages=(32,),
        implementation_notes=(
            "Quality question: major or minor.",
            "Tone question: identify root, third, or fifth after a broken triad.",
        ),
    ),
    "level_4": ChordRequirement(
        level_id="level_4",
        label="Level 4",
        question_types=("Quality identification", "Chord-tone identification"),
        chord_qualities=("Major triad", "Minor triad"),
        positions=("Root position",),
        playback="Solid/blocked triad for quality; broken triad before single-note identification.",
        source_pages=(39,),
        implementation_notes=(
            "Same chord-test types as Level 3.",
        ),
    ),
    "level_5": ChordRequirement(
        level_id="level_5",
        label="Level 5",
        question_types=("Quality identification",),
        chord_qualities=("Major triad", "Minor triad", "Dominant 7th"),
        positions=("Root position", "Close position"),
        playback="Solid/blocked chord in close position.",
        source_pages=(46,),
        implementation_notes=(
            "Add dominant seventh quality to major/minor triad choices.",
        ),
    ),
    "level_6": ChordRequirement(
        level_id="level_6",
        label="Level 6",
        question_types=("Quality identification",),
        chord_qualities=("Major triad", "Minor triad", "Dominant 7th", "Diminished 7th"),
        positions=("Root position", "Close position"),
        playback="Solid/blocked chord in close position.",
        source_pages=(54,),
        implementation_notes=(
            "Add diminished seventh quality.",
        ),
    ),
    "level_7": ChordRequirement(
        level_id="level_7",
        label="Level 7",
        question_types=("Quality identification",),
        chord_qualities=(
            "Major triad",
            "Minor triad",
            "Dominant 7th",
            "Diminished 7th",
            "Augmented triad",
        ),
        positions=("Root position", "Close position"),
        playback="Solid/blocked chord in close position.",
        source_pages=(62,),
        implementation_notes=(
            "Add augmented triad to the quality choices.",
        ),
    ),
    "level_8": ChordRequirement(
        level_id="level_8",
        label="Level 8",
        question_types=("Quality identification",),
        chord_qualities=(
            "Major triad",
            "Minor triad",
            "Dominant 7th",
            "Diminished 7th",
            "Augmented triad",
        ),
        positions=("Root position", "Close position"),
        playback="Solid/blocked chord in close position.",
        source_pages=(71,),
        implementation_notes=(
            "Same chord-quality pool as Level 7.",
        ),
    ),
    "level_9": ChordRequirement(
        level_id="level_9",
        label="Level 9",
        question_types=("Quality identification",),
        chord_qualities=(
            "Major four-note chord",
            "Minor four-note chord",
            "Augmented triad",
            "Dominant 7th",
            "Diminished 7th",
        ),
        positions=("Root position", "1st inversion", "Close position"),
        playback="Solid/blocked chord in close position.",
        source_pages=(81,),
        implementation_notes=(
            "Future expansion: include root and first-inversion four-note major/minor chords.",
        ),
    ),
    "level_10": ChordRequirement(
        level_id="level_10",
        label="Level 10",
        question_types=("Quality identification",),
        chord_qualities=(
            "Major four-note chord",
            "Minor four-note chord",
            "Augmented triad",
            "Dominant 7th",
            "Diminished 7th",
            "Major-major 7th",
            "Minor-minor 7th",
        ),
        positions=("Root position", "1st inversion", "Close position"),
        playback="Solid/blocked chord in close position.",
        source_pages=(92,),
        implementation_notes=(
            "Future expansion: add major-major seventh and minor-minor seventh qualities.",
        ),
    ),
    "arct": ChordRequirement(
        level_id="arct",
        label="ARCT",
        question_types=("Advanced chord identification",),
        chord_qualities=(),
        positions=(),
        playback="To be confirmed from the ARCT-specific syllabus requirements.",
        source_pages=(),
        implementation_notes=(
            "Future expansion: confirm ARCT aural/chord requirements separately before implementing.",
        ),
    ),
}


def get_chord_requirement(level_id):
    """Return the chord requirement for a level id."""

    return CHORD_REQUIREMENTS[level_id]


def chord_notes(root_note, chord_quality):
    """Return note names for a root-position chord."""

    root_midi = note_to_midi(root_note)
    return [
        midi_to_note(root_midi + semitones)
        for semitones in CHORD_FORMULAS[chord_quality]
    ]


def chord_quality_choices(level):
    """Return chord qualities tested at a level."""

    return get_chord_requirement(f"level_{level}").chord_qualities


def chord_display_name(chord_quality, level):
    """Return the user-facing chord quality label for a level."""

    if level >= 5:
        return ADVANCED_CHORD_DISPLAY_NAMES[chord_quality]

    return CHORD_DISPLAY_NAMES[chord_quality]


def chord_question_type(level):
    """Return the chord question type for a level."""

    if level in (3, 4):
        return "quality_and_tone"

    return "quality"


def possible_chord_roots(chord_quality):
    """Return roots whose chord tones all have samples in range."""

    low_midi = note_to_midi(SAMPLE_LOW_NOTE)
    high_midi = note_to_midi(SAMPLE_HIGH_NOTE)
    highest_interval = max(CHORD_FORMULAS[chord_quality])
    roots = []

    for midi_number in range(low_midi, high_midi - highest_interval + 1):
        root_note = midi_to_note(midi_number)
        notes = chord_notes(root_note, chord_quality)

        if all(note_has_sample(note) for note in notes):
            roots.append(root_note)

    return roots


def trim_or_pad_event(audio, duration):
    """Trim or pad a generated chord event to a fixed duration."""

    target_length = int(config.SAMPLE_RATE * duration)

    if len(audio) > target_length:
        return audio[:target_length]

    return np.concatenate([audio, np.zeros(target_length - len(audio))])


def create_solid_chord_waveform(notes):
    """Create a blocked chord by summing aligned note samples."""

    chord_audio = np.zeros(int(config.SAMPLE_RATE * CHORD_EVENT_DURATION))

    for note in notes:
        note_audio = trim_or_pad_event(load_piano_sample(note), CHORD_EVENT_DURATION)
        chord_audio += note_audio

    max_value = np.max(np.abs(chord_audio))

    if max_value > 0:
        chord_audio = chord_audio / max_value

    return add_fade_out(chord_audio * config.VOLUME)


def create_broken_chord_waveform(notes):
    """Create a broken chord with equal note lengths."""

    note_segments = []

    for note in notes:
        note_audio = trim_or_pad_event(load_piano_sample(note), BROKEN_CHORD_NOTE_DURATION)
        note_segments.append(note_audio)

    return np.concatenate(note_segments)


def create_chord_waveform(level, root_note, chord_quality, target_tone=None):
    """Create the playback waveform for a chord question."""

    notes = chord_notes(root_note, chord_quality)
    pause = create_silence(config.PAUSE_DURATION)

    if level == 1:
        return np.concatenate([
            create_broken_chord_waveform(notes),
            pause,
            create_solid_chord_waveform(notes),
        ])

    if level == 2:
        return create_solid_chord_waveform(notes)

    if level in (3, 4):
        target_note_index = TONE_CHOICES.index(target_tone)
        target_note = notes[target_note_index]
        return np.concatenate([
            create_solid_chord_waveform(notes),
            pause,
            create_broken_chord_waveform(notes),
            pause,
            trim_or_pad_event(load_piano_sample(target_note), CHORD_EVENT_DURATION),
        ])

    return create_solid_chord_waveform(notes)


def create_chord_quality_waveform(root_note, chord_quality):
    """Create the first audio part for Level 3/4 chord-quality identification."""

    return create_solid_chord_waveform(chord_notes(root_note, chord_quality))


def create_chord_tone_waveform(root_note, chord_quality, target_tone):
    """Create the second audio part for Level 3/4 chord-tone identification."""

    notes = chord_notes(root_note, chord_quality)
    target_note_index = TONE_CHOICES.index(target_tone)
    target_note = notes[target_note_index]

    return np.concatenate([
        create_broken_chord_waveform(notes),
        create_silence(config.PAUSE_DURATION),
        trim_or_pad_event(load_piano_sample(target_note), CHORD_EVENT_DURATION),
    ])


def create_chord_audio(level, root_note, chord_quality, target_tone, file_path):
    """Write a chord question WAV file."""

    sf.write(
        file_path,
        create_chord_waveform(level, root_note, chord_quality, target_tone),
        config.SAMPLE_RATE,
    )


def create_chord_part_audio(waveform, file_path):
    """Write one split chord-question WAV file."""

    sf.write(file_path, waveform, config.SAMPLE_RATE)


def chord_audio_cache_label():
    """Return the cache label for generated chord audio."""

    return "chords_v1_strong_attack"


def split_chord_audio_cache_label():
    """Return the cache label for split Level 3/4 chord audio."""

    return "chords_v2_split_parts"


def create_chord_question(level):
    """Create one chord question and generated audio file."""

    if level < 1 or level > 8:
        raise ValueError(f"Chords are currently implemented for levels 1-8, not {level}.")

    config.ensure_audio_folder(config.CHORD_AUDIO_FOLDER)
    qualities = chord_quality_choices(level)
    chord_quality = random.choice(qualities)
    roots = possible_chord_roots(chord_quality)

    if not roots:
        raise FileNotFoundError(f"No usable piano samples found for {chord_quality}.")

    root_note = random.choice(roots)
    question_type = chord_question_type(level)
    target_tone = None

    if question_type == "quality_and_tone":
        target_tone = random.choice(TONE_CHOICES)
        quality_answer = chord_display_name(chord_quality, level)
        quality_choices = tuple(chord_display_name(quality, level) for quality in qualities)
        tone_answer = target_tone
        tone_choices = TONE_CHOICES
        answer = f"{quality_answer} / {tone_answer}"
        choices = []
        prompt = "Identify the chord quality, then identify the single note played after the broken triad."
    else:
        quality_answer = None
        quality_choices = ()
        tone_answer = None
        tone_choices = ()
        answer = chord_display_name(chord_quality, level)
        choices = [chord_display_name(quality, level) for quality in qualities]
        prompt = "Identify the chord quality."

    tone_audio_file = ""

    if question_type == "quality_and_tone":
        quality_filename = make_safe_filename(
            f"level_{level}_{root_note}_{chord_quality}_{target_tone}_quality_{split_chord_audio_cache_label()}.wav"
        )
        tone_filename = make_safe_filename(
            f"level_{level}_{root_note}_{chord_quality}_{target_tone}_tone_{split_chord_audio_cache_label()}.wav"
        )
        quality_file_path = config.CHORD_AUDIO_FOLDER / quality_filename
        tone_file_path = config.CHORD_AUDIO_FOLDER / tone_filename

        if not quality_file_path.exists():
            create_chord_part_audio(
                create_chord_quality_waveform(root_note, chord_quality),
                quality_file_path,
            )

        if not tone_file_path.exists():
            create_chord_part_audio(
                create_chord_tone_waveform(root_note, chord_quality, target_tone),
                tone_file_path,
            )

        filename = quality_filename
        tone_audio_file = f"chords/{tone_filename}"
    else:
        filename = make_safe_filename(
            f"level_{level}_{root_note}_{chord_quality}_{target_tone or question_type}_{chord_audio_cache_label()}.wav"
        )
        file_path = config.CHORD_AUDIO_FOLDER / filename

        if not file_path.exists():
            create_chord_audio(level, root_note, chord_quality, target_tone, file_path)

    return ChordQuestion(
        level=level,
        root_note=root_note,
        chord_quality=chord_quality,
        answer=answer,
        choices=choices,
        audio_file=f"chords/{filename}",
        prompt=prompt,
        question_type=question_type,
        tone_audio_file=tone_audio_file,
        target_tone=target_tone,
        quality_answer=quality_answer,
        quality_choices=quality_choices,
        tone_answer=tone_answer,
        tone_choices=tone_choices,
    )
