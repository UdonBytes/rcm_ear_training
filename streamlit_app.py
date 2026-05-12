"""
RCM Ear Training Practice App
Streamlit version

This app:
1. Uses real piano AIFF note samples
2. Lets students choose their RCM level
3. Creates interval audio files from piano samples
4. Plays interval patterns based on level
5. Checks the student's answer
"""

from pathlib import Path
import random

import numpy as np
import soundfile as sf
import streamlit as st


st.set_page_config(
    page_title="RCM Ear Training",
    page_icon="🎧",
    layout="centered"
)


SAMPLE_RATE = 44100
NOTE_DURATION = 1.2
PAUSE_DURATION = 0.25
VOLUME = 0.35

AUDIO_FOLDER = Path("static/audio")
PIANO_SAMPLE_FOLDER = Path("static/piano_samples")

AUDIO_FOLDER.mkdir(parents=True, exist_ok=True)


NOTE_NAMES = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B"
]

SHARP_TO_FLAT = {
    "C#": "Db",
    "D#": "Eb",
    "F#": "Gb",
    "G#": "Ab",
    "A#": "Bb"
}

INTERVALS = {
    "Minor 2nd": 1,
    "Major 2nd": 2,
    "Minor 3rd": 3,
    "Major 3rd": 4,
    "Perfect 4th": 5,
    "Augmented 4th / Diminished 5th": 6,
    "Perfect 5th": 7,
    "Minor 6th": 8,
    "Major 6th": 9,
    "Minor 7th": 10,
    "Major 7th": 11,
    "Perfect 8ve": 12
}

LEVEL_INTERVALS = {
    1: [
        "Major 3rd",
        "Minor 3rd"
    ],
    2: [
        "Major 3rd",
        "Minor 3rd",
        "Perfect 5th"
    ],
    3: [
        "Major 3rd",
        "Minor 3rd",
        "Perfect 4th",
        "Perfect 5th"
    ],
    4: [
        "Major 3rd",
        "Minor 3rd",
        "Perfect 4th",
        "Perfect 5th",
        "Perfect 8ve"
    ],
    5: [
        "Major 3rd",
        "Minor 3rd",
        "Perfect 4th",
        "Perfect 5th",
        "Major 6th",
        "Minor 6th",
        "Perfect 8ve"
    ],
    6: [
        "Major 2nd",
        "Minor 2nd",
        "Major 3rd",
        "Minor 3rd",
        "Perfect 4th",
        "Perfect 5th",
        "Major 6th",
        "Minor 6th",
        "Perfect 8ve"
    ],
    7: [
        "Major 2nd",
        "Minor 2nd",
        "Major 3rd",
        "Minor 3rd",
        "Perfect 4th",
        "Perfect 5th",
        "Major 6th",
        "Minor 6th",
        "Major 7th",
        "Minor 7th",
        "Perfect 8ve"
    ],
    8: [
        "Major 2nd",
        "Minor 2nd",
        "Major 3rd",
        "Minor 3rd",
        "Perfect 4th",
        "Augmented 4th / Diminished 5th",
        "Perfect 5th",
        "Major 6th",
        "Minor 6th",
        "Major 7th",
        "Minor 7th",
        "Perfect 8ve"
    ]
}

SAMPLE_LOW_NOTE = "A3"
SAMPLE_HIGH_NOTE = "E5"


def note_to_midi(note):
    """
    Converts a note name such as C4 or F#4 into a MIDI number.
    C4 is MIDI note 60.
    """

    if len(note) == 2:
        note_name = note[0]
        octave = int(note[1])
    else:
        note_name = note[:2]
        octave = int(note[2])

    note_index = NOTE_NAMES.index(note_name)
    midi_number = 12 * (octave + 1) + note_index

    return midi_number


def midi_to_note(midi_number):
    """
    Converts a MIDI number into a note name.
    Example: 60 becomes C4.
    """

    note_name = NOTE_NAMES[midi_number % 12]
    octave = midi_number // 12 - 1

    return f"{note_name}{octave}"


def make_sample_note_name(note):
    """
    Converts sharp note names into flat note names when needed.
    Example: C#4 becomes Db4.
    """

    if len(note) == 2:
        sample_note = note
    else:
        note_name = note[:2]
        octave = note[2]

        if note_name in SHARP_TO_FLAT:
            sample_note = f"{SHARP_TO_FLAT[note_name]}{octave}"
        else:
            sample_note = note

    return sample_note


def find_sample_path(note):
    """
    Finds the matching piano sample file.
    Example app note: C#4
    Example sample file: Piano.mf.Db4.aiff
    """

    sample_note = make_sample_note_name(note)

    possible_paths = [
        PIANO_SAMPLE_FOLDER / f"Piano.mf.{sample_note}.aiff",
        PIANO_SAMPLE_FOLDER / f"Piano.mf.{sample_note}.aif",
        PIANO_SAMPLE_FOLDER / f"piano.mf.{sample_note}.aiff",
        PIANO_SAMPLE_FOLDER / f"piano.mf.{sample_note}.aif"
    ]

    found_path = None

    for sample_path in possible_paths:
        if sample_path.exists():
            found_path = sample_path

    if found_path is None:
        available_files = sorted(
            file.name for file in PIANO_SAMPLE_FOLDER.glob("*")
        )

        raise FileNotFoundError(
            f"Missing piano sample for {note}. "
            f"Available files: {available_files}"
        )

    return found_path


def note_has_sample(note):
    """
    Checks whether a sample exists for this note.
    """

    has_sample = True

    try:
        find_sample_path(note)
    except FileNotFoundError:
        has_sample = False

    return has_sample


def get_possible_starting_notes(interval_name):
    """
    Finds starting notes that fit inside the available sample range.
    """

    interval_distance = INTERVALS[interval_name]

    low_midi = note_to_midi(SAMPLE_LOW_NOTE)
    high_midi = note_to_midi(SAMPLE_HIGH_NOTE)

    possible_notes = []

    for midi_number in range(low_midi, high_midi + 1):
        start_note = midi_to_note(midi_number)
        end_midi = midi_number + interval_distance
        end_note = midi_to_note(end_midi)

        if end_midi <= high_midi:
            if note_has_sample(start_note) and note_has_sample(end_note):
                possible_notes.append(start_note)

    return possible_notes


def resample_audio(audio, original_rate, target_rate):
    """
    Resamples audio if the piano sample rate does not match the app sample rate.
    """

    if original_rate == target_rate:
        resampled_audio = audio
    else:
        original_positions = np.linspace(0, 1, len(audio))
        target_length = int(len(audio) * target_rate / original_rate)
        target_positions = np.linspace(0, 1, target_length)
        resampled_audio = np.interp(
            target_positions,
            original_positions,
            audio
        )

    return resampled_audio


def trim_or_pad_audio(audio, duration):
    """
    Trims or pads audio to a fixed duration.
    """

    target_length = int(SAMPLE_RATE * duration)

    if len(audio) > target_length:
        adjusted_audio = audio[:target_length]
    else:
        padding = np.zeros(target_length - len(audio))
        adjusted_audio = np.concatenate([audio, padding])

    return adjusted_audio


def add_fade_out(audio):
    """
    Adds a short fade out so the note does not click at the end.
    """

    fade_length = int(SAMPLE_RATE * 0.04)

    if len(audio) > fade_length:
        fade_out = np.linspace(1, 0, fade_length)
        audio[-fade_length:] *= fade_out

    return audio


def load_piano_sample(note):
    """
    Loads one real piano sample from static/piano_samples.
    """

    sample_path = find_sample_path(note)
    audio, sample_rate = sf.read(sample_path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = resample_audio(audio, sample_rate, SAMPLE_RATE)
    audio = trim_or_pad_audio(audio, NOTE_DURATION)

    max_value = np.max(np.abs(audio))

    if max_value > 0:
        audio = audio / max_value

    audio = audio * VOLUME
    audio = add_fade_out(audio)

    return audio


def create_silence(duration):
    """
    Creates silence between notes.
    """

    return np.zeros(int(SAMPLE_RATE * duration))


def create_harmonic_audio(first_note, second_note):
    """
    Creates one harmonic interval by playing both notes together.
    """

    first_audio = load_piano_sample(first_note)
    second_audio = load_piano_sample(second_note)

    harmonic_audio = first_audio + second_audio

    max_value = np.max(np.abs(harmonic_audio))

    if max_value > 0:
        harmonic_audio = harmonic_audio / max_value

    harmonic_audio = harmonic_audio * VOLUME
    harmonic_audio = add_fade_out(harmonic_audio)

    return harmonic_audio


def create_interval_audio(level, start_note, end_note, direction, file_path):
    """
    Creates the interval audio file.

    Levels 1 to 4:
    ascending then descending, example C E C.

    Levels 5 to 8:
    ascending or descending melodically first,
    then harmonically.
    """

    pause = create_silence(PAUSE_DURATION)

    if level <= 4:
        first_note = load_piano_sample(start_note)
        second_note = load_piano_sample(end_note)
        third_note = load_piano_sample(start_note)

        audio = np.concatenate([
            first_note,
            pause,
            second_note,
            pause,
            third_note
        ])

    else:
        if direction == "ascending":
            melodic_first = load_piano_sample(start_note)
            melodic_second = load_piano_sample(end_note)
        else:
            melodic_first = load_piano_sample(end_note)
            melodic_second = load_piano_sample(start_note)

        harmonic_interval = create_harmonic_audio(start_note, end_note)

        audio = np.concatenate([
            melodic_first,
            pause,
            melodic_second,
            pause,
            harmonic_interval
        ])

    sf.write(file_path, audio, SAMPLE_RATE)


def make_safe_filename(text):
    """
    Makes text safe to use as a file name.
    """

    return (
        text
        .replace("#", "sharp")
        .replace("/", "or")
        .replace(" ", "_")
        .lower()
    )


def create_question(level):
    """
    Creates one random interval question for the chosen level.
    """

    interval_choices = LEVEL_INTERVALS[level]
    correct_answer = random.choice(interval_choices)

    possible_starting_notes = get_possible_starting_notes(correct_answer)

    if len(possible_starting_notes) == 0:
        raise FileNotFoundError(
            f"No usable piano samples found for {correct_answer}."
        )

    start_note = random.choice(possible_starting_notes)

    start_midi = note_to_midi(start_note)
    interval_distance = INTERVALS[correct_answer]
    end_midi = start_midi + interval_distance
    end_note = midi_to_note(end_midi)

    if level <= 4:
        direction = "ascending_descending"
    else:
        direction = random.choice([
            "ascending",
            "descending"
        ])

    filename = make_safe_filename(
        f"level_{level}_{start_note}_{end_note}_{correct_answer}_{direction}.wav"
    )

    file_path = AUDIO_FOLDER / filename

    if not file_path.exists():
        create_interval_audio(
            level,
            start_note,
            end_note,
            direction,
            file_path
        )

    choices = interval_choices.copy()
    random.shuffle(choices)

    question = {
        "level": level,
        "start_note": start_note,
        "end_note": end_note,
        "answer": correct_answer,
        "choices": choices,
        "audio_file": filename,
        "direction": direction
    }

    return question


def get_level_description(level):
    """
    Creates a readable interval list for the home page.
    """

    intervals = ", ".join(LEVEL_INTERVALS[level])

    return intervals


def reset_question():
    """
    Clears the current question and result.
    """

    st.session_state.current_question = None
    st.session_state.result_checked = False
    st.session_state.is_correct = False
    st.session_state.selected_answer = None


if "current_question" not in st.session_state:
    st.session_state.current_question = None

if "result_checked" not in st.session_state:
    st.session_state.result_checked = False

if "is_correct" not in st.session_state:
    st.session_state.is_correct = False

if "selected_answer" not in st.session_state:
    st.session_state.selected_answer = None


st.title("RCM Ear Training")
st.caption("Choose your level to practise interval identification.")

levels = sorted(LEVEL_INTERVALS.keys())

selected_level = st.selectbox(
    "Choose level",
    levels,
    format_func=lambda level: f"Level {level}",
    on_change=reset_question
)

with st.expander("Show intervals by level"):
    for level in levels:
        st.write(f"Level {level}: {get_level_description(level)}")

if st.session_state.current_question is None:
    try:
        st.session_state.current_question = create_question(selected_level)
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

question = st.session_state.current_question

st.subheader(f"Level {question['level']} Intervals")

if question["level"] <= 4:
    st.write("Listen to the interval played up and back down, then choose the interval.")
else:
    st.write("Listen to the interval melodically, then harmonically, then choose the interval.")

audio_path = AUDIO_FOLDER / question["audio_file"]

if audio_path.exists():
    with open(audio_path, "rb") as audio_file:
        st.audio(audio_file.read(), format="audio/wav")
else:
    st.error("The audio file was not created.")
    st.stop()

st.write("Choose your answer:")

for choice in question["choices"]:
    if st.button(choice, use_container_width=True):
        st.session_state.selected_answer = choice
        st.session_state.is_correct = choice == question["answer"]
        st.session_state.result_checked = True

if st.session_state.result_checked:
    if st.session_state.is_correct:
        st.success("Correct!")
    else:
        st.error(f"Not quite. The answer was {question['answer']}.")

    if st.button("Next Question", use_container_width=True):
        reset_question()
        st.rerun()

if st.button("New Question", use_container_width=True):
    reset_question()
    st.rerun()