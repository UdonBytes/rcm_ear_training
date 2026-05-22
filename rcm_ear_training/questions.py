"""Question generation and generated-audio cache naming."""

from dataclasses import dataclass
import random

from rcm_ear_training.audio import create_interval_audio
from rcm_ear_training.config import AUDIO_FOLDER, ensure_audio_folder
from rcm_ear_training.samples import get_possible_starting_notes
from rcm_ear_training.theory import INTERVALS, LEVEL_INTERVALS, midi_to_note, note_to_midi


@dataclass(frozen=True)
class Question:
    level: int
    start_note: str
    end_note: str
    answer: str
    choices: list[str]
    audio_file: str
    direction: str


def repeat_limit_for_level(level):
    """Return the maximum allowed streak for one interval answer."""

    if level <= 2:
        return 3

    return 2


def current_answer_streak(answer_history):
    """Return the answer and length of the active streak."""

    if not answer_history:
        return None, 0

    current_answer = answer_history[-1]
    streak_length = 0

    for answer in reversed(answer_history):
        if answer != current_answer:
            break

        streak_length += 1

    return current_answer, streak_length


def choose_interval(level, answer_history=None):
    """Choose an interval while avoiding overly long answer streaks."""

    interval_choices = LEVEL_INTERVALS[level]
    history = answer_history or []
    repeated_answer, streak_length = current_answer_streak(history)
    max_repeats = repeat_limit_for_level(level)

    if repeated_answer is not None and streak_length >= max_repeats:
        varied_choices = [
            interval for interval in interval_choices if interval != repeated_answer
        ]

        if varied_choices:
            return random.choice(varied_choices)

    return random.choice(interval_choices)


def make_safe_filename(text):
    """Make text safe to use as a file name."""

    return (
        text
        .replace("#", "sharp")
        .replace("/", "or")
        .replace(" ", "_")
        .lower()
    )


def audio_cache_label(level):
    """Return the synthesis cache label for a level."""

    if level <= 4:
        return "strong_attack_steady_articulation_connected"

    return "strong_attack_aligned_melodic_harmonic"


def create_question(level, answer_history=None):
    """Create one random interval question for the chosen level."""

    if level not in LEVEL_INTERVALS:
        raise ValueError(f"Level {level} has not been created yet.")

    ensure_audio_folder()
    interval_choices = LEVEL_INTERVALS[level]
    correct_answer = choose_interval(level, answer_history)
    possible_starting_notes = get_possible_starting_notes(correct_answer)

    if len(possible_starting_notes) == 0:
        raise FileNotFoundError(f"No usable piano samples found for {correct_answer}.")

    start_note = random.choice(possible_starting_notes)
    start_midi = note_to_midi(start_note)
    interval_distance = INTERVALS[correct_answer]
    end_note = midi_to_note(start_midi + interval_distance)

    if level <= 4:
        direction = "ascending_descending"
    else:
        direction = random.choice(["ascending", "descending"])

    filename = make_safe_filename(
        f"level_{level}_{start_note}_{end_note}_{correct_answer}_{direction}_{audio_cache_label(level)}.wav"
    )
    file_path = AUDIO_FOLDER / filename

    if not file_path.exists():
        create_interval_audio(level, start_note, end_note, direction, file_path)

    choices = interval_choices.copy()
    random.shuffle(choices)

    return Question(
        level=level,
        start_note=start_note,
        end_note=end_note,
        answer=correct_answer,
        choices=choices,
        audio_file=filename,
        direction=direction,
    )
