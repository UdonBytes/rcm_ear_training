"""
RCM Ear Training Practice App
Sample based Flask version with selectable levels

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
from flask import Flask, request, render_template_string, url_for


app = Flask(__name__)


# Audio settings

SAMPLE_RATE = 44100
NOTE_DURATION = 1.2
PAUSE_DURATION = 0.25
VOLUME = 0.35

AUDIO_FOLDER = Path("static/audio")
PIANO_SAMPLE_FOLDER = Path("static/piano_samples")

AUDIO_FOLDER.mkdir(parents=True, exist_ok=True)


# Music data

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


# Music helper functions

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
    ascending then descending, example C E C

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


# HTML templates

HOME_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>RCM Ear Training</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 920px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.6;
            background: #fafafa;
        }

        h1 {
            font-size: 38px;
            margin-bottom: 8px;
        }

        .card {
            border: 1px solid #ddd;
            border-radius: 18px;
            padding: 30px;
            background: white;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
        }

        .level-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 14px;
            margin-top: 24px;
        }

        .level-button {
            display: block;
            padding: 18px;
            border-radius: 14px;
            background: #222;
            color: white;
            text-decoration: none;
            text-align: center;
            font-size: 18px;
        }

        .level-button:hover {
            background: #444;
        }

        .muted {
            color: #666;
        }

        .level-info {
            margin-top: 26px;
            padding: 16px;
            border-radius: 14px;
            background: #f5f5f5;
        }

        .level-info p {
            margin: 8px 0;
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>RCM Ear Training</h1>
        <p class="muted">Choose your level to practise interval identification.</p>

        <div class="level-grid">
            {% for level in levels %}
                <a class="level-button" href="/practice/{{ level }}">
                    Level {{ level }}
                </a>
            {% endfor %}
        </div>

        <div class="level-info">
            {% for level in levels %}
                <p><strong>Level {{ level }}:</strong> {{ descriptions[level] }}</p>
            {% endfor %}
        </div>
    </div>
</body>
</html>
"""


PRACTICE_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Practice</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 920px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.6;
            background: #fafafa;
        }

        .card {
            border: 1px solid #ddd;
            border-radius: 18px;
            padding: 30px;
            background: white;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
        }

        h1 {
            margin-bottom: 8px;
        }

        audio {
            width: 100%;
            margin: 22px 0;
        }

        button {
            display: block;
            width: 100%;
            margin: 10px 0;
            padding: 15px;
            border-radius: 12px;
            border: 1px solid #bbb;
            background: white;
            font-size: 16px;
            cursor: pointer;
        }

        button:hover {
            background: #f3f3f3;
        }

        .small {
            color: #777;
            font-size: 14px;
            margin-top: 22px;
        }

        .home-link {
            display: inline-block;
            margin-top: 18px;
            color: #333;
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>Level {{ question.level }} Intervals</h1>

        {% if question.level <= 4 %}
            <p>Listen to the interval played up and back down, then choose the interval.</p>
        {% else %}
            <p>Listen to the interval melodically, then harmonically, then choose the interval.</p>
        {% endif %}

        <audio controls>
            <source src="{{ audio_url }}" type="audio/wav">
        </audio>

        <form action="/check" method="post">
            <input type="hidden" name="level" value="{{ question.level }}">
            <input type="hidden" name="correct_answer" value="{{ question.answer }}">

            {% for choice in question.choices %}
                <button type="submit" name="selected_answer" value="{{ choice }}">
                    {{ choice }}
                </button>
            {% endfor %}
        </form>

        <a class="home-link" href="/">Back to levels</a>
    </div>
</body>
</html>
"""


RESULT_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Result</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 920px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.6;
            background: #fafafa;
        }

        .card {
            border: 1px solid #ddd;
            border-radius: 18px;
            padding: 30px;
            background: white;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
        }

        .button {
            display: inline-block;
            margin-top: 20px;
            margin-right: 10px;
            padding: 13px 20px;
            border-radius: 12px;
            background: #222;
            color: white;
            text-decoration: none;
            font-size: 16px;
        }

        .secondary {
            background: #666;
        }

        .correct {
            color: green;
        }

        .wrong {
            color: #b00020;
        }
    </style>
</head>
<body>
    <div class="card">
        {% if is_correct %}
            <h1 class="correct">Correct!</h1>
        {% else %}
            <h1 class="wrong">Not quite</h1>
            <p>The answer was {{ correct_answer }}.</p>
        {% endif %}

        <a class="button" href="/practice/{{ level }}">Next Question</a>
        <a class="button secondary" href="/">Choose Level</a>
    </div>
</body>
</html>
"""


ERROR_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Missing Sample</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 920px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.6;
            background: #fafafa;
        }

        .card {
            border: 1px solid #ddd;
            border-radius: 18px;
            padding: 30px;
            background: white;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
        }

        code {
            background: #f3f3f3;
            padding: 3px 6px;
            border-radius: 6px;
        }

        a {
            display: inline-block;
            margin-top: 20px;
            padding: 13px 20px;
            border-radius: 12px;
            background: #222;
            color: white;
            text-decoration: none;
            font-size: 16px;
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>Missing piano sample</h1>
        <p>{{ error_message }}</p>
        <p>Check that the needed AIFF file exists inside:</p>
        <p><code>static/piano_samples</code></p>
        <a href="/">Back Home</a>
    </div>
</body>
</html>
"""


# Website routes

@app.route("/")
def home():
    levels = sorted(LEVEL_INTERVALS.keys())

    descriptions = {}

    for level in levels:
        descriptions[level] = get_level_description(level)

    return render_template_string(
        HOME_PAGE,
        levels=levels,
        descriptions=descriptions
    )


@app.route("/practice/<int:level>")
def practice(level):
    if level not in LEVEL_INTERVALS:
        page = render_template_string(
            ERROR_PAGE,
            error_message=f"Level {level} has not been created yet."
        )
    else:
        try:
            question = create_question(level)

            audio_url = url_for(
                "static",
                filename=f"audio/{question['audio_file']}"
            )

            page = render_template_string(
                PRACTICE_PAGE,
                question=question,
                audio_url=audio_url
            )

        except FileNotFoundError as error:
            page = render_template_string(
                ERROR_PAGE,
                error_message=str(error)
            )

    return page


@app.route("/check", methods=["POST"])
def check():
    selected_answer = request.form.get("selected_answer")
    correct_answer = request.form.get("correct_answer")
    level = request.form.get("level")

    is_correct = selected_answer == correct_answer

    return render_template_string(
        RESULT_PAGE,
        is_correct=is_correct,
        correct_answer=correct_answer,
        level=level
    )


# Run app

if __name__ == "__main__":
    app.run(debug=True)