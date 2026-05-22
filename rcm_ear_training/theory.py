"""Music theory data and note conversion helpers."""


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
    "B",
]

SHARP_TO_FLAT = {
    "C#": "Db",
    "D#": "Eb",
    "F#": "Gb",
    "G#": "Ab",
    "A#": "Bb",
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
    "Perfect 8ve": 12,
}

LEVEL_INTERVALS = {
    1: [
        "Major 3rd",
        "Minor 3rd",
    ],
    2: [
        "Major 3rd",
        "Minor 3rd",
        "Perfect 5th",
    ],
    3: [
        "Major 3rd",
        "Minor 3rd",
        "Perfect 4th",
        "Perfect 5th",
    ],
    4: [
        "Major 3rd",
        "Minor 3rd",
        "Perfect 4th",
        "Perfect 5th",
        "Perfect 8ve",
    ],
    5: [
        "Major 3rd",
        "Minor 3rd",
        "Perfect 4th",
        "Perfect 5th",
        "Major 6th",
        "Minor 6th",
        "Perfect 8ve",
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
        "Perfect 8ve",
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
        "Perfect 8ve",
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
        "Perfect 8ve",
    ],
}

SAMPLE_LOW_NOTE = "F3"
SAMPLE_HIGH_NOTE = "A5"


def note_to_midi(note):
    """Convert a note name such as C4 or F#4 into a MIDI number."""

    if len(note) == 2:
        note_name = note[0]
        octave = int(note[1])
    else:
        note_name = note[:2]
        octave = int(note[2])

    note_index = NOTE_NAMES.index(note_name)
    return 12 * (octave + 1) + note_index


def midi_to_note(midi_number):
    """Convert a MIDI number into a note name."""

    note_name = NOTE_NAMES[midi_number % 12]
    octave = midi_number // 12 - 1
    return f"{note_name}{octave}"


def make_sample_note_name(note):
    """Convert sharp note names into flat names when sample files use flats."""

    if len(note) == 2:
        return note

    note_name = note[:2]
    octave = note[2]

    if note_name in SHARP_TO_FLAT:
        return f"{SHARP_TO_FLAT[note_name]}{octave}"

    return note


def get_level_description(level):
    """Return a readable interval list for one level."""

    return ", ".join(LEVEL_INTERVALS[level])
