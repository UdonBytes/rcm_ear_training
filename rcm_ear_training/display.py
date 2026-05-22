"""Display helpers for arranging interval choices."""

from rcm_ear_training.theory import INTERVALS


INTERVAL_GROUPS = {
    "Minor 2nd": "2nd",
    "Major 2nd": "2nd",
    "Minor 3rd": "3rd",
    "Major 3rd": "3rd",
    "Perfect 4th": "4th",
    "Augmented 4th / Diminished 5th": "tritone",
    "Perfect 5th": "5th",
    "Minor 6th": "6th",
    "Major 6th": "6th",
    "Minor 7th": "7th",
    "Major 7th": "7th",
    "Perfect 8ve": "8ve",
}


def group_answer_choices(choices):
    """Group answer choices into musically related display rows."""

    grouped_choices = []
    current_group = None
    current_row = []

    for choice in sorted(choices, key=lambda interval: INTERVALS[interval]):
        choice_group = INTERVAL_GROUPS[choice]

        if current_row and choice_group != current_group:
            grouped_choices.append(current_row)
            current_row = []

        current_group = choice_group
        current_row.append(choice)

    if current_row:
        grouped_choices.append(current_row)

    return grouped_choices
