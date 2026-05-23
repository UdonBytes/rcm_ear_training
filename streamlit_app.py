"""Streamlit entry point for the RCM ear-training app."""

import streamlit as st

from rcm_ear_training.config import AUDIO_FOLDER
from rcm_ear_training.chords import create_chord_question
from rcm_ear_training.clapback import (
    choose_unplayed_example_index,
    create_clapback_question,
    get_playable_clapback_examples,
)
from rcm_ear_training.curriculum import (
    IMPLEMENTED,
    CHORDS,
    CLAPBACK,
    INTERVALS,
    PLAYBACK,
    current_levels,
    future_levels,
    get_level,
)
from rcm_ear_training.display import group_answer_choices
from rcm_ear_training.questions import create_question
from rcm_ear_training.theory import get_level_description


st.set_page_config(
    page_title="RCM Ear Tests",
    layout="centered",
)

st.markdown(
    """
    <style>
        div[data-testid="stButton"] button[kind="primary"] {
            background-color: #5f7288 !important;
            border-color: #5f7288 !important;
            color: white !important;
        }

        div[data-testid="stButton"] button[kind="primary"]:hover {
            background-color: #52677e !important;
            border-color: #52677e !important;
            color: white !important;
        }

        div[data-testid="stButton"] button[kind="primary"]:focus {
            box-shadow: 0 0 0 0.2rem rgba(95, 114, 136, 0.35) !important;
            color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def reset_question():
    """Clear the current question and result."""

    st.session_state.current_question = None
    st.session_state.result_checked = False
    st.session_state.is_correct = False
    st.session_state.selected_answer = None
    st.session_state.selected_chord_quality = None
    st.session_state.selected_chord_tone = None


def reset_test():
    """Clear the selected test and current question."""

    st.session_state.selected_test_id = None
    st.session_state.clapback_example_index = 0
    st.session_state.clapback_played_ids = []
    st.session_state.clapback_previous_key = None
    reset_question()


def initialize_session_state():
    """Create Streamlit session keys used by the practice flow."""

    defaults = {
        "current_question": None,
        "answer_history": [],
        "selected_level_id": None,
        "selected_test_id": None,
        "result_checked": False,
        "is_correct": False,
        "selected_answer": None,
        "selected_chord_quality": None,
        "selected_chord_tone": None,
        "clapback_example_index": 0,
        "clapback_played_ids": [],
        "clapback_previous_key": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_level_home():
    """Render the first page of selectable levels."""

    st.subheader("Choose Level")

    level_rows = [
        current_levels()[index:index + 4]
        for index in range(0, len(current_levels()), 4)
    ]

    for row in level_rows:
        columns = st.columns(len(row))

        for column, level in zip(columns, row):
            with column:
                if st.button(level.label, use_container_width=True):
                    st.session_state.selected_level_id = level.id
                    reset_test()
                    st.rerun()

    with st.expander("Planned expansion"):
        for level in future_levels():
            test_labels = ", ".join(test.label for test in level.tests)
            st.write(f"{level.label}: {test_labels}")


def render_test_menu(level):
    """Render the test menu for a selected level."""

    if st.button("Back to Levels"):
        st.session_state.selected_level_id = None
        reset_test()
        st.rerun()

    st.subheader(level.label)

    if level.interval_level is not None and level.interval_level <= 8:
        st.caption(f"Intervals: {get_level_description(level.interval_level)}")

    st.write("Choose a test:")

    for test in level.tests:
        if level.interval_level == 5 and test.id == PLAYBACK:
            continue

        disabled = test.status != IMPLEMENTED
        if level.interval_level == 5 and test.id == CLAPBACK:
            button_label = "Clapback / Playback"
        else:
            button_label = test.label if not disabled else f"{test.label} (planned)"

        if st.button(
            button_label,
            disabled=disabled,
            use_container_width=True,
        ):
            st.session_state.selected_test_id = test.id
            reset_question()
            st.rerun()


def get_current_question(selected_level):
    """Return the current question, creating it on demand."""

    if st.session_state.current_question is None:
        level_history = [
            answer
            for level, answer in st.session_state.answer_history
            if level == selected_level
        ]
        st.session_state.current_question = create_question(
            selected_level,
            answer_history=level_history,
        )
        st.session_state.answer_history.append(
            (selected_level, st.session_state.current_question.answer)
        )
        st.session_state.answer_history = st.session_state.answer_history[-40:]

    return st.session_state.current_question


def select_answer(choice, correct_answer):
    """Record a single answer selection."""

    st.session_state.selected_answer = choice
    st.session_state.is_correct = choice == correct_answer
    st.session_state.result_checked = True


def update_chord_result(question):
    """Update result state when both chord answer parts are selected."""

    if st.session_state.selected_chord_quality and st.session_state.selected_chord_tone:
        st.session_state.selected_answer = (
            f"{st.session_state.selected_chord_quality} / "
            f"{st.session_state.selected_chord_tone}"
        )
        st.session_state.is_correct = (
            st.session_state.selected_chord_quality == question.quality_answer
            and st.session_state.selected_chord_tone == question.tone_answer
        )
        st.session_state.result_checked = True


def select_chord_quality(choice, question):
    """Record the selected chord quality."""

    st.session_state.selected_chord_quality = choice
    update_chord_result(question)


def select_chord_tone(choice, question):
    """Record the selected chord tone."""

    st.session_state.selected_chord_tone = choice
    update_chord_result(question)


def render_question(question):
    """Render audio and answer buttons for a question."""

    st.subheader(f"Level {question.level} Intervals")

    if question.level <= 4:
        st.write("Listen to the interval played up and back down, then choose the interval.")
    else:
        st.write("Listen to the interval melodically, then harmonically, then choose the interval.")

    audio_path = AUDIO_FOLDER / question.audio_file

    if audio_path.exists():
        with open(audio_path, "rb") as audio_file:
            st.audio(audio_file.read(), format="audio/wav")
    else:
        st.error("The audio file was not created.")
        st.stop()

    st.write("Choose your answer:")

    for row in group_answer_choices(question.choices):
        columns = st.columns(len(row))

        for column, choice in zip(columns, row):
            with column:
                st.button(
                    choice,
                    use_container_width=True,
                    type="primary" if st.session_state.selected_answer == choice else "secondary",
                    on_click=select_answer,
                    args=(choice, question.answer),
                )


def render_result(question):
    """Render answer feedback and question reset controls."""

    if not st.session_state.result_checked:
        return

    if st.session_state.is_correct:
        st.success("Correct!")
    else:
        st.error(f"Not quite. The answer was {question.answer}.")

    if st.button("Next Question", use_container_width=True, type="primary"):
        reset_question()
        st.rerun()


def render_interval_practice(level):
    """Render the currently implemented interval practice test."""

    if st.button("Back to Tests"):
        st.session_state.selected_test_id = None
        reset_question()
        st.rerun()

    try:
        question = get_current_question(level.interval_level)
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.stop()

    render_question(question)
    render_result(question)


def render_chord_plan(level):
    """Render the chord practice test."""

    if st.button("Back to Tests"):
        st.session_state.selected_test_id = None
        reset_question()
        st.rerun()

    st.subheader(f"{level.label} Chords")

    if st.session_state.current_question is None:
        st.session_state.current_question = create_chord_question(level.interval_level)

    question = st.session_state.current_question
    audio_path = AUDIO_FOLDER / question.audio_file

    st.write(question.prompt)

    if audio_path.exists():
        with open(audio_path, "rb") as audio_file:
            st.audio(audio_file.read(), format="audio/wav")
    else:
        st.error("The audio file was not created.")
        st.stop()

    st.write("Choose your answer:")

    if question.question_type == "quality_and_tone":
        st.write("Chord quality:")
        quality_columns = st.columns(len(question.quality_choices))

        for column, choice in zip(quality_columns, question.quality_choices):
            with column:
                st.button(
                    choice,
                    use_container_width=True,
                    type=(
                        "primary"
                        if st.session_state.selected_chord_quality == choice
                        else "secondary"
                    ),
                    on_click=select_chord_quality,
                    args=(choice, question),
                )

        st.write("Single note:")
        tone_columns = st.columns(len(question.tone_choices))

        for column, choice in zip(tone_columns, question.tone_choices):
            with column:
                st.button(
                    choice,
                    use_container_width=True,
                    type=(
                        "primary"
                        if st.session_state.selected_chord_tone == choice
                        else "secondary"
                    ),
                    on_click=select_chord_tone,
                    args=(choice, question),
                )

        if st.session_state.selected_chord_quality:
            st.write(f"Quality selected: {st.session_state.selected_chord_quality}")

        if st.session_state.selected_chord_tone:
            st.write(f"Note selected: {st.session_state.selected_chord_tone}")

    else:
        for choice in question.choices:
            st.button(
                choice,
                use_container_width=True,
                type="primary" if st.session_state.selected_answer == choice else "secondary",
                on_click=select_answer,
                args=(choice, question.answer),
            )

    render_result(question)


def render_clapback_practice(level):
    """Render approved clapback examples."""

    if st.button("Back to Tests"):
        st.session_state.selected_test_id = None
        reset_question()
        st.rerun()

    if level.interval_level == 5:
        st.subheader(f"{level.label} Clapback / Playback")
    else:
        st.subheader(f"{level.label} Clapback")

    examples = get_playable_clapback_examples(level.interval_level)

    if not examples:
        st.info("No approved clapback examples have been added for this level yet.")
        return

    if st.session_state.current_question is None:
        played_ids = st.session_state.clapback_played_ids

        if len(set(played_ids)) >= len(examples):
            played_ids = []

        st.session_state.clapback_example_index = choose_unplayed_example_index(
            examples,
            played_ids,
            st.session_state.clapback_previous_key,
        )

    selected_example = examples[st.session_state.clapback_example_index % len(examples)]
    stale_question = st.session_state.current_question is not None and (
        st.session_state.current_question.example_id != selected_example.id
        or selected_example.audio_version not in st.session_state.current_question.audio_file
    )

    if st.session_state.current_question is None or stale_question:
        st.session_state.current_question = create_clapback_question(
            level.interval_level,
            st.session_state.clapback_example_index,
        )

    question = st.session_state.current_question
    audio_path = AUDIO_FOLDER / question.audio_file

    st.write(f"Time signature: {question.time_signature}")
    st.write(f"Key: {question.key}")

    if audio_path.exists():
        if question.playback_audio_file:
            st.write("Clapback")
        with open(audio_path, "rb") as audio_file:
            st.audio(audio_file.read(), format="audio/wav")
    else:
        st.error("The draft audio file was not created.")
        st.stop()

    if question.playback_audio_file:
        playback_audio_path = AUDIO_FOLDER / question.playback_audio_file

        st.write("Playback")

        if playback_audio_path.exists():
            with open(playback_audio_path, "rb") as audio_file:
                st.audio(audio_file.read(), format="audio/wav")
        else:
            st.error("The playback audio file was not created.")
            st.stop()

    if len(examples) > 1 and st.button("Next Example", use_container_width=True, type="primary"):
        st.session_state.clapback_played_ids = (
            st.session_state.clapback_played_ids + [question.example_id]
        )[-len(examples):]
        st.session_state.clapback_previous_key = question.key
        reset_question()
        st.rerun()


def main():
    """Run the Streamlit app."""

    initialize_session_state()

    st.title("RCM Ear Tests")
    st.caption("Choose a level, then choose a musicianship test.")

    if st.session_state.selected_level_id is None:
        render_level_home()
        return

    level = get_level(st.session_state.selected_level_id)

    if st.session_state.selected_test_id is None:
        render_test_menu(level)
        return

    if st.session_state.selected_test_id == INTERVALS:
        render_interval_practice(level)
        return

    if st.session_state.selected_test_id == CHORDS:
        render_chord_plan(level)
        return

    if st.session_state.selected_test_id == CLAPBACK:
        render_clapback_practice(level)
        return

    st.info("This test is planned but not implemented yet.")


main()
