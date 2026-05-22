"""Streamlit entry point for the RCM ear-training app."""

import streamlit as st

from rcm_ear_training.config import AUDIO_FOLDER
from rcm_ear_training.display import group_answer_choices
from rcm_ear_training.questions import create_question
from rcm_ear_training.theory import LEVEL_INTERVALS, get_level_description


st.set_page_config(
    page_title="RCM Ear Training",
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


def initialize_session_state():
    """Create Streamlit session keys used by the practice flow."""

    defaults = {
        "current_question": None,
        "answer_history": [],
        "result_checked": False,
        "is_correct": False,
        "selected_answer": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_level_picker():
    """Render the level selector and interval reference."""

    levels = sorted(LEVEL_INTERVALS.keys())
    selected_level = st.selectbox(
        "Choose level",
        levels,
        format_func=lambda level: f"Level {level}",
        on_change=reset_question,
    )

    with st.expander("Show intervals by level"):
        for level in levels:
            st.write(f"Level {level}: {get_level_description(level)}")

    return selected_level


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
                if st.button(choice, use_container_width=True):
                    st.session_state.selected_answer = choice
                    st.session_state.is_correct = choice == question.answer
                    st.session_state.result_checked = True


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


def main():
    """Run the Streamlit app."""

    initialize_session_state()

    st.title("RCM Ear Training")
    st.caption("Choose your level to practise interval identification.")

    selected_level = render_level_picker()

    try:
        question = get_current_question(selected_level)
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.stop()

    render_question(question)
    render_result(question)


main()
