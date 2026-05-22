import unittest

from rcm_ear_training.audio import create_interval_waveform
from rcm_ear_training.config import (
    ATTACK_THRESHOLD_RATIO,
    CONNECTED_NOTE_SPACING,
    NOTE_DURATION,
    PAUSE_DURATION,
    SAMPLE_RATE,
)
from rcm_ear_training.display import group_answer_choices
from rcm_ear_training.questions import (
    audio_cache_label,
    choose_interval,
    current_answer_streak,
    make_safe_filename,
    repeat_limit_for_level,
)
from rcm_ear_training.samples import get_possible_starting_notes, load_piano_sample
from rcm_ear_training.theory import (
    SAMPLE_HIGH_NOTE,
    SAMPLE_LOW_NOTE,
    midi_to_note,
    note_to_midi,
)


class TheoryTests(unittest.TestCase):
    def test_note_to_midi_and_back(self):
        self.assertEqual(note_to_midi("C4"), 60)
        self.assertEqual(note_to_midi("F#4"), 66)
        self.assertEqual(midi_to_note(73), "C#5")


class SampleTests(unittest.TestCase):
    def test_possible_starting_notes_stay_within_sample_range(self):
        notes = get_possible_starting_notes("Perfect 8ve")

        self.assertIn("F3", notes)
        self.assertIn("A4", notes)
        self.assertNotIn("A#4", notes)

    def test_loaded_samples_are_aligned_to_audible_attack(self):
        audio = load_piano_sample("C4")
        threshold = max(abs(audio)) * ATTACK_THRESHOLD_RATIO
        first_audible_sample = next(
            index for index, value in enumerate(audio) if abs(value) >= threshold
        )

        self.assertLessEqual(first_audible_sample, int(SAMPLE_RATE * 0.01))

    def test_all_configured_samples_have_equal_length_and_attack(self):
        expected_length = int(SAMPLE_RATE * NOTE_DURATION)
        expected_attack = int(SAMPLE_RATE * 0.005)

        for midi_number in range(
            note_to_midi(SAMPLE_LOW_NOTE),
            note_to_midi(SAMPLE_HIGH_NOTE) + 1,
        ):
            note = midi_to_note(midi_number)

            with self.subTest(note=note):
                audio = load_piano_sample(note)
                threshold = max(abs(audio)) * ATTACK_THRESHOLD_RATIO
                first_audible_sample = next(
                    index
                    for index, value in enumerate(audio)
                    if abs(value) >= threshold
                )

                self.assertEqual(len(audio), expected_length)
                self.assertEqual(first_audible_sample, expected_attack)

    def test_major_sixth_pair_has_aligned_strong_attacks(self):
        lower_note = load_piano_sample("F4")
        upper_note = load_piano_sample("D5")

        onsets = []

        for audio in [lower_note, upper_note]:
            threshold = max(abs(audio)) * ATTACK_THRESHOLD_RATIO
            onsets.append(
                next(
                    index
                    for index, value in enumerate(audio)
                    if abs(value) >= threshold
                )
            )

        self.assertEqual(onsets[0], onsets[1])


class AudioTests(unittest.TestCase):
    def test_level_1_audio_uses_consistent_connected_spacing(self):
        waveform = create_interval_waveform(
            level=1,
            start_note="C4",
            end_note="E4",
            direction="ascending_descending",
        )

        note_samples = int(SAMPLE_RATE * NOTE_DURATION)
        transition_samples = int(SAMPLE_RATE * CONNECTED_NOTE_SPACING)
        expected_samples = (note_samples * 3) + (transition_samples * 2)
        self.assertEqual(len(waveform), expected_samples)

        starts = [
            0,
            note_samples + transition_samples,
            (note_samples + transition_samples) * 2,
        ]

        for start in starts:
            segment = waveform[start:start + note_samples]
            threshold = max(abs(segment)) * ATTACK_THRESHOLD_RATIO
            first_audible_sample = next(
                index
                for index, value in enumerate(segment)
                if abs(value) >= threshold
            )

            self.assertEqual(first_audible_sample, int(SAMPLE_RATE * 0.005))

    def test_level_5_audio_keeps_melodic_and_harmonic_pauses(self):
        waveform = create_interval_waveform(
            level=5,
            start_note="C4",
            end_note="E4",
            direction="ascending",
        )

        note_samples = int(SAMPLE_RATE * NOTE_DURATION)
        pause_samples = int(SAMPLE_RATE * PAUSE_DURATION)
        expected_samples = (note_samples * 3) + (pause_samples * 2)
        self.assertEqual(len(waveform), expected_samples)


class QuestionCacheTests(unittest.TestCase):
    def test_audio_cache_labels_distinguish_playback_patterns(self):
        self.assertEqual(
            audio_cache_label(1),
            "strong_attack_steady_articulation_connected",
        )
        self.assertEqual(
            audio_cache_label(4),
            "strong_attack_steady_articulation_connected",
        )
        self.assertEqual(
            audio_cache_label(5),
            "strong_attack_aligned_melodic_harmonic",
        )

    def test_safe_filename_normalizes_interval_names(self):
        filename = make_safe_filename(
            "level_8_C#4_C#5_Augmented 4th / Diminished 5th_ascending.wav"
        )

        self.assertEqual(
            filename,
            "level_8_csharp4_csharp5_augmented_4th_or_diminished_5th_ascending.wav",
        )


class QuestionVarietyTests(unittest.TestCase):
    def test_repeat_limits_are_looser_for_early_levels(self):
        self.assertEqual(repeat_limit_for_level(1), 3)
        self.assertEqual(repeat_limit_for_level(2), 3)
        self.assertEqual(repeat_limit_for_level(3), 2)
        self.assertEqual(repeat_limit_for_level(8), 2)

    def test_current_answer_streak_counts_trailing_answers(self):
        answer, streak = current_answer_streak(
            ["Major 3rd", "Minor 3rd", "Minor 3rd"]
        )

        self.assertEqual(answer, "Minor 3rd")
        self.assertEqual(streak, 2)

    def test_level_1_allows_three_repeats_before_forcing_variety(self):
        answer = choose_interval(
            1,
            answer_history=["Minor 3rd", "Minor 3rd", "Minor 3rd"],
        )

        self.assertEqual(answer, "Major 3rd")

    def test_level_3_forces_variety_after_two_repeats(self):
        answers = {
            choose_interval(
                3,
                answer_history=["Perfect 5th", "Perfect 5th"],
            )
            for _ in range(20)
        }

        self.assertNotIn("Perfect 5th", answers)


class DisplayTests(unittest.TestCase):
    def test_answer_choices_group_minor_and_major_intervals_on_same_row(self):
        rows = group_answer_choices(
            [
                "Perfect 5th",
                "Major 3rd",
                "Perfect 4th",
                "Minor 3rd",
            ]
        )

        self.assertEqual(
            rows,
            [
                ["Minor 3rd", "Major 3rd"],
                ["Perfect 4th"],
                ["Perfect 5th"],
            ],
        )

    def test_answer_choices_group_larger_minor_and_major_intervals(self):
        rows = group_answer_choices(
            [
                "Major 7th",
                "Minor 6th",
                "Major 6th",
                "Minor 7th",
                "Perfect 8ve",
            ]
        )

        self.assertEqual(
            rows,
            [
                ["Minor 6th", "Major 6th"],
                ["Minor 7th", "Major 7th"],
                ["Perfect 8ve"],
            ],
        )


if __name__ == "__main__":
    unittest.main()
