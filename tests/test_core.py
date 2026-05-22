import unittest

from rcm_ear_training.audio import create_interval_waveform
from rcm_ear_training.config import (
    ATTACK_THRESHOLD_RATIO,
    CONNECTED_NOTE_SPACING,
    NOTE_DURATION,
    PAUSE_DURATION,
    SAMPLE_RATE,
)
from rcm_ear_training.chords import get_chord_requirement
from rcm_ear_training.chords import (
    AUGMENTED_TRIAD,
    DIMINISHED_7TH,
    DOMINANT_7TH,
    MAJOR_TRIAD,
    MINOR_TRIAD,
    TONE_CHOICES,
    chord_notes,
    chord_quality_choices,
    create_chord_question,
    create_chord_waveform,
    chord_display_name,
    possible_chord_roots,
)
from rcm_ear_training.curriculum import (
    CHORDS,
    CLAPBACK,
    FUTURE,
    IMPLEMENTED,
    INTERVALS,
    PLAYBACK,
    current_levels,
    future_levels,
    get_level,
    implemented_tests,
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


class CurriculumTests(unittest.TestCase):
    def test_current_levels_cover_rcm_1_to_8(self):
        levels = current_levels()

        self.assertEqual([level.label for level in levels], [
            "Level 1",
            "Level 2",
            "Level 3",
            "Level 4",
            "Level 5",
            "Level 6",
            "Level 7",
            "Level 8",
        ])
        self.assertEqual([level.interval_level for level in levels], list(range(1, 9)))

    def test_levels_1_to_8_map_to_musicianship_test_categories(self):
        level = get_level("level_1")

        self.assertEqual(
            [test.id for test in level.tests],
            [INTERVALS, CHORDS, CLAPBACK, PLAYBACK],
        )
        self.assertEqual(implemented_tests(level)[0].id, INTERVALS)
        self.assertEqual(implemented_tests(level)[0].status, IMPLEMENTED)
        self.assertEqual(implemented_tests(level)[1].id, CHORDS)

    def test_future_expansion_levels_are_mapped_but_not_current(self):
        planned_levels = future_levels()

        self.assertEqual(
            [level.id for level in planned_levels],
            ["prep_a", "prep_b", "level_9", "level_10", "arct"],
        )
        self.assertTrue(all(level.status == FUTURE for level in planned_levels))
        self.assertNotIn(INTERVALS, [test.id for test in get_level("prep_a").tests])


class ChordRequirementTests(unittest.TestCase):
    def test_level_1_chords_use_major_minor_root_position_triads(self):
        requirement = get_chord_requirement("level_1")

        self.assertEqual(requirement.chord_qualities, ("Major triad", "Minor triad"))
        self.assertEqual(requirement.positions, ("Root position",))
        self.assertEqual(
            requirement.playback,
            "Broken triad, then solid/blocked triad.",
        )

    def test_level_3_adds_chord_tone_identification(self):
        requirement = get_chord_requirement("level_3")

        self.assertIn("Chord-tone identification", requirement.question_types)
        self.assertIn("root, third, or fifth", requirement.implementation_notes[1])

    def test_level_5_adds_dominant_seventh(self):
        requirement = get_chord_requirement("level_5")

        self.assertEqual(
            requirement.chord_qualities,
            ("Major triad", "Minor triad", "Dominant 7th"),
        )
        self.assertIn("Close position", requirement.positions)

    def test_level_7_and_8_include_augmented_triads(self):
        self.assertIn("Augmented triad", get_chord_requirement("level_7").chord_qualities)
        self.assertIn("Augmented triad", get_chord_requirement("level_8").chord_qualities)

    def test_future_level_10_adds_seventh_chord_qualities(self):
        requirement = get_chord_requirement("level_10")

        self.assertIn("Major-major 7th", requirement.chord_qualities)
        self.assertIn("Minor-minor 7th", requirement.chord_qualities)


class ChordGenerationTests(unittest.TestCase):
    def test_chord_formulas_create_expected_notes(self):
        self.assertEqual(chord_notes("C4", MAJOR_TRIAD), ["C4", "E4", "G4"])
        self.assertEqual(chord_notes("C4", MINOR_TRIAD), ["C4", "D#4", "G4"])
        self.assertEqual(chord_notes("C4", DOMINANT_7TH), ["C4", "E4", "G4", "A#4"])
        self.assertEqual(chord_notes("C4", DIMINISHED_7TH), ["C4", "D#4", "F#4", "A4"])
        self.assertEqual(chord_notes("C4", AUGMENTED_TRIAD), ["C4", "E4", "G#4"])

    def test_chord_quality_choices_expand_by_level(self):
        self.assertEqual(chord_quality_choices(1), (MAJOR_TRIAD, MINOR_TRIAD))
        self.assertEqual(chord_quality_choices(5), (MAJOR_TRIAD, MINOR_TRIAD, DOMINANT_7TH))
        self.assertIn(DIMINISHED_7TH, chord_quality_choices(6))
        self.assertIn(AUGMENTED_TRIAD, chord_quality_choices(7))

    def test_major_minor_display_names_include_triad_for_levels_5_to_8(self):
        self.assertEqual(chord_display_name(MAJOR_TRIAD, 1), "Major")
        self.assertEqual(chord_display_name(MINOR_TRIAD, 1), "Minor")
        self.assertEqual(chord_display_name(MAJOR_TRIAD, 5), "Major (Triad)")
        self.assertEqual(chord_display_name(MINOR_TRIAD, 5), "Minor (Triad)")

    def test_possible_chord_roots_stay_inside_sample_range(self):
        roots = possible_chord_roots(DOMINANT_7TH)

        self.assertIn("F3", roots)
        self.assertNotIn("A5", roots)

    def test_level_1_chord_audio_is_broken_then_solid(self):
        waveform = create_chord_waveform(1, "C4", MAJOR_TRIAD)

        broken_samples = int(SAMPLE_RATE * 0.55) * 3
        pause_samples = int(SAMPLE_RATE * PAUSE_DURATION)
        solid_samples = int(SAMPLE_RATE * 1.4)
        self.assertEqual(len(waveform), broken_samples + pause_samples + solid_samples)

    def test_level_3_chord_audio_includes_target_tone(self):
        waveform = create_chord_waveform(3, "C4", MAJOR_TRIAD, "Third")

        broken_samples = int(SAMPLE_RATE * 0.55) * 3
        pause_samples = int(SAMPLE_RATE * PAUSE_DURATION)
        event_samples = int(SAMPLE_RATE * 1.4)
        expected_samples = event_samples + pause_samples + broken_samples + pause_samples + event_samples
        self.assertEqual(len(waveform), expected_samples)

    def test_level_3_chord_question_asks_for_quality_and_chord_tone(self):
        question = create_chord_question(3)

        self.assertEqual(question.question_type, "quality_and_tone")
        self.assertIn(question.quality_answer, ["Major", "Minor"])
        self.assertEqual(question.quality_choices, ("Major", "Minor"))
        self.assertIn(question.tone_answer, TONE_CHOICES)
        self.assertEqual(question.tone_choices, TONE_CHOICES)
        self.assertEqual(question.choices, [])

    def test_level_1_and_2_chord_choices_are_not_randomized(self):
        for level in [1, 2]:
            question = create_chord_question(level)

            with self.subTest(level=level):
                self.assertEqual(question.choices, ["Major", "Minor"])

    def test_level_5_to_8_chord_choices_follow_display_order(self):
        expected_choices = {
            5: ["Major (Triad)", "Minor (Triad)", "Dominant 7th"],
            6: ["Major (Triad)", "Minor (Triad)", "Dominant 7th", "Diminish 7th"],
            7: [
                "Major (Triad)",
                "Minor (Triad)",
                "Dominant 7th",
                "Diminish 7th",
                "Augmented (Triad)",
            ],
            8: [
                "Major (Triad)",
                "Minor (Triad)",
                "Dominant 7th",
                "Diminish 7th",
                "Augmented (Triad)",
            ],
        }

        for level, choices in expected_choices.items():
            question = create_chord_question(level)

            with self.subTest(level=level):
                self.assertEqual(question.choices, choices)


if __name__ == "__main__":
    unittest.main()
