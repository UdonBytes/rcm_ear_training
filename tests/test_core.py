import unittest

from rcm_ear_training.audio import create_interval_waveform
from rcm_ear_training.config import (
    ATTACK_THRESHOLD_RATIO,
    CONNECTED_NOTE_SPACING,
    CLAPBACK_EXAMPLE_FOLDER,
    NOTE_DURATION,
    PAUSE_DURATION,
    SAMPLE_RATE,
)
from rcm_ear_training.clapback import (
    BEAT_SECONDS,
    CLAPBACK_AUDIO_VERSION,
    LEVEL_1_CLAPBACK_EXAMPLES,
    LEVEL_1_PLAYBACK_EXAMPLES,
    LEVEL_2_PLAYBACK_EXAMPLES,
    LEVEL_3_PLAYBACK_EXAMPLES,
    LEVEL_4_PLAYBACK_EXAMPLES,
    LEVEL_5_CLAPBACK_PLAYBACK_EXAMPLES,
    beats_per_measure,
    choose_unplayed_example_index,
    create_clapback_question,
    create_clapback_waveform,
    create_playback_question,
    create_playback_waveform,
    events_with_completed_length,
    get_clapback_examples,
    get_playable_clapback_examples,
    get_playable_playback_examples,
    get_playback_examples,
    total_event_beats,
)
from rcm_ear_training.chords import get_chord_requirement
from rcm_ear_training.chord_progressions import (
    CHORD_PROGRESSION_CHORD_DURATION,
    CHORD_PROGRESSION_REPEAT_PAUSE,
    LEVEL_5_CHORD_PROGRESSION_EXAMPLES,
    LEVEL_6_MINOR_CHORD_PROGRESSION_EXAMPLES,
    STRICT_D_TO_V_BASS_LINES,
    create_chord_progression_question,
    create_chord_progression_waveform,
    get_chord_progression_examples,
    validate_strict_bass_line,
)
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
    create_chord_quality_waveform,
    create_chord_tone_waveform,
    create_chord_waveform,
    chord_display_name,
    possible_chord_roots,
)
from rcm_ear_training.curriculum import (
    CHORDS,
    CHORD_PROGRESSIONS,
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
        self.assertEqual(
            [test.id for test in implemented_tests(level)],
            [INTERVALS, CHORDS, CLAPBACK],
        )

        level_5 = get_level("level_5")
        self.assertEqual(
            [test.id for test in implemented_tests(level_5)],
            [INTERVALS, CHORDS, CHORD_PROGRESSIONS, CLAPBACK],
        )
        level_6 = get_level("level_6")
        self.assertEqual(
            [test.id for test in implemented_tests(level_6)],
            [INTERVALS, CHORDS, CHORD_PROGRESSIONS, CLAPBACK],
        )

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

    def test_level_3_split_chord_audio_maps_to_two_answer_parts(self):
        quality_waveform = create_chord_quality_waveform("C4", MAJOR_TRIAD)
        tone_waveform = create_chord_tone_waveform("C4", MAJOR_TRIAD, "Third")

        broken_samples = int(SAMPLE_RATE * 0.55) * 3
        pause_samples = int(SAMPLE_RATE * PAUSE_DURATION)
        event_samples = int(SAMPLE_RATE * 1.4)

        self.assertEqual(len(quality_waveform), event_samples)
        self.assertEqual(len(tone_waveform), broken_samples + pause_samples + event_samples)

    def test_level_3_chord_question_asks_for_quality_and_chord_tone(self):
        question = create_chord_question(3)

        self.assertEqual(question.question_type, "quality_and_tone")
        self.assertIn(question.quality_answer, ["Major", "Minor"])
        self.assertEqual(question.quality_choices, ("Major", "Minor"))
        self.assertIn(question.tone_answer, TONE_CHOICES)
        self.assertEqual(question.tone_choices, TONE_CHOICES)
        self.assertEqual(question.choices, [])
        self.assertIn("_quality_chords_v2_split_parts.wav", question.audio_file)
        self.assertIn("_tone_chords_v2_split_parts.wav", question.tone_audio_file)
        self.assertNotEqual(question.audio_file, question.tone_audio_file)

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


class ChordProgressionTests(unittest.TestCase):
    def test_level_5_chord_progressions_match_approved_pool(self):
        examples = get_chord_progression_examples(5)

        self.assertEqual(examples, LEVEL_5_CHORD_PROGRESSION_EXAMPLES)
        self.assertEqual(len(examples), 60)
        self.assertEqual(examples[0].id, "prog5-iv-1")
        self.assertEqual(examples[0].key, "C major")
        self.assertEqual(examples[0].progression, "I IV I")
        self.assertEqual(examples[0].inversion_pattern, "I6, IV, I6")
        self.assertEqual(examples[0].events[0].top_notes, ("E4", "G4", "C5"))
        self.assertEqual(examples[0].events[0].bass_note, "C3")
        self.assertEqual(examples[9].progression, "I V I")
        self.assertEqual(examples[9].events[-1].top_notes, ("G4", "C5", "E5"))
        self.assertEqual(examples[-1].id, "prog5-f-v-5")
        self.assertEqual(examples[-1].key, "F major")
        self.assertEqual(examples[-1].events[-1].top_notes, ("C4", "F4", "A4"))
        self.assertEqual(examples[-1].events[-1].bass_note, "F2")

    def test_level_5_chord_progression_voicings_match_explicit_octave_map(self):
        examples_by_id = {
            example.id: tuple(event.top_notes for event in example.events)
            for example in get_chord_progression_examples(5)
            if example.key == "C major"
        }
        expected_top_notes = {
            "prog5-iv-1": (("E4", "G4", "C5"), ("F4", "A4", "C5"), ("E4", "G4", "C5")),
            "prog5-iv-2": (("C4", "E4", "G4"), ("C4", "F4", "A4"), ("C4", "E4", "G4")),
            "prog5-iv-3": (("G3", "C4", "E4"), ("A3", "C4", "F4"), ("G3", "C4", "E4")),
            "prog5-iv-4": (("C4", "E4", "G4"), ("A3", "C4", "F4"), ("G3", "C4", "E4")),
            "prog5-iv-5": (("G3", "C4", "E4"), ("A3", "C4", "F4"), ("C4", "E4", "G4")),
            "prog5-v-1": (("C4", "E4", "G4"), ("B3", "D4", "G4"), ("C4", "E4", "G4")),
            "prog5-v-2": (("E4", "G4", "C5"), ("D4", "G4", "B4"), ("E4", "G4", "C5")),
            "prog5-v-3": (("G4", "C5", "E5"), ("G4", "B4", "D5"), ("E4", "G4", "C5")),
            "prog5-v-4": (("E4", "G4", "C5"), ("G4", "B4", "D5"), ("G4", "C5", "E5")),
            "prog5-v-5": (("G4", "C5", "E5"), ("G4", "B4", "D5"), ("G4", "C5", "E5")),
        }

        self.assertEqual(examples_by_id, expected_top_notes)

    def test_level_5_g_major_chord_progression_voicings_match_explicit_octave_map(self):
        examples_by_id = {
            example.id: tuple(event.top_notes for event in example.events)
            for example in get_chord_progression_examples(5)
            if example.key == "G major"
        }
        expected_top_notes = {
            "prog5-g-iv-1": (("B3", "D4", "G4"), ("C4", "E4", "G4"), ("B3", "D4", "G4")),
            "prog5-g-iv-2": (("G3", "B3", "D4"), ("G3", "C4", "E4"), ("G3", "B3", "D4")),
            "prog5-g-iv-3": (("D4", "G4", "B4"), ("E4", "G4", "C5"), ("D4", "G4", "B4")),
            "prog5-g-iv-4": (("G4", "B4", "D5"), ("E4", "G4", "C5"), ("D4", "G4", "B4")),
            "prog5-g-iv-5": (("D4", "G4", "B4"), ("E4", "G4", "C5"), ("G4", "B4", "D5")),
            "prog5-g-v-1": (("G4", "B4", "D5"), ("F#4", "A4", "D5"), ("G4", "B4", "D5")),
            "prog5-g-v-2": (("B3", "D4", "G4"), ("A3", "D4", "F#4"), ("B3", "D4", "G4")),
            "prog5-g-v-3": (("D4", "G4", "B4"), ("D4", "F#4", "A4"), ("B3", "D4", "G4")),
            "prog5-g-v-4": (("B3", "D4", "G4"), ("D4", "F#4", "A4"), ("D4", "G4", "B4")),
            "prog5-g-v-5": (("D4", "G4", "B4"), ("D4", "F#4", "A4"), ("D4", "G4", "B4")),
        }

        self.assertEqual(examples_by_id, expected_top_notes)

    def test_level_5_a_major_chord_progression_voicings_match_explicit_octave_map(self):
        examples = [
            example
            for example in get_chord_progression_examples(5)
            if example.key == "A major"
        ]
        examples_by_id = {
            example.id: tuple(event.top_notes for event in example.events)
            for example in examples
        }
        expected_top_notes = {
            "prog5-a-iv-1": (("C#4", "E4", "A4"), ("D4", "F#4", "A4"), ("C#4", "E4", "A4")),
            "prog5-a-iv-2": (("A3", "C#4", "E4"), ("A3", "D4", "F#4"), ("A3", "C#4", "E4")),
            "prog5-a-iv-3": (("E4", "A4", "C#5"), ("F#4", "A4", "D5"), ("E4", "A4", "C#5")),
            "prog5-a-iv-4": (("A4", "C#5", "E5"), ("F#4", "A4", "D5"), ("E4", "A4", "C#5")),
            "prog5-a-iv-5": (("E4", "A4", "C#5"), ("F#4", "A4", "D5"), ("A4", "C#5", "E5")),
            "prog5-a-v-1": (("A4", "C#5", "E5"), ("G#4", "B4", "E5"), ("A4", "C#5", "E5")),
            "prog5-a-v-2": (("C#4", "E4", "A4"), ("B3", "E4", "G#4"), ("C#4", "E4", "A4")),
            "prog5-a-v-3": (("E4", "A4", "C#5"), ("E4", "G#4", "B4"), ("C#4", "E4", "A4")),
            "prog5-a-v-4": (("C#4", "E4", "A4"), ("E4", "G#4", "B4"), ("E4", "A4", "C#5")),
            "prog5-a-v-5": (("E4", "A4", "C#5"), ("E4", "G#4", "B4"), ("E4", "A4", "C#5")),
        }

        self.assertEqual(len(examples), 10)
        self.assertEqual(examples_by_id, expected_top_notes)
        self.assertTrue(all(event.bass_note in ("A2", "D3", "E3") for example in examples for event in example.events))

    def test_level_5_d_major_chord_progression_voicings_match_explicit_octave_map(self):
        examples = [
            example
            for example in get_chord_progression_examples(5)
            if example.key == "D major"
        ]
        examples_by_id = {
            example.id: tuple(event.top_notes for event in example.events)
            for example in examples
        }
        expected_top_notes = {
            "prog5-d-iv-1": (("F#4", "A4", "D5"), ("G4", "B4", "D5"), ("F#4", "A4", "D5")),
            "prog5-d-iv-2": (("D4", "F#4", "A4"), ("D4", "G4", "B4"), ("D4", "F#4", "A4")),
            "prog5-d-iv-3": (("A3", "D4", "F#4"), ("B3", "D4", "G4"), ("A3", "D4", "F#4")),
            "prog5-d-iv-4": (("D4", "F#4", "A4"), ("B3", "D4", "G4"), ("A3", "D4", "F#4")),
            "prog5-d-iv-5": (("A3", "D4", "F#4"), ("B3", "D4", "G4"), ("D4", "F#4", "A4")),
            "prog5-d-v-1": (("D4", "F#4", "A4"), ("C#4", "E4", "A4"), ("D4", "F#4", "A4")),
            "prog5-d-v-2": (("F#4", "A4", "D5"), ("E4", "A4", "C#5"), ("F#4", "A4", "D5")),
            "prog5-d-v-3": (("A4", "D5", "F#5"), ("A4", "C#5", "E5"), ("F#4", "A4", "D5")),
            "prog5-d-v-4": (("F#4", "A4", "D5"), ("A4", "C#5", "E5"), ("A4", "D5", "F#5")),
            "prog5-d-v-5": (("A4", "D5", "F#5"), ("A4", "C#5", "E5"), ("A4", "D5", "F#5")),
        }

        self.assertEqual(len(examples), 10)
        self.assertEqual(examples_by_id, expected_top_notes)
        self.assertTrue(all(event.bass_note in ("D3", "G3", "A3") for example in examples for event in example.events))

    def test_level_5_d_major_v_progressions_use_a3_bass(self):
        examples_by_id = {
            example.id: tuple(event.bass_note for event in example.events)
            for example in get_chord_progression_examples(5)
            if example.key == "D major"
        }

        for example_id in [
            "prog5-d-v-1",
            "prog5-d-v-2",
            "prog5-d-v-3",
            "prog5-d-v-4",
            "prog5-d-v-5",
        ]:
            with self.subTest(example=example_id):
                self.assertEqual(examples_by_id[example_id], ("D3", "A3", "D3"))

    def test_strict_d_to_v_examples_reject_a2_octave_error(self):
        examples_by_id = {
            example.id: example
            for level in [5, 6]
            for example in get_chord_progression_examples(level)
        }

        for example_id, expected_bass_line in STRICT_D_TO_V_BASS_LINES.items():
            with self.subTest(example=example_id):
                example = examples_by_id[example_id]
                self.assertEqual(
                    tuple(event.bass_note for event in example.events),
                    expected_bass_line,
                )
                validate_strict_bass_line(example)

    def test_strict_d_to_v_validation_treats_a2_as_wrong_octave(self):
        example = next(
            example
            for example in get_chord_progression_examples(5)
            if example.id == "prog5-d-v-1"
        )
        bad_example = type(example)(
            id=example.id,
            level=example.level,
            key=example.key,
            progression=example.progression,
            inversion_pattern=example.inversion_pattern,
            events=(
                example.events[0],
                type(example.events[1])(
                    example.events[1].label,
                    example.events[1].top_notes,
                    "A2",
                ),
                example.events[2],
            ),
        )

        with self.assertRaises(ValueError):
            validate_strict_bass_line(bad_example)

    def test_level_5_e_major_chord_progression_voicings_match_explicit_octave_map(self):
        examples = [
            example
            for example in get_chord_progression_examples(5)
            if example.key == "E major"
        ]
        examples_by_id = {
            example.id: tuple(event.top_notes for event in example.events)
            for example in examples
        }
        expected_top_notes = {
            "prog5-e-iv-1": (("G#4", "B4", "E5"), ("A4", "C#5", "E5"), ("G#4", "B4", "E5")),
            "prog5-e-iv-2": (("E4", "G#4", "B4"), ("E4", "A4", "C#5"), ("E4", "G#4", "B4")),
            "prog5-e-iv-3": (("B3", "E4", "G#4"), ("C#4", "E4", "A4"), ("B3", "E4", "G#4")),
            "prog5-e-iv-4": (("E4", "G#4", "B4"), ("C#4", "E4", "A4"), ("B3", "E4", "G#4")),
            "prog5-e-iv-5": (("B3", "E4", "G#4"), ("C#4", "E4", "A4"), ("E4", "G#4", "B4")),
            "prog5-e-v-1": (("E4", "G#4", "B4"), ("D#4", "F#4", "B4"), ("E4", "G#4", "B4")),
            "prog5-e-v-2": (("G#4", "B4", "E5"), ("F#4", "B4", "D#5"), ("G#4", "B4", "E5")),
            "prog5-e-v-3": (("B4", "E5", "G#5"), ("B4", "D#5", "F#5"), ("G#4", "B4", "E5")),
            "prog5-e-v-4": (("G#4", "B4", "E5"), ("B4", "D#5", "F#5"), ("B4", "E5", "G#5")),
            "prog5-e-v-5": (("B4", "E5", "G#5"), ("B4", "D#5", "F#5"), ("B4", "E5", "G#5")),
        }

        self.assertEqual(len(examples), 10)
        self.assertEqual(examples_by_id, expected_top_notes)
        self.assertTrue(all(event.bass_note in ("E3", "A3", "B3") for example in examples for event in example.events))

    def test_level_5_f_major_chord_progression_voicings_match_explicit_octave_map(self):
        examples = [
            example
            for example in get_chord_progression_examples(5)
            if example.key == "F major"
        ]
        examples_by_id = {
            example.id: tuple(event.top_notes for event in example.events)
            for example in examples
        }
        expected_top_notes = {
            "prog5-f-iv-1": (("A3", "C4", "F4"), ("Bb3", "D4", "F4"), ("A3", "C4", "F4")),
            "prog5-f-iv-2": (("F3", "A3", "C4"), ("F3", "Bb3", "D4"), ("F3", "A3", "C4")),
            "prog5-f-iv-3": (("C4", "F4", "A4"), ("D4", "F4", "Bb4"), ("C4", "F4", "A4")),
            "prog5-f-iv-4": (("F4", "A4", "C5"), ("D4", "F4", "Bb4"), ("C4", "F4", "A4")),
            "prog5-f-iv-5": (("C4", "F4", "A4"), ("D4", "F4", "Bb4"), ("F4", "A4", "C5")),
            "prog5-f-v-1": (("F4", "A4", "C5"), ("E4", "G4", "C5"), ("F4", "A4", "C5")),
            "prog5-f-v-2": (("A3", "C4", "F4"), ("G3", "C4", "E4"), ("A3", "C4", "F4")),
            "prog5-f-v-3": (("C4", "F4", "A4"), ("C4", "E4", "G4"), ("A3", "C4", "F4")),
            "prog5-f-v-4": (("A3", "C4", "F4"), ("C4", "E4", "G4"), ("C4", "F4", "A4")),
            "prog5-f-v-5": (("C4", "F4", "A4"), ("C4", "E4", "G4"), ("C4", "F4", "A4")),
        }

        self.assertEqual(len(examples), 10)
        self.assertEqual(examples_by_id, expected_top_notes)
        self.assertTrue(all(event.bass_note in ("F2", "Bb2", "C3") for example in examples for event in example.events))

    def test_level_5_chord_progression_waveform_plays_twice(self):
        example = LEVEL_5_CHORD_PROGRESSION_EXAMPLES[0]
        waveform = create_chord_progression_waveform(example)

        self.assertEqual(len(waveform), SAMPLE_RATE * 7)

    def test_chord_progression_waveform_has_no_inter_chord_silence(self):
        example = LEVEL_5_CHORD_PROGRESSION_EXAMPLES[0]
        waveform = create_chord_progression_waveform(example)
        progression_pass_samples = int(
            SAMPLE_RATE * CHORD_PROGRESSION_CHORD_DURATION * len(example.events)
        )
        repeat_pause_samples = int(SAMPLE_RATE * CHORD_PROGRESSION_REPEAT_PAUSE)

        self.assertEqual(
            len(waveform),
            (progression_pass_samples * 2) + repeat_pause_samples,
        )

    def test_level_6_chord_progressions_include_level_5_and_minor_pool(self):
        examples = get_chord_progression_examples(6)

        self.assertEqual(len(examples), 120)
        self.assertEqual(examples[:60], LEVEL_5_CHORD_PROGRESSION_EXAMPLES)
        self.assertEqual(examples[60:], LEVEL_6_MINOR_CHORD_PROGRESSION_EXAMPLES)
        self.assertEqual(examples[60].id, "prog6-f-minor-iv-1")
        self.assertEqual(examples[60].key, "F minor")
        self.assertEqual(examples[60].progression, "i iv i")
        self.assertEqual(examples[60].events[0].top_notes, ("Ab3", "C4", "F4"))
        self.assertEqual(examples[60].events[1].top_notes, ("Bb3", "Db4", "F4"))
        self.assertEqual(examples[60].events[0].bass_note, "F2")
        self.assertEqual(examples[-1].id, "prog6-e-minor-v-5")
        self.assertEqual(examples[-1].key, "E minor")
        self.assertEqual(examples[-1].progression, "i V i")
        self.assertEqual(examples[-1].events[-1].top_notes, ("B4", "E5", "G5"))
        self.assertEqual(examples[-1].events[-1].bass_note, "E3")

    def test_level_6_minor_chord_progression_voicings_match_explicit_octave_map(self):
        examples = get_chord_progression_examples(6)[60:]
        examples_by_id = {
            example.id: tuple(event.top_notes for event in example.events)
            for example in examples
        }

        expected_spot_checks = {
            "prog6-g-minor-v-2": (("Bb3", "D4", "G4"), ("A3", "D4", "F#4"), ("Bb3", "D4", "G4")),
            "prog6-a-minor-iv-4": (("A4", "C5", "E5"), ("F4", "A4", "D5"), ("E4", "A4", "C5")),
            "prog6-c-minor-v-3": (("G4", "C5", "Eb5"), ("G4", "B4", "D5"), ("Eb4", "G4", "C5")),
            "prog6-d-minor-v-5": (("A4", "D5", "F5"), ("A4", "C#5", "E5"), ("A4", "D5", "F5")),
            "prog6-e-minor-v-5": (("B4", "E5", "G5"), ("B4", "D#5", "F#5"), ("B4", "E5", "G5")),
        }

        for example_id, expected_top_notes in expected_spot_checks.items():
            with self.subTest(example=example_id):
                self.assertEqual(examples_by_id[example_id], expected_top_notes)

    def test_create_level_5_chord_progression_question_creates_audio(self):
        question = create_chord_progression_question(5, 0)

        self.assertEqual(question.example_id, "prog5-iv-1")
        self.assertEqual(question.key, "C major")
        self.assertEqual(question.progression, "I IV I")
        self.assertEqual(question.answer, "I-IV-I")
        self.assertEqual(question.choices, ("I-IV-I", "I-V-I"))
        self.assertIn("chord_progressions/", question.audio_file)

    def test_create_level_6_chord_progression_question_groups_major_and_minor_choices(self):
        question = create_chord_progression_question(6, 60)

        self.assertEqual(question.example_id, "prog6-f-minor-iv-1")
        self.assertEqual(question.key, "F minor")
        self.assertEqual(question.progression, "i iv i")
        self.assertEqual(question.answer, "I-IV-I / i-iv-i")
        self.assertEqual(question.choices, ("I-IV-I / i-iv-i", "I-V-I / i-V-i"))
        self.assertIn("chord_progressions/", question.audio_file)

    def test_create_level_6_major_chord_progression_uses_grouped_answer(self):
        question = create_chord_progression_question(6, 0)

        self.assertEqual(question.example_id, "prog5-iv-1")
        self.assertEqual(question.progression, "I IV I")
        self.assertEqual(question.answer, "I-IV-I / i-iv-i")


class ClapbackTests(unittest.TestCase):
    def test_level_1_contains_approved_source_examples(self):
        examples = get_clapback_examples(1)

        self.assertEqual(examples, LEVEL_1_CLAPBACK_EXAMPLES)
        self.assertEqual(len(examples), 23)
        self.assertEqual(examples[0].id, "clap1-1")
        self.assertEqual(examples[0].key, "D major")
        self.assertEqual(examples[0].status, "approved")
        self.assertEqual(examples[1].id, "clap1-2")
        self.assertEqual(examples[1].status, "approved")
        self.assertNotEqual(examples[1].draft_events, ())
        self.assertEqual(examples[2].id, "clap1-3")
        self.assertEqual(examples[2].status, "approved")
        self.assertEqual(
            tuple((event.note, event.beats) for event in examples[2].draft_events),
            (
                ("E4", 1.5),
                ("C4", 0.5),
                ("D4", 1),
                ("E4", 1),
                ("E4", 0.5),
                ("F4", 0.5),
                ("G4", 1),
                ("C4", 3),
            ),
        )
        self.assertEqual(examples[3].id, "clap1-4")
        self.assertEqual(examples[3].status, "approved")
        self.assertEqual(
            tuple((event.note, event.beats) for event in examples[3].draft_events),
            (
                ("D5", 1),
                ("B4", 0.5),
                ("A4", 0.5),
                ("B4", 1.5),
                ("C5", 0.5),
                ("A4", 1),
                ("F#4", 1),
                ("G4", 2),
            ),
        )
        self.assertEqual(examples[4].id, "clap1-5")
        self.assertEqual(examples[4].status, "approved")
        self.assertEqual(
            tuple((event.note, event.beats) for event in examples[4].draft_events),
            (
                ("F4", 1),
                ("F4", 1),
                ("A4", 0.5),
                ("F4", 0.5),
                ("C4", 1),
                ("C4", 1.5),
                ("E4", 0.5),
                ("F4", 3),
            ),
        )
        self.assertTrue((CLAPBACK_EXAMPLE_FOLDER / examples[0].image_file).exists())
        self.assertTrue((CLAPBACK_EXAMPLE_FOLDER / examples[1].image_file).exists())
        self.assertTrue((CLAPBACK_EXAMPLE_FOLDER / examples[2].image_file).exists())
        self.assertTrue((CLAPBACK_EXAMPLE_FOLDER / examples[4].image_file).exists())
        expected_new_examples = {
            "clap1-6": (
                ("C4", 0.5), ("D4", 0.5), ("E4", 0.5), ("C4", 0.5),
                ("F4", 1.5), ("F4", 0.5), ("G4", 1), ("C5", 1),
                ("F4", 2),
            ),
            "clap1-7": (
                ("C4", 0.5), ("D4", 0.5), ("E4", 0.5), ("F4", 0.5),
                ("G4", 1), ("F4", 1.5), ("E4", 0.5), ("D4", 1),
                ("C4", 3),
            ),
            "clap1-8": (
                ("C5", 0.5), ("Bb4", 0.5), ("A4", 1), ("Bb4", 1.5),
                ("A4", 0.5), ("G4", 1), ("E4", 1), ("F4", 2),
            ),
            "clap1-9": (
                ("C4", 1), ("F4", 0.5), ("G4", 0.5), ("A4", 1),
                ("C4", 1.5), ("A4", 0.5), ("G4", 1), ("F4", 3),
            ),
            "clap1-10": (
                ("A4", 1.5), ("A4", 0.5), ("G4", 0.5), ("F#4", 0.5),
                ("E4", 0.5), ("D4", 0.5), ("C#4", 1), ("E4", 1),
                ("D4", 2),
            ),
            "clap1-11": (
                ("D5", 1), ("C5", 0.5), ("B4", 0.5), ("A4", 0.5),
                ("G4", 0.5), ("F#4", 1), ("E4", 0.5), ("D4", 0.5),
                ("F#4", 1), ("G4", 3),
            ),
            "clap1-12": (
                ("C4", 1.5), ("D4", 0.5), ("E4", 1), ("G4", 1),
                ("F4", 1), ("D4", 1), ("C4", 2),
            ),
            "clap1-13": (
                ("G4", 1), ("A4", 1), ("B4", 1), ("C5", 0.5),
                ("D5", 0.5), ("B4", 1.5), ("A4", 0.5), ("G4", 2),
            ),
            "clap1-14": (
                ("G5", 0.5), ("A5", 0.5), ("G5", 1), ("G5", 1),
                ("E5", 1.5), ("E5", 0.5), ("D5", 1), ("C5", 3),
            ),
            "clap1-15": (
                ("Bb3", 1.5), ("D4", 0.5), ("F4", 2), ("G4", 1),
                ("A4", 0.5), ("F4", 0.5), ("Bb4", 2),
            ),
            "clap1-16": (
                ("D4", 0.5), ("F#4", 0.5), ("A4", 1), ("A4", 1),
                ("B4", 1.5), ("G4", 0.5), ("E4", 1), ("D4", 3),
            ),
            "clap1-17": (
                ("B4", 0.5), ("C5", 0.5), ("D5", 1), ("G5", 1),
                ("D5", 1), ("C5", 1.5), ("A4", 0.5), ("B4", 2),
            ),
            "clap1-18": (
                ("G4", 0.5), ("A4", 0.5), ("G4", 1), ("E4", 0.5),
                ("G4", 0.5), ("F4", 0.5), ("G4", 0.5), ("F4", 1),
                ("D4", 1), ("C4", 3),
            ),
            "clap1-19": (
                ("D5", 1.5), ("C#5", 0.5), ("D5", 1), ("A4", 1),
                ("B4", 1), ("A4", 0.5), ("G4", 0.5), ("F#4", 2),
            ),
            "clap1-20": (
                ("G4", 0.5), ("A4", 0.5), ("B4", 1), ("C5", 1),
                ("D5", 0.5), ("E5", 0.5), ("D5", 1.5), ("A4", 0.5),
                ("B4", 3),
            ),
            "clap1-21": (
                ("D5", 1), ("G4", 0.5), ("A4", 0.5), ("B4", 1),
                ("C5", 1.5), ("F#4", 0.5), ("A4", 1), ("G4", 3),
            ),
            "clap1-22": (
                ("C4", 1.5), ("E4", 0.5), ("G4", 1.5), ("E4", 0.5),
                ("F4", 0.5), ("E4", 0.5), ("D4", 0.5), ("E4", 0.5),
                ("C4", 2),
            ),
            "clap1-23": (
                ("F5", 0.5), ("E5", 0.5), ("D5", 0.5), ("C5", 0.5),
                ("Bb4", 1.5), ("C5", 0.5), ("A4", 0.5), ("Bb4", 0.5),
                ("G4", 1), ("F4", 2),
            ),
        }

        for example in examples[5:]:
            self.assertEqual(example.status, "approved")
            if example.id in expected_new_examples:
                self.assertEqual(
                    tuple((event.note, event.beats) for event in example.draft_events),
                    expected_new_examples[example.id],
                )
            self.assertTrue((CLAPBACK_EXAMPLE_FOLDER / example.image_file).exists())

    def test_level_1_playable_examples_include_only_approved_audio(self):
        examples = get_playable_clapback_examples(1)

        self.assertEqual(len(examples), 23)
        self.assertEqual(examples[0].id, "clap1-1")
        self.assertEqual(examples[1].id, "clap1-2")
        self.assertEqual(examples[2].id, "clap1-3")
        self.assertEqual(examples[3].id, "clap1-4")
        self.assertEqual(examples[4].id, "clap1-5")
        self.assertTrue((CLAPBACK_EXAMPLE_FOLDER / examples[0].image_file).exists())

    def test_level_1_contains_approved_playback_examples(self):
        examples = get_playback_examples(1)

        self.assertEqual(examples, LEVEL_1_PLAYBACK_EXAMPLES)
        self.assertEqual(len(examples), 66)
        self.assertEqual(examples[0].id, "play1-1a")
        self.assertEqual(examples[0].key, "C major")
        self.assertEqual(examples[0].starting_chord_notes, ("C4", "E4", "G4"))
        self.assertEqual(examples[0].playback_repetitions, 2)
        self.assertEqual(
            tuple((event.note, event.beats) for event in examples[0].draft_events),
            (
                ("C4", 1),
                ("D4", 1),
                ("E4", 1),
                ("G4", 1),
                ("E4", 4),
            ),
        )
        self.assertEqual(examples[1].id, "play1-1b")
        self.assertEqual(examples[1].starting_chord_notes, ("G4", "B4", "D5"))
        self.assertEqual(examples[2].id, "play1-1c")
        self.assertEqual(examples[2].starting_chord_notes, ("A4", "C5", "E5"))
        self.assertEqual(examples[-1].id, "play1-22c")
        self.assertEqual(
            tuple((event.note, event.beats) for event in examples[-1].draft_events),
            (
                ("A4", 1),
                ("B4", 1),
                ("D5", 1),
                ("D5", 1),
                ("E5", 4),
            ),
        )

        for example in examples:
            with self.subTest(example=example.id):
                self.assertEqual(example.status, "approved")
                self.assertEqual(example.time_signature, "4/4")
                self.assertEqual(example.measures, 2)
                self.assertEqual(example.playback_repetitions, 2)
                self.assertEqual(
                    total_event_beats(events_with_completed_length(example)),
                    beats_per_measure(example.time_signature) * example.measures,
                )

    def test_level_1_playable_playback_examples(self):
        examples = get_playable_playback_examples(1)

        self.assertEqual(len(examples), 66)
        self.assertEqual(examples[0].id, "play1-1a")

    def test_level_2_contains_approved_playback_examples(self):
        examples = get_playback_examples(2)

        self.assertEqual(examples, LEVEL_2_PLAYBACK_EXAMPLES)
        self.assertEqual(len(examples), 66)
        self.assertEqual(examples[0].id, "play2-1a")
        self.assertEqual(examples[0].key, "F major")
        self.assertEqual(examples[0].starting_chord_notes, ("F4", "A4", "C5"))
        self.assertEqual(
            tuple((event.note, event.beats) for event in examples[0].draft_events),
            (
                ("C5", 1),
                ("Bb4", 1),
                ("Bb4", 1),
                ("G4", 1),
                ("F4", 4),
            ),
        )
        self.assertEqual(examples[1].id, "play2-1b")
        self.assertEqual(examples[1].starting_chord_notes, ("G4", "B4", "D5"))
        self.assertEqual(examples[2].id, "play2-1c")
        self.assertEqual(examples[2].starting_chord_notes, ("D4", "F4", "A4"))

        examples_by_id = {example.id: example for example in examples}
        expected_example_21_events = {
            "play2-21a": (
                ("A4", 1), ("C5", 1), ("G4", 1), ("Bb4", 1), ("F4", 4),
            ),
            "play2-21b": (
                ("B4", 1), ("D5", 1), ("A4", 1), ("C5", 1), ("G4", 4),
            ),
            "play2-21c": (
                ("F4", 1), ("A4", 1), ("E4", 1), ("G4", 1), ("D4", 4),
            ),
        }

        for example_id, expected_events in expected_example_21_events.items():
            with self.subTest(example=example_id):
                self.assertEqual(
                    tuple(
                        (event.note, event.beats)
                        for event in examples_by_id[example_id].draft_events
                    ),
                    expected_events,
                )

        self.assertEqual(examples[-1].id, "play2-22c")
        self.assertEqual(
            tuple((event.note, event.beats) for event in examples[-1].draft_events),
            (
                ("D4", 1),
                ("E4", 1),
                ("E4", 1),
                ("G4", 1),
                ("A4", 4),
            ),
        )

        for example in examples:
            with self.subTest(example=example.id):
                self.assertEqual(example.status, "approved")
                self.assertEqual(example.time_signature, "4/4")
                self.assertEqual(example.measures, 2)
                self.assertEqual(example.playback_repetitions, 2)
                self.assertEqual(
                    total_event_beats(events_with_completed_length(example)),
                    beats_per_measure(example.time_signature) * example.measures,
                )

    def test_level_2_playable_playback_examples(self):
        examples = get_playable_playback_examples(2)

        self.assertEqual(len(examples), 66)
        self.assertEqual(examples[0].id, "play2-1a")

    def test_level_3_contains_approved_playback_examples(self):
        examples = get_playback_examples(3)

        self.assertEqual(examples, LEVEL_3_PLAYBACK_EXAMPLES)
        self.assertEqual(len(examples), 88)
        self.assertEqual(examples[0].id, "play3-1a")
        self.assertEqual(examples[0].key, "F major")
        self.assertEqual(examples[0].starting_chord_notes, ("F4", "A4", "C5"))
        self.assertEqual(
            tuple((event.note, event.beats) for event in examples[0].draft_events),
            (
                ("F4", 0.5),
                ("G4", 0.5),
                ("A4", 1),
                ("F4", 1),
                ("C5", 3),
            ),
        )

        examples_by_id = {example.id: example for example in examples}
        expected_triads = {
            "play3-1a": ("F4", "A4", "C5"),
            "play3-1b": ("G4", "Bb4", "D5"),
            "play3-1c": ("D4", "F#4", "A4"),
            "play3-1d": ("D4", "F4", "A4"),
        }

        for example_id, expected_triad in expected_triads.items():
            with self.subTest(triad=example_id):
                self.assertEqual(
                    examples_by_id[example_id].starting_chord_notes,
                    expected_triad,
                )

        self.assertEqual(
            tuple((event.note, event.beats) for event in examples_by_id["play3-16a"].draft_events),
            (
                ("A4", 1),
                ("C5", 1),
                ("F4", 1),
                ("G4", 1),
                ("A4", 4),
            ),
        )
        self.assertEqual(examples_by_id["play3-18a"].time_signature, "3/4")
        self.assertEqual(
            tuple((event.note, event.beats) for event in examples_by_id["play3-18a"].draft_events),
            (
                ("F4", 0.5),
                ("G4", 0.5),
                ("A4", 1),
                ("Bb4", 0.5),
                ("C5", 0.5),
                ("F4", 3),
            ),
        )
        self.assertEqual(examples[-1].id, "play3-22d")
        self.assertEqual(
            tuple((event.note, event.beats) for event in examples[-1].draft_events),
            (
                ("D4", 1),
                ("E4", 1),
                ("F4", 1),
                ("A4", 0.5),
                ("F4", 0.5),
                ("D4", 4),
            ),
        )

        for example in examples:
            with self.subTest(example=example.id):
                self.assertEqual(example.status, "approved")
                self.assertEqual(example.measures, 2)
                self.assertEqual(example.playback_repetitions, 2)
                self.assertEqual(
                    total_event_beats(events_with_completed_length(example)),
                    beats_per_measure(example.time_signature) * example.measures,
                )

    def test_level_3_playable_playback_examples(self):
        examples = get_playable_playback_examples(3)

        self.assertEqual(len(examples), 88)
        self.assertEqual(examples[0].id, "play3-1a")

    def test_level_4_contains_approved_playback_examples(self):
        examples = get_playback_examples(4)

        self.assertEqual(examples, LEVEL_4_PLAYBACK_EXAMPLES)
        self.assertEqual(len(examples), 88)
        self.assertEqual(examples[0].id, "play4-1a")
        self.assertEqual(examples[0].key, "C minor")
        self.assertEqual(examples[0].starting_chord_notes, ("C4", "Eb4", "G4"))
        self.assertEqual(
            tuple((event.note, event.beats) for event in examples[0].draft_events),
            (
                ("G4", 1),
                ("C4", 0.5),
                ("G4", 0.5),
                ("F4", 1),
                ("D4", 1),
                ("Eb4", 2),
            ),
        )

        examples_by_id = {example.id: example for example in examples}
        expected_triads = {
            "play4-1a": ("C4", "Eb4", "G4"),
            "play4-1b": ("G4", "Bb4", "D5"),
            "play4-1c": ("D4", "F#4", "A4"),
            "play4-1d": ("A4", "C#5", "E5"),
        }

        for example_id, expected_triad in expected_triads.items():
            with self.subTest(triad=example_id):
                self.assertEqual(
                    examples_by_id[example_id].starting_chord_notes,
                    expected_triad,
                )

        self.assertEqual(
            tuple((event.note, event.beats) for event in examples_by_id["play4-4a"].draft_events),
            (
                ("C4", 0.5),
                ("G4", 0.5),
                ("F4", 0.5),
                ("G4", 0.5),
                ("Eb4", 1),
                ("D4", 1),
                ("C4", 2),
            ),
        )
        self.assertEqual(examples_by_id["play4-19a"].measures, 4)
        self.assertEqual(examples_by_id["play4-22d"].measures, 3)
        self.assertEqual(examples[-1].id, "play4-22d")
        self.assertEqual(
            tuple((event.note, event.beats) for event in examples[-1].draft_events),
            (
                ("C#5", 1),
                ("D5", 1),
                ("E5", 2),
                ("C#5", 1.5),
                ("B4", 0.5),
                ("A4", 2),
                ("D5", 2),
                ("C#5", 2),
            ),
        )

        for example in examples:
            with self.subTest(example=example.id):
                self.assertEqual(example.status, "approved")
                self.assertEqual(example.playback_repetitions, 2)
                self.assertLessEqual(
                    total_event_beats(example.draft_events),
                    beats_per_measure(example.time_signature) * example.measures,
                )
                self.assertEqual(
                    total_event_beats(events_with_completed_length(example)),
                    beats_per_measure(example.time_signature) * example.measures,
                )

    def test_level_4_playable_playback_examples(self):
        examples = get_playable_playback_examples(4)

        self.assertEqual(len(examples), 88)
        self.assertEqual(examples[0].id, "play4-1a")

    def test_level_5_contains_approved_clapback_playback_examples(self):
        examples = get_clapback_examples(5)

        self.assertEqual(examples, LEVEL_5_CLAPBACK_PLAYBACK_EXAMPLES)
        self.assertEqual(len(examples), 22)
        self.assertEqual(examples[0].id, "clap5-1")
        self.assertEqual(examples[0].key, "E major")
        self.assertEqual(examples[0].starting_chord_label, "E major chord")
        self.assertEqual(examples[0].starting_chord_notes, ("E4", "G#4", "B4", "E5"))
        self.assertEqual(
            tuple((event.note, event.beats) for event in examples[0].draft_events),
            (
                ("E4", 1.5),
                ("G#4", 0.5),
                ("B4", 0.5),
                ("A4", 0.5),
                ("G#4", 0.5),
                ("F#4", 0.5),
                ("G#4", 4),
            ),
        )
        self.assertEqual(examples[-1].id, "clap5-22")
        self.assertEqual(examples[-1].key, "A minor")
        self.assertEqual(examples[-1].starting_chord_notes, ("A4", "C5", "E5", "A5"))

        examples_by_id = {example.id: example for example in examples}
        expected_corrections = {
            "clap5-6": {
                "starting_chord_notes": ("A4", "C#5", "E5", "A5"),
            },
            "clap5-7": {
                "events": (
                    ("B4", 1), ("E5", 1), ("G#4", 1), ("F#4", 1.5),
                    ("G#4", 0.5), ("A4", 1), ("G#4", 3),
                ),
            },
            "clap5-8": {
                "events": (
                    ("G4", 1), ("G4", 0.5), ("A4", 0.5), ("B4", 1),
                    ("F#4", 2), ("B4", 1), ("E4", 3),
                ),
            },
            "clap5-9": {
                "starting_chord_notes": ("A4", "C5", "E5", "A5"),
            },
            "clap5-11": {
                "starting_chord_notes": ("A4", "C5", "E5", "A5"),
            },
            "clap5-13": {
                "starting_chord_notes": ("A4", "C5", "E5", "A5"),
            },
            "clap5-15": {
                "starting_chord_notes": ("A4", "C5", "E5", "A5"),
            },
            "clap5-17": {
                "starting_chord_notes": ("A4", "C5", "E5", "A5"),
            },
            "clap5-21": {
                "events": (
                    ("G#4", 1.5), ("A4", 0.5), ("G#4", 1), ("E5", 2),
                    ("B4", 0.5), ("A4", 0.5), ("G#4", 3),
                ),
            },
            "clap5-22": {
                "starting_chord_notes": ("A4", "C5", "E5", "A5"),
            },
        }

        for example_id, expectation in expected_corrections.items():
            with self.subTest(correction=example_id):
                example = examples_by_id[example_id]

                if "starting_chord_notes" in expectation:
                    self.assertEqual(
                        example.starting_chord_notes,
                        expectation["starting_chord_notes"],
                    )

                if "events" in expectation:
                    self.assertEqual(
                        tuple((event.note, event.beats) for event in example.draft_events),
                        expectation["events"],
                    )

        for example in examples:
            with self.subTest(example=example.id):
                self.assertEqual(example.status, "approved")
                self.assertNotEqual(example.starting_chord_notes, ())
                self.assertTrue((CLAPBACK_EXAMPLE_FOLDER / example.image_file).exists())
                self.assertEqual(
                    total_event_beats(events_with_completed_length(example)),
                    beats_per_measure(example.time_signature) * example.measures,
                )

    def test_level_5_clapback_and_playback_use_separate_audio_files(self):
        question = create_clapback_question(5)

        self.assertEqual(question.example_id, "clap5-1")
        self.assertEqual(question.starting_chord_label, "E major chord")
        self.assertIn("_clapback_", question.audio_file)
        self.assertIn("_playback_", question.playback_audio_file)
        self.assertNotEqual(question.audio_file, question.playback_audio_file)

    def test_create_level_1_playback_question_creates_audio(self):
        question = create_playback_question(1)

        self.assertEqual(question.example_id, "play1-1a")
        self.assertEqual(question.time_signature, "4/4")
        self.assertEqual(question.key, "C major")
        self.assertIn("_playback_", question.audio_file)

    def test_create_level_2_playback_question_creates_audio(self):
        question = create_playback_question(2)

        self.assertEqual(question.example_id, "play2-1a")
        self.assertEqual(question.time_signature, "4/4")
        self.assertEqual(question.key, "F major")
        self.assertIn("_playback_", question.audio_file)

    def test_create_level_3_playback_question_creates_audio(self):
        question = create_playback_question(3)

        self.assertEqual(question.example_id, "play3-1a")
        self.assertEqual(question.time_signature, "3/4")
        self.assertEqual(question.key, "F major")
        self.assertIn("_playback_", question.audio_file)

    def test_create_level_4_playback_question_creates_audio(self):
        question = create_playback_question(4)

        self.assertEqual(question.example_id, "play4-1a")
        self.assertEqual(question.time_signature, "4/4")
        self.assertEqual(question.key, "C minor")
        self.assertIn("_playback_", question.audio_file)

    def test_random_clapback_rotation_prefers_unplayed_examples(self):
        examples = get_playable_clapback_examples(1)
        chosen_index = choose_unplayed_example_index(
            examples,
            played_example_ids=[example.id for example in examples[:-1]],
        )

        self.assertEqual(chosen_index, 22)

    def test_random_clapback_rotation_prefers_different_key(self):
        examples = get_playable_clapback_examples(1)
        f_major_ids = [
            example.id
            for example in examples
            if example.key == "F major"
        ]
        chosen_index = choose_unplayed_example_index(
            examples,
            played_example_ids=[
                example.id
                for example in examples
                if example.key != "F major"
            ],
            previous_key="F major",
        )

        self.assertIn(examples[chosen_index].id, f_major_ids)

        chosen_index = choose_unplayed_example_index(
            examples,
            played_example_ids=[],
            previous_key="F major",
        )

        self.assertNotEqual(examples[chosen_index].key, "F major")

    def test_draft_audio_completes_written_length_and_repeats(self):
        example = LEVEL_1_CLAPBACK_EXAMPLES[0]
        waveform = create_clapback_waveform(example)
        count_in_samples = int(SAMPLE_RATE * BEAT_SECONDS) * beats_per_measure(
            example.time_signature
        )
        melody_samples = sum(
            int(SAMPLE_RATE * event.beats * BEAT_SECONDS)
            for event in events_with_completed_length(example)
        )
        rest_samples = int(
            SAMPLE_RATE * beats_per_measure(example.time_signature) * BEAT_SECONDS
        )

        self.assertEqual(
            len(waveform),
            count_in_samples + melody_samples + rest_samples + melody_samples,
        )

    def test_level_5_clapback_and_playback_lengths(self):
        example = LEVEL_5_CLAPBACK_PLAYBACK_EXAMPLES[0]
        clapback_waveform = create_clapback_waveform(example)
        playback_waveform = create_playback_waveform(example)
        intro_samples = int(SAMPLE_RATE * BEAT_SECONDS * 3)
        count_in_samples = int(SAMPLE_RATE * BEAT_SECONDS) * beats_per_measure(
            example.time_signature
        )
        melody_samples = sum(
            int(SAMPLE_RATE * event.beats * BEAT_SECONDS)
            for event in events_with_completed_length(example)
        )
        rest_samples = int(
            SAMPLE_RATE * beats_per_measure(example.time_signature) * BEAT_SECONDS
        )

        self.assertEqual(
            len(clapback_waveform),
            intro_samples + count_in_samples + melody_samples + rest_samples + melody_samples,
        )
        self.assertEqual(len(playback_waveform), intro_samples + melody_samples)

    def test_level_1_playback_audio_plays_melody_twice(self):
        example = LEVEL_1_PLAYBACK_EXAMPLES[0]
        waveform = create_playback_waveform(example)
        intro_samples = int(SAMPLE_RATE * BEAT_SECONDS * 3)
        melody_samples = sum(
            int(SAMPLE_RATE * event.beats * BEAT_SECONDS)
            for event in events_with_completed_length(example)
        )
        rest_samples = int(
            SAMPLE_RATE * beats_per_measure(example.time_signature) * BEAT_SECONDS
        )

        self.assertEqual(
            len(waveform),
            intro_samples + melody_samples + rest_samples + melody_samples,
        )

    def test_create_clapback_question_creates_approved_audio(self):
        question = create_clapback_question(1)

        self.assertEqual(question.example_id, "clap1-1")
        self.assertEqual(question.status, "approved")
        self.assertTrue(
            question.audio_file.endswith(f"_clapback_{CLAPBACK_AUDIO_VERSION}.wav")
        )

    def test_create_clapback_question_creates_all_approved_audio(self):
        for index in range(23):
            with self.subTest(index=index):
                question = create_clapback_question(1, index)

                self.assertEqual(question.example_id, f"clap1-{index + 1}")
                self.assertEqual(question.status, "approved")
                self.assertTrue(
                    question.audio_file.endswith(
                        f"_clapback_{CLAPBACK_AUDIO_VERSION}.wav"
                    )
                )


if __name__ == "__main__":
    unittest.main()
