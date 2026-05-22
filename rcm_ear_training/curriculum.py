"""Curriculum map for level and test navigation."""

from dataclasses import dataclass


IMPLEMENTED = "implemented"
PLANNED = "planned"
FUTURE = "future"

INTERVALS = "intervals"
CHORDS = "chords"
CLAPBACK = "clapback"
PLAYBACK = "playback"
CHORD_PROGRESSIONS = "chord_progressions"


@dataclass(frozen=True)
class TestDefinition:
    id: str
    label: str
    status: str
    description: str


@dataclass(frozen=True)
class LevelDefinition:
    id: str
    label: str
    status: str
    interval_level: int | None
    tests: tuple[TestDefinition, ...]


INTERVAL_TEST = TestDefinition(
    id=INTERVALS,
    label="Intervals",
    status=IMPLEMENTED,
    description="Identify melodic and harmonic intervals for this level.",
)
CHORD_TEST = TestDefinition(
    id=CHORDS,
    label="Chords",
    status=IMPLEMENTED,
    description="View chord requirements and planned chord-test buildout.",
)
CLAPBACK_TEST = TestDefinition(
    id=CLAPBACK,
    label="Clapback",
    status=PLANNED,
    description="Clap, tap, or sing back a short melody or rhythm.",
)
PLAYBACK_TEST = TestDefinition(
    id=PLAYBACK,
    label="Playback",
    status=PLANNED,
    description="Play back a short melody after hearing it.",
)
CHORD_PROGRESSIONS_TEST = TestDefinition(
    id=CHORD_PROGRESSIONS,
    label="Chord Progressions",
    status=PLANNED,
    description="Identify common chord progressions at advanced levels.",
)

FOUNDATION_TESTS = (
    CLAPBACK_TEST,
    CHORD_TEST,
    PLAYBACK_TEST,
)
LEVEL_1_TO_8_TESTS = (
    INTERVAL_TEST,
    CHORD_TEST,
    CLAPBACK_TEST,
    PLAYBACK_TEST,
)
ADVANCED_TESTS = (
    INTERVAL_TEST,
    CHORD_TEST,
    CHORD_PROGRESSIONS_TEST,
    CLAPBACK_TEST,
    PLAYBACK_TEST,
)

CURRENT_LEVELS = tuple(
    LevelDefinition(
        id=f"level_{level}",
        label=f"Level {level}",
        status=IMPLEMENTED,
        interval_level=level,
        tests=LEVEL_1_TO_8_TESTS,
    )
    for level in range(1, 9)
)

FUTURE_LEVELS = (
    LevelDefinition(
        id="prep_a",
        label="Preparatory A",
        status=FUTURE,
        interval_level=None,
        tests=FOUNDATION_TESTS,
    ),
    LevelDefinition(
        id="prep_b",
        label="Preparatory B",
        status=FUTURE,
        interval_level=None,
        tests=FOUNDATION_TESTS,
    ),
    LevelDefinition(
        id="level_9",
        label="Level 9",
        status=FUTURE,
        interval_level=9,
        tests=ADVANCED_TESTS,
    ),
    LevelDefinition(
        id="level_10",
        label="Level 10",
        status=FUTURE,
        interval_level=10,
        tests=ADVANCED_TESTS,
    ),
    LevelDefinition(
        id="arct",
        label="ARCT",
        status=FUTURE,
        interval_level=None,
        tests=ADVANCED_TESTS,
    ),
)

ALL_LEVELS = CURRENT_LEVELS + FUTURE_LEVELS
LEVELS_BY_ID = {level.id: level for level in ALL_LEVELS}


def get_level(level_id):
    """Return one level definition by id."""

    return LEVELS_BY_ID[level_id]


def get_test(level, test_id):
    """Return one test definition from a level."""

    for test in level.tests:
        if test.id == test_id:
            return test

    raise KeyError(test_id)


def current_levels():
    """Return levels currently selectable in the app."""

    return CURRENT_LEVELS


def future_levels():
    """Return planned future levels."""

    return FUTURE_LEVELS


def implemented_tests(level):
    """Return tests currently implemented for a level."""

    return tuple(test for test in level.tests if test.status == IMPLEMENTED)
