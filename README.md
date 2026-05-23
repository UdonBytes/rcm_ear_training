# RCM Ear Tests

A simple ear training practice app for RCM students.

**Live App:** https://rcm-ear-training.streamlit.app/

Students can choose their level and practise interval identification using piano samples. The app plays intervals according to the level requirements, then students choose the correct answer.

## Features

- Level 1 to Level 8 interval practice
- Piano sample based playback
- Level specific interval options
- Melodic interval playback for lower levels
- Melodic plus harmonic interval playback for higher levels
- Instant answer checking

## Levels

- Level 1: Major 3rd, Minor 3rd
- Level 2: Major 3rd, Minor 3rd, Perfect 5th
- Level 3: Major 3rd, Minor 3rd, Perfect 4th, Perfect 5th
- Level 4: Major 3rd, Minor 3rd, Perfect 4th, Perfect 5th, Perfect 8ve
- Level 5: Major 3rd, Minor 3rd, Perfect 4th, Perfect 5th, Major 6th, Minor 6th, Perfect 8ve
- Level 6: Major 2nd, Minor 2nd, Major 3rd, Minor 3rd, Perfect 4th, Perfect 5th, Major 6th, Minor 6th, Perfect 8ve
- Level 7: Major 2nd, Minor 2nd, Major 3rd, Minor 3rd, Perfect 4th, Perfect 5th, Major 6th, Minor 6th, Major 7th, Minor 7th, Perfect 8ve
- Level 8: Major 2nd, Minor 2nd, Major 3rd, Minor 3rd, Perfect 4th, Augmented 4th / Diminished 5th, Perfect 5th, Major 6th, Minor 6th, Major 7th, Minor 7th, Perfect 8ve

## Built With

- Python
- Streamlit
- NumPy
- SoundFile

## Local Setup

Install requirements:

```bash
py -m pip install -r requirements.txt
