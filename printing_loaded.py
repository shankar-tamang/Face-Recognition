from pathlib import Path
import pickle


DEFAULT_ENCODINGS_PATH = Path("encodings/encodings.pkl")


def load_known_encodings(encodings_location):
    encodings_location = Path(encodings_location)

    with encodings_location.open(mode='rb') as f:
        known_encodings = pickle.load(f)

    return known_encodings


names_encodings = load_known_encodings(DEFAULT_ENCODINGS_PATH)

print(names_encodings)