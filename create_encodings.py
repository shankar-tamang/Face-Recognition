

from pathlib import Path
from PIL import Image, ImageDraw
import face_recognition
import pickle

DEFAULT_ENCODINGS_PATH = Path("encodings/encodings.pkl")



def encode_known_faces(model="hog", encodings_location=DEFAULT_ENCODINGS_PATH):
    comparison_details = {}

    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        # Store face encodings as a single list for each person
        if name not in comparison_details:
            comparison_details[name] = face_encodings
        else:
            comparison_details[name].extend(face_encodings)

    with encodings_location.open(mode="wb") as f:
        pickle.dump(comparison_details, f)


encode_known_faces()