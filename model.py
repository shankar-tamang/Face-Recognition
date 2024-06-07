import cv2
import face_recognition
import pickle
from pathlib import Path
import dlib

DEFAULT_ENCODINGS_PATH = Path("encodings/encodings.pkl")

# Function to load known encodings
def load_known_encodings(encodings_location):
    encodings_location = Path(encodings_location)

    with encodings_location.open(mode='rb') as f:
        known_encodings = pickle.load(f)

    return known_encodings

# Function to compare unknown encoding with known encodings and return the result
def compare_unknown_encoding(unknown_encoding, known_encodings, threshold=0.6):
    best_match = None
    best_distance = float('inf')

    for name, encodings in known_encodings.items():
        if name == "person_to_check":
            for known_encoding in encodings:
                distance = face_recognition.face_distance([known_encoding], unknown_encoding)
                if distance < best_distance:
                    best_match = name
                    best_distance = distance

    if best_distance <= threshold:
        return best_match
    else:
        return "Unknown"

# Load known encodings
known_encodings = load_known_encodings(DEFAULT_ENCODINGS_PATH)

# Start video capture
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")

    unknown_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Loop through each detected face
    for unknown_encoding, (top, right, bottom, left) in zip(unknown_encodings, face_locations):
        # Check if the face belongs to the person we're interested in
        result = compare_unknown_encoding([unknown_encoding], known_encodings)

        # Scale the face locations back up since the frame was resized
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, result, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
