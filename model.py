import cv2
import face_recognition
import pickle
from pathlib import Path
import numpy as np

import time 

DEFAULT_ENCODINGS_PATH = Path("encodings/encodings.pkl")

# Function to load known encodings
def load_known_encodings(encodings_location):
    encodings_location = Path(encodings_location)

    with encodings_location.open(mode='rb') as f:
        known_encodings = pickle.load(f)

    return known_encodings

# Function to compare unknown encoding with known encodings and return the result
def compare_unknown_encoding(unknown_encoding, known_encodings, threshold=0.5):
    best_match = None
    best_distance = float('inf')

    for name, encodings in known_encodings.items():
        for known_encoding in encodings:
            distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]
            if distance < best_distance:
                best_match = name
                best_distance = distance

    if best_distance <= threshold:
        return best_match
    else:
        return "Unknown"


# def preprocess_known_encodings(known_encodings):
#     all_known_encodings = []
#     all_names = []

#     for name, encodings in known_encodings.items():
#         all_known_encodings.extend(encodings)
#         all_names.extend([name] * len(encodings))

#     all_known_encodings = np.array(all_known_encodings)
#     return all_known_encodings, all_names


known_encodings = load_known_encodings(DEFAULT_ENCODINGS_PATH)

# all_known_encodings, all_names = preprocess_known_encodings(known_encodings)


# def compare_unknown_encoding(unknown_encoding, all_known_encodings, all_names, threshold=0.5):
#     best_match = None
#     # best_distance = float('inf')

#     # # Calculate distances in batch
#     # distances = face_recognition.face_distance(all_known_encodings, unknown_encoding)

#     # # Find the best match
#     # best_index = np.argmin(distances)
#     # best_distance = distances[best_index]
#     # best_match = all_names[best_index]

#     # if best_distance <= threshold:
#     #     return best_match
#     # else:
#     #     return "Unknown"

#     decisions = face_recognition.compare_faces(all_known_encodings, unknown_encoding, tolerance=threshold)
#     names_count = {}
#     for n, bool_decision in enumerate(decisions):
#         if bool_decision:
#             name = all_names[n]
#             if name not in names_count:
#                 names_count[name] = 1
#             else:
#                 names_count[name] += 1
    
#     best_id = None
#     best_count = 0

#     if names_count == {}:
#         return "Unknown"
    
#     for name, count in names_count.items():
#         if best_count > count:
#             best_id = name

#     if best_id is None:
#         return "Unknown"
#     else:
#         return best_id

    
                



# Load known encodings

# Start video capture
video_capture = cv2.VideoCapture(0)
start_time = time.time()

while True:

    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)

    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")

    unknown_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, model="small")

    # Loop through each detected face
    for unknown_encoding, (top, right, bottom, left) in zip(unknown_encodings, face_locations):
        # Check if the face belongs to the person we're interested in
        # result = compare_unknown_encoding(unknown_encoding, all_known_encodings, all_names, threshold=0.5)
        result = compare_unknown_encoding(unknown_encoding, known_encodings, threshold=0.5)

        # Scale the face locations back up since the frame was resized
        # top *= 2
        # right *= 2
        # bottom *= 2
        # left *= 2

        # Draw a box around the face
        cv2.rectangle(small_frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # Draw a label with a name below the face
        cv2.putText(small_frame, result, (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)


        # # Draw a box around the face
        # cv2.rectangle(small_frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # # Draw a label with a name below the face
        # cv2.rectangle(small_frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        # font = cv2.FONT_HERSHEY_DUPLEX
        # cv2.putText(small_frame, result, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


    end_time = time.time()
    time_diff = (end_time - start_time) 

    fps = 1.0 / time_diff
    start_time = end_time

    cv2.putText(small_frame, f"FPS: {fps: .2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # Display the resulting image
    cv2.imshow('Video', small_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()


