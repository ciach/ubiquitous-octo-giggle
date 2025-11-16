import pickle
from pathlib import Path
from typing import List

import cv2
import face_recognition
import numpy as np


ENCODINGS_PATH = Path("encodings.pkl")
TOLERANCE = 0.5  # lower = stricter (0.6 is common, 0.5 is stricter)


def load_encodings(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Encodings file not found: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    encodings: List[np.ndarray] = data["encodings"]
    labels: List[str] = data["labels"]

    print(f"[INFO] Loaded {len(encodings)} known faces.")
    return encodings, labels


def main() -> None:
    known_encodings, known_labels = load_encodings(ENCODINGS_PATH)

    # Open default camera (0). Change index if you have multiple cameras.
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("[WARN] Failed to grab frame.")
            break

        # Resize frame to speed up processing (optional)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert BGR (OpenCV) to RGB (face_recognition)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )

        face_names: List[str] = []

        for face_encoding in face_encodings:
            # Compare with known encodings
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(distances) == 0:
                face_names.append("Unknown")
                continue

            best_match_index = np.argmin(distances)
            if distances[best_match_index] < TOLERANCE:
                name = known_labels[best_match_index]
            else:
                name = "Unknown"

            face_names.append(name)

        # Draw boxes & labels
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up since we resized to 1/4
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Label background
            cv2.rectangle(
                frame,
                (left, bottom - 25),
                (right, bottom),
                (0, 255, 0),
                cv2.FILLED,
            )

            # Put label text
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                frame,
                name,
                (left + 6, bottom - 6),
                font,
                0.6,
                (0, 0, 0),
                1,
            )

        cv2.imshow("Family Face Recognition", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
