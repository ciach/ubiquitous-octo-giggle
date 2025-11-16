from pathlib import Path
import pickle
from typing import List, Tuple

import face_recognition
import numpy as np


DATASET_DIR = Path("dataset")
ENCODINGS_PATH = Path("encodings.pkl")


def load_images_and_labels(dataset_dir: Path) -> Tuple[List[np.ndarray], List[str]]:
    encodings: List[np.ndarray] = []
    labels: List[str] = []

    for person_dir in dataset_dir.iterdir():
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name
        print(f"[INFO] Processing person: {person_name}")

        for img_path in person_dir.glob("*.*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            print(f"  [IMG] {img_path}")

            # Load image
            image = face_recognition.load_image_file(str(img_path))

            # Detect face locations
            face_locations = face_recognition.face_locations(image)

            if len(face_locations) == 0:
                print("    [WARN] No face found, skipping.")
                continue

            # Use the first detected face (for family photos usually fine)
            face_encoding = face_recognition.face_encodings(image, face_locations)[0]
            encodings.append(face_encoding)
            labels.append(person_name)

    return encodings, labels


def main() -> None:
    if not DATASET_DIR.exists():
        print(f"[ERROR] Dataset directory '{DATASET_DIR}' does not exist.")
        return

    encodings, labels = load_images_and_labels(DATASET_DIR)

    if not encodings:
        print("[ERROR] No encodings created. Check your dataset images.")
        return

    data = {
        "encodings": encodings,
        "labels": labels,
    }

    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(data, f)

    print(f"[OK] Saved {len(encodings)} encodings to {ENCODINGS_PATH}")


if __name__ == "__main__":
    main()
