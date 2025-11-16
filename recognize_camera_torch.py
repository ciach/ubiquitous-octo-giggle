from typing import List, Tuple

import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from pathlib import Path


EMBEDDINGS_PATH = Path("embeddings.pt")

# Tune these later if needed
DISTANCE_THRESHOLD = 0.9  # lower = stricter, higher = more forgiving
PADDING = 40  # increase if you want the box even farther away


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("[INFO] Using CUDA")
        return torch.device("cuda")
    print("[INFO] Using CPU")
    return torch.device("cpu")


def load_reference_embeddings(
    path: Path,
) -> Tuple[torch.Tensor, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    data = torch.load(path, map_location="cpu")
    embeddings: torch.Tensor = data["embeddings"]
    labels: List[str] = data["labels"]

    print(f"[INFO] Loaded {len(labels)} reference embeddings.")
    return embeddings, labels


def main() -> None:
    device = get_device()

    known_embeddings, known_labels = load_reference_embeddings(EMBEDDINGS_PATH)

    # Normalize embeddings for cosine-ish distance if you want
    known_embeddings = torch.nn.functional.normalize(known_embeddings, p=2, dim=1)

    mtcnn = MTCNN(
        image_size=160,
        margin=20,
        keep_all=True,
        device=device,
    )

    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame from camera.")
            break

        # Convert BGR -> RGB and wrap as PIL image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        # This returns aligned face tensors [N, 3, 160, 160]
        faces = mtcnn(pil_img)

        names: List[str] = []
        boxes_xyxy: List[Tuple[int, int, int, int]] = []

        if faces is not None:
            if faces.ndim == 3:
                faces = faces.unsqueeze(0)

            faces = faces.to(device)

            with torch.no_grad():
                embeddings = resnet(faces)  # [N, 512]

            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu()

            for emb in embeddings:
                # Compute L2 distance to all known embeddings
                dists = torch.norm(known_embeddings - emb, dim=1)
                min_dist, idx = torch.min(dists, dim=0)

                if min_dist.item() < DISTANCE_THRESHOLD:
                    name = known_labels[idx.item()]
                else:
                    name = "Unknown"

                names.append(name)

            # We need boxes. Easiest: ask MTCNN for detection boxes.
            # Re-run lightweight detect to get bbox coords.
            boxes, probs = mtcnn.detect(pil_img)
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.astype(int)

                    # Expand box by padding
                    x1 -= PADDING
                    y1 -= PADDING
                    x2 += PADDING
                    y2 += PADDING

                    # Clamp to frame boundaries
                    x1 = max(x1, 0)
                    y1 = max(y1, 0)
                    x2 = min(x2, frame.shape[1])
                    y2 = min(y2, frame.shape[0])

                    boxes_xyxy.append((x1, y1, x2, y2))

        # Draw results
        for (x1, y1, x2, y2), name in zip(boxes_xyxy, names):
            # Rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label background
            cv2.rectangle(
                frame,
                (x1, y2 - 40),
                (x2, y2),
                (0, 255, 0),
                cv2.FILLED,
            )

            cv2.putText(
                frame,
                name,
                (x1 + 5, y2 - 7),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                (0, 0, 0),
                1,
            )

        cv2.imshow("Family Face Recognition (PyTorch)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
