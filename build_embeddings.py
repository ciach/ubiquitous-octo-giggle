from pathlib import Path
from typing import List, Tuple

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image


DATASET_DIR = Path("dataset")
EMBEDDINGS_PATH = Path("embeddings.pt")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("[INFO] Using CUDA")
        return torch.device("cuda")
    print("[INFO] Using CPU")
    return torch.device("cpu")


def build_embeddings(
    dataset_dir: Path,
    device: torch.device,
) -> Tuple[torch.Tensor, List[str]]:
    mtcnn = MTCNN(
        image_size=160,
        margin=20,
        keep_all=False,
        device=device,
    )

    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    all_embeddings: List[torch.Tensor] = []
    all_labels: List[str] = []

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    for person_dir in sorted(dataset_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        label = person_dir.name
        print(f"[INFO] Processing person: {label}")

        image_paths = sorted(
            p
            for p in person_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        if not image_paths:
            print(f"  [WARN] No images for {label}, skipping.")
            continue

        for img_path in image_paths:
            print(f"  [IMG] {img_path}")
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"    [WARN] Failed to open {img_path}: {e}")
                continue

            # Detect & align face
            face_tensor = mtcnn(img)

            if face_tensor is None:
                print("    [WARN] No face detected, skipping.")
                continue

            # Shape: [3, 160, 160] -> add batch dim
            face_tensor = face_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = resnet(face_tensor).cpu()  # [1, 512]

            all_embeddings.append(embedding)
            all_labels.append(label)

    if not all_embeddings:
        raise RuntimeError("No embeddings created. Check your images & faces.")

    # Concatenate into [N, 512]
    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    return embeddings_tensor, all_labels


def main() -> None:
    device = get_device()
    embeddings, labels = build_embeddings(DATASET_DIR, device)

    data = {
        "embeddings": embeddings,  # torch.Tensor [N, 512]
        "labels": labels,  # List[str]
    }

    torch.save(data, EMBEDDINGS_PATH)
    print(f"[OK] Saved {len(labels)} embeddings to {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
