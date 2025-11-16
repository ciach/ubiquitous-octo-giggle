# Face Recognition (PyTorch)

Real-time face recognition pipeline that detects faces with **MTCNN**, encodes them via **InceptionResnetV1**, and classifies against pre-built embeddings. Includes tooling to build embeddings from a labeled dataset and run live recognition from a webcam.

## Requirements

1. Install Python 3.12 (tested on Linux). Ensure you have CUDA if you plan to use GPU acceleration.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

> Note: `torch`/`torchvision` wheels differ per platform; follow the [official instructions](https://pytorch.org/get-started/locally/) if you need a specific CUDA version.

## Project Structure

```
face-recognition/
├─ build_embeddings.py       # Generates embeddings.pt from dataset/
├─ recognize_camera_torch.py # Runs webcam recognition
├─ dataset/                  # One folder per person (kept empty in git via .gitkeep)
├─ embeddings.pt             # Generated embeddings (ignored by git)
├─ requirements.txt
└─ README.md
```

### Dataset Layout

Add your reference images under `dataset/` with one subfolder per person:

```
dataset/
├─ alice/
│  ├─ img1.jpg
│  └─ img2.png
└─ bob/
   ├─ selfie.jpg
   └─ portrait.png
```

High-quality, front-facing photos with good lighting lead to better embeddings.

## Building Embeddings

Run once (or whenever the dataset changes) to create `embeddings.pt`:

```bash
python build_embeddings.py
```

This script:
- Iterates through each person folder in `dataset/`
- Detects and aligns faces with MTCNN
- Extracts 512-d embeddings with InceptionResnetV1
- Saves tensors + labels to `embeddings.pt`

## Running Real-Time Recognition

Ensure `embeddings.pt` exists, then start the webcam loop:

```bash
python recognize_camera_torch.py
```

What happens:
1. Loads embeddings into memory (normalizes for cosine-like distance)
2. Captures frames from the default webcam
3. Detects faces, pads bounding boxes (controlled via `PADDING`)
4. Computes embedding per detected face and matches via L2 distance
5. Draws green boxes / labels; press `q` to exit

### Tunable Parameters

- `DISTANCE_THRESHOLD` (in `recognize_camera_torch.py`): lower = stricter matches.
- `PADDING`: controls how far the drawn box extends around the face.
- `MTCNN` arguments (`image_size`, `margin`, etc.) if you need different crops.

## Legacy Scripts

Older `face_recognition`-based utilities remain under `old/` for reference but are not part of the primary workflow.


