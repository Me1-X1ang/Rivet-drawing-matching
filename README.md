# RDM - Rivet Drawing Matcher

RDM is an image retrieval project for matching a damaged rivet or part photo back to the most likely engineering drawing in a local drawing library.

The project supports two retrieval paths:

- **DINOv2 zero-shot retrieval**: works without training. It extracts embeddings from each drawing and the query photo, then searches by cosine similarity.
- **ArcFace retrieval**: uses augmented drawing images to train a metric-learning model, then searches with trained embeddings.

This repository intentionally does **not** include datasets, generated images, trained models, or cached indexes. Those files are local artifacts and are ignored by Git.

## What Is Included

```text
RDM/
|-- README.md              Project documentation
|-- requirements.txt       Python dependencies
|-- utils.py               Shared config, labels, feature extraction, vector index
|-- inference.py           Main search/inference entry point
|-- augment.py             Synthetic image augmentation for training
|-- train.py               ArcFace training script
|-- recrop.py              Drawing subject extraction / recropping utility
|-- run.bat                Windows menu launcher
`-- .gitignore             Keeps datasets and artifacts out of Git
```

## Local Folders Not Committed

The following folders are expected to exist locally when you run the project, but they are ignored and should not be uploaded to GitHub:

```text
data/          Original drawing PNG files
labels/        Generated label mapping files
augmented/     Synthetic training images from augment.py
models/        DINOv2 / ArcFace indexes, checkpoints, logs, trained weights
test/          Local test/query images
__pycache__/   Python bytecode cache
```

Large model files such as `*.pt`, `*.pth`, `*.onnx`, `*.npz`, `*.h5`, and training outputs such as `runs/` are also ignored.

## Requirements

- Windows is the assumed environment for the current path defaults.
- Python 3.10+ is recommended.
- A CUDA-capable GPU is recommended for ArcFace training, but DINOv2 retrieval can run on CPU.
- First-time DINOv2 usage downloads model weights through `torch.hub`.

Install dependencies:

```bash
pip install -r requirements.txt
```

Equivalent manual install:

```bash
pip install torch torchvision opencv-python pillow numpy tqdm faiss-cpu
```

If `faiss-cpu` is unavailable on your machine, the project falls back to a NumPy similarity search.

## Important Path Configuration

The current project path is hardcoded in [`utils.py`](utils.py):

```python
PROJECT_DIR = Path(r"D:\Project\RDM")
```

If you clone or move the project to another directory, update `Config.PROJECT_DIR` in `utils.py` before running the scripts.

## Dataset Layout

Put engineering drawing images in:

```text
D:\Project\RDM\data\
```

Expected file format:

```text
data/
|-- D000001.png
|-- D000002.png
|-- D000003.png
`-- ...
```

The drawing ID is taken from the file name without `.png`. For example, `D000042.png` becomes drawing ID `D000042`.

On first run, the project scans `data/` and generates:

```text
labels/labels.json
labels/labels.csv
```

These files map drawing IDs to numeric class indexes.

## Quick Start: DINOv2 Search

DINOv2 mode is the fastest way to start because it does not require training.

```bash
cd D:\Project\RDM
python inference.py --query path\to\query_photo.jpg
```

Return more candidates:

```bash
python inference.py --query path\to\query_photo.jpg --top_k 10
```

Show matched drawing file paths:

```bash
python inference.py --query path\to\query_photo.jpg --verbose
```

Use interactive mode:

```bash
python inference.py --interactive
```

Rebuild the drawing index after adding, removing, or replacing drawings:

```bash
python inference.py --rebuild
```

The first DINOv2 run will:

1. Scan `data/`.
2. Build label mappings.
3. Download/load the DINOv2 model.
4. Extract drawing embeddings.
5. Save the index under `models/`.
6. Search the query image and print the top matches.

## ArcFace Workflow

ArcFace mode is intended for cases where zero-shot DINOv2 retrieval is not accurate enough and you want to train a domain-specific embedding model.

### 1. Generate Augmented Images

```bash
python augment.py --num 50
```

Useful options:

```bash
python augment.py --num 50 --workers 2
python augment.py --num 20 --limit 100
python augment.py --data D:\Project\RDM\data --output D:\Project\RDM\augmented
```

`augment.py` creates synthetic variants for each drawing and writes an index file:

```text
augmented/augment_index.json
```

The augmentation pipeline includes geometric changes, blur, noise, texture overlays, rust-like color changes, scratches, brightness changes, random erasing, and background changes.

### 2. Train ArcFace

```bash
python train.py
```

The training script uses augmented images from `augmented/` and saves outputs under `models/`, including:

```text
models/best.pth
models/checkpoint_epoch20.pth
models/checkpoint_epoch40.pth
models/checkpoint_epoch60.pth
models/training_log.csv
```

Supported backbone choices:

```bash
python train.py --backbone resnet50
python train.py --backbone efficientnet
```

Resume from a checkpoint:

```bash
python train.py --resume models\checkpoint_epoch20.pth
```

Current training note: `train.py` defines CLI arguments for `--epochs`, `--batch`, and `--workers`, but the current implementation still uses fixed internal loop values for epochs, batch size, and workers. If you need to tune those values, update the relevant constants inside `train.py`.

### 3. Search With ArcFace

```bash
python inference.py --mode arcface --model models\best.pth --query path\to\query_photo.jpg
```

Interactive ArcFace mode:

```bash
python inference.py --mode arcface --model models\best.pth --interactive
```

Rebuild the ArcFace index:

```bash
python inference.py --mode arcface --model models\best.pth --rebuild
```

## Recropping Drawings

`recrop.py` extracts the main drawing subject from original drawing images and writes normalized 1024 x 1024 outputs.

Preview a small batch:

```bash
python recrop.py --input D:\Project\RDM\data --output D:\Project\RDM\data_clean --preview 20 --debug
```

Process the full folder:

```bash
python recrop.py --input D:\Project\RDM\data --output D:\Project\RDM\data_clean
```

Tuning options:

```bash
python recrop.py --dilate-ratio 0.08 --dilate-iter 3
```

If the output looks good, copy or move the cleaned PNG files into `data/`, then rebuild the retrieval index:

```bash
python inference.py --rebuild
```

## Windows Launcher

The project includes a simple batch menu:

```bat
run.bat
```

It provides shortcuts for DINOv2 query, interactive search, index rebuild, ArcFace query, augmentation, and training.

## CLI Reference

### `inference.py`

```text
--query, -q          Query image path
--interactive, -i    Run continuous interactive search
--rebuild, -r        Force index rebuild
--mode, -m           Retrieval mode: dinov2 or arcface
--model              ArcFace model path
--top_k, -k          Number of returned matches
--verbose, -v        Print matched drawing paths
```

### `augment.py`

```text
--num                Variants generated per drawing
--workers            Parallel worker process count
--data               Input drawing directory
--output             Output augmented-image directory
--limit              Only process the first N drawings; 0 means all
```

### `train.py`

```text
--lr                 Learning rate
--arcface-s          ArcFace scale
--arcface-m          ArcFace angular margin
--backbone           resnet50 or efficientnet
--resume             Checkpoint path to resume from
```

### `recrop.py`

```text
--input              Input PNG folder
--output             Output folder
--preview            Process only a sample batch
--debug              Save debug images with detected crop boxes
--dilate-ratio       Morphological dilation kernel ratio
--dilate-iter        Dilation iteration count
```

## Typical Workflow

```text
1. Put original drawings into data/
2. Run DINOv2 search:
   python inference.py --query photo.jpg
3. If drawings changed:
   python inference.py --rebuild
4. If DINOv2 accuracy is not enough:
   python augment.py --num 50
   python train.py
   python inference.py --mode arcface --model models\best.pth --query photo.jpg
```

## Troubleshooting

### `data/` has no PNG files

Make sure the original drawing files are placed directly under `data/` and use `.png` extension.

### DINOv2 downloads slowly

The first DINOv2 run downloads model weights through PyTorch Hub. Network speed depends on your GitHub access. After the first successful download, PyTorch caches the model locally.

### `faiss-cpu` cannot be installed

The code can still run with the NumPy fallback. FAISS is faster, but not mandatory for small and medium drawing libraries.

### ArcFace model does not exist

Run augmentation and training first:

```bash
python augment.py --num 50
python train.py
```

Then use:

```bash
python inference.py --mode arcface --model models\best.pth --query photo.jpg
```

### Results are stale after changing drawings

Rebuild the index:

```bash
python inference.py --rebuild
```

For ArcFace:

```bash
python inference.py --mode arcface --model models\best.pth --rebuild
```

## GitHub Upload Notes

This repository is prepared so only source code and project documentation are uploaded. The dataset and generated artifacts remain local because they are listed in `.gitignore`.

Before pushing to GitHub, verify the staged or tracked files with:

```bash
git status --short --ignored
```

You should see code and documentation as tracked files, while dataset folders appear as ignored.
