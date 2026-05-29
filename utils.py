```python
"""
utils.py - RDM 公共工具
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms


# 配置

class Config:

    PROJECT_DIR = Path(__file__).resolve().parent

    DATA_DIR = PROJECT_DIR / "data"
    LABEL_DIR = PROJECT_DIR / "labels"
    AUGMENTED_DIR = PROJECT_DIR / "augmented"
    MODEL_DIR = PROJECT_DIR / "models"
    INDEX_DIR = PROJECT_DIR / "indexes"

    IMAGE_SIZE = 518
    AUGMENT_SIZE = 256

    DINOV2_MODEL = "dinov2_vitl14"

    EMBEDDING_DIM = 512

    NUM_EPOCHS = 60
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4

    TOP_K = 5

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    @classmethod
    def ensure_dirs(cls):

        for path in [
            cls.LABEL_DIR,
            cls.AUGMENTED_DIR,
            cls.MODEL_DIR,
            cls.INDEX_DIR
        ]:
            path.mkdir(
                parents=True,
                exist_ok=True
            )


Config.ensure_dirs()


# 标签

def build_labels(
    data_dir=None,
    label_dir=None
):

    data_dir = Path(
        data_dir or Config.DATA_DIR
    )

    label_dir = Path(
        label_dir or Config.LABEL_DIR
    )

    png_files = sorted(
        data_dir.glob("*.png")
    )

    if not png_files:
        raise FileNotFoundError(
            "未找到 PNG 图纸"
        )

    mapping = {
        path.stem: idx
        for idx, path in enumerate(png_files)
    }

    with open(
        label_dir / "labels.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            mapping,
            f,
            ensure_ascii=False,
            indent=2
        )

    return mapping


def load_labels():

    label_file = (
        Config.LABEL_DIR /
        "labels.json"
    )

    if not label_file.exists():
        return build_labels()

    with open(
        label_file,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# DINOv2

class DINOv2Extractor:

    def __init__(
        self,
        model_name=None,
        device=None
    ):

        self.device = (
            device or Config.DEVICE
        )

        self.model_name = (
            model_name or
            Config.DINOV2_MODEL
        )

        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            self.model_name
        )

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((
                Config.IMAGE_SIZE,
                Config.IMAGE_SIZE
            )),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_image(self, image_or_path):

        if isinstance(
            image_or_path,
            (str, Path)
        ):
            return Image.open(
                image_or_path
            ).convert("RGB")

        if isinstance(
            image_or_path,
            np.ndarray
        ):
            return Image.fromarray(
                image_or_path
            ).convert("RGB")

        return image_or_path.convert("RGB")

    @torch.no_grad()
    def extract(self, image_or_path):

        img = self._load_image(
            image_or_path
        )

        tensor = (
            self.transform(img)
            .unsqueeze(0)
            .to(self.device)
        )

        feat = self.model(tensor)

        return (
            feat.cpu()
            .numpy()
            .squeeze()
            .astype("float32")
        )

    def extract_batch(self, image_paths):

        features = [
            self.extract(path)
            for path in image_paths
        ]

        return np.stack(features)


# ArcFace


class ArcFaceExtractor:

    def __init__(
        self,
        model_path,
        device=None
    ):

        self.device = (
            device or Config.DEVICE
        )

        from torchvision.models import (
            resnet50
        )

        self.backbone = resnet50(
            weights=None
        )

        in_features = (
            self.backbone.fc.in_features
        )

        self.backbone.fc = (
            torch.nn.Identity()
        )

        self.embedding = torch.nn.Linear(
            in_features,
            Config.EMBEDDING_DIM
        )

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        self.backbone.load_state_dict({
            k.replace("backbone.", ""): v
            for k, v in checkpoint.items()
            if k.startswith("backbone.")
        })

        self.embedding.load_state_dict({
            k.replace("embedding.", ""): v
            for k, v in checkpoint.items()
            if k.startswith("embedding.")
        })

        self.backbone.to(
            self.device
        ).eval()

        self.embedding.to(
            self.device
        ).eval()

        self.transform = transforms.Compose([
            transforms.Resize((
                Config.AUGMENT_SIZE,
                Config.AUGMENT_SIZE
            )),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_image(self, image_or_path):

        if isinstance(
            image_or_path,
            (str, Path)
        ):
            return Image.open(
                image_or_path
            ).convert("RGB")

        if isinstance(
            image_or_path,
            np.ndarray
        ):
            return Image.fromarray(
                image_or_path
            ).convert("RGB")

        return image_or_path.convert("RGB")

    @torch.no_grad()
    def extract(self, image_or_path):

        img = self._load_image(
            image_or_path
        )

        tensor = (
            self.transform(img)
            .unsqueeze(0)
            .to(self.device)
        )

        x = self.backbone(tensor)
        x = self.embedding(x)

        x = F.normalize(
            x,
            p=2,
            dim=1
        )

        return (
            x.cpu()
            .numpy()
            .squeeze()
            .astype("float32")
        )

    def extract_batch(self, image_paths):

        features = [
            self.extract(path)
            for path in image_paths
        ]

        return np.stack(features)


# FAISS

try:

    import faiss

    HAS_FAISS = True

except ImportError:

    HAS_FAISS = False


class VectorIndex:

    def __init__(
        self,
        features=None,
        labels=None,
        use_faiss=True
    ):

        self.use_faiss = (
            use_faiss and HAS_FAISS
        )

        self.features = None
        self.labels = None
        self.index = None

        if features is not None:
            self.build(
                features,
                labels
            )

    def build(
        self,
        features,
        labels
    ):

        self.features = features.astype(
            "float32"
        )

        self.labels = np.array(labels)

        if self.use_faiss:

            faiss.normalize_L2(
                self.features
            )

            self.index = faiss.IndexFlatIP(
                self.features.shape[1]
            )

            self.index.add(
                self.features
            )

        else:

            norms = np.linalg.norm(
                self.features,
                axis=1,
                keepdims=True
            )

            self.features = (
                self.features /
                (norms + 1e-8)
            )

    def search(
        self,
        query_feat,
        k=5
    ):

        if query_feat.ndim == 1:
            query_feat = query_feat.reshape(1, -1)

        query_feat = query_feat.astype(
            "float32"
        )

        if self.use_faiss:

            faiss.normalize_L2(
                query_feat
            )

            scores, indices = (
                self.index.search(
                    query_feat,
                    k
                )
            )

            return [
                (
                    self.labels[idx],
                    float(scores[0][i])
                )
                for i, idx in enumerate(indices[0])
            ]

        norms = np.linalg.norm(
            query_feat,
            axis=1,
            keepdims=True
        )

        query_feat = (
            query_feat /
            (norms + 1e-8)
        )

        sims = np.dot(
            query_feat,
            self.features.T
        ).squeeze()

        top_k = np.argsort(sims)[::-1][:k]

        return [
            (
                self.labels[i],
                float(sims[i])
            )
            for i in top_k
        ]

    def save(self, path):

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        np.savez(
            path,
            features=self.features,
            labels=self.labels
        )

    @classmethod
    def load(cls, path):

        data = np.load(
            f"{path}.npz",
            allow_pickle=True
        )

        return cls(
            data["features"],
            data["labels"]
        )


# 图纸库

class DrawingGallery:

    def __init__(
        self,
        use_dinov2=True,
        arcface_model_path=None
    ):

        self.data_dir = Config.DATA_DIR

        self.labels = load_labels()

        self.label_list = sorted(
            self.labels.keys(),
            key=lambda x: self.labels[x]
        )

        if use_dinov2:

            self.extractor = (
                DINOv2Extractor()
            )

            self.index_path = (
                Config.INDEX_DIR /
                "dinov2_index"
            )

        else:

            self.extractor = (
                ArcFaceExtractor(
                    arcface_model_path
                )
            )

            self.index_path = (
                Config.INDEX_DIR /
                "arcface_index"
            )

        self.index = None

    def build_index(
        self,
        force_rebuild=False
    ):

        index_file = Path(
            f"{self.index_path}.npz"
        )

        if (
            index_file.exists() and
            not force_rebuild
        ):

            self.index = (
                VectorIndex.load(
                    self.index_path
                )
            )

            return

        image_paths = [
            self.data_dir / f"{did}.png"
            for did in self.label_list
        ]

        features = (
            self.extractor.extract_batch(
                image_paths
            )
        )

        labels = list(
            range(len(image_paths))
        )

        self.index = VectorIndex(
            features,
            labels
        )

        self.index.save(
            self.index_path
        )

    def query(
        self,
        image_path,
        k=None
    ):

        if self.index is None:
            self.build_index()

        k = k or Config.TOP_K

        feat = self.extractor.extract(
            image_path
        )

        results = self.index.search(
            feat,
            k
        )

        output = []

        for label_idx, score in results:

            drawing_id = (
                self.label_list[label_idx]
            )

            output.append({
                "drawing_id": drawing_id,
                "similarity": round(
                    float(score),
                    4
                ),
                "drawing_path": str(
                    self.data_dir /
                    f"{drawing_id}.png"
                )
            })

        return output
```
