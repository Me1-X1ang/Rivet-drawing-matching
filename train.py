```python
"""
train.py - ArcFace 训练

特性：
  - 单数据集实例，减少内存占用
  - 支持断点恢复
  - 分层划分 train / val
  - 支持 ResNet50 / EfficientNet-B3
"""

import sys
import json
import argparse
import numpy as np

from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision import transforms
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    efficientnet_b3,
    EfficientNet_B3_Weights
)

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from utils import Config, load_labels


class AugmentedDataset(Dataset):

    def __init__(self, augment_dir):
        index_file = Path(augment_dir) / "augment_index.json"

        if index_file.exists():
            with open(index_file, "r", encoding="utf-8") as f:
                self.samples = json.load(f)["samples"]

        else:
            labels = load_labels()

            self.samples = []

            for png_path in Path(augment_dir).glob("*_aug*.png"):

                drawing_id = png_path.stem.split("_aug")[0]

                class_idx = labels.get(drawing_id, -1)

                if class_idx >= 0:
                    self.samples.append({
                        "path": str(png_path),
                        "class_idx": class_idx
                    })

        print(f"[数据集] {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)


class TransformWrapper(Dataset):

    def __init__(self, base_dataset, indices, transform):
        self.base = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        sample = self.base.samples[self.indices[idx]]

        img = Image.open(sample["path"]).convert("RGB")

        img = self.transform(img)

        return img, sample["class_idx"]


class ArcFaceHead(nn.Module):

    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super().__init__()

        self.s = s
        self.m = m

        self.weight = nn.Parameter(
            torch.FloatTensor(out_features, in_features)
        )

        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):

        cosine = F.linear(
            embeddings,
            F.normalize(self.weight)
        )

        theta = torch.acos(
            torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        )

        one_hot = torch.zeros_like(cosine)

        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        output = (
            one_hot * torch.cos(theta + self.m) +
            (1 - one_hot) * cosine
        )

        return output * self.s


class ArcFaceModel(nn.Module):

    def __init__(
        self,
        num_classes,
        embedding_dim=512,
        s=30.0,
        m=0.5,
        backbone_name='resnet50'
    ):
        super().__init__()

        if backbone_name == 'resnet50':

            backbone = resnet50(
                weights=ResNet50_Weights.DEFAULT
            )

            in_features = backbone.fc.in_features

            backbone.fc = nn.Identity()

        elif backbone_name == 'efficientnet':

            backbone = efficientnet_b3(
                weights=EfficientNet_B3_Weights.DEFAULT
            )

            in_features = backbone.classifier[1].in_features

            backbone.classifier = nn.Identity()

        else:
            raise ValueError(backbone_name)

        self.backbone = backbone

        self.embedding = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        self.arcface = ArcFaceHead(
            embedding_dim,
            num_classes,
            s,
            m
        )

    def forward(self, x, labels=None):

        x = self.backbone(x)

        x = self.embedding(x)

        x = F.normalize(x, p=2, dim=1)

        if labels is not None:
            return self.arcface(x, labels)

        return x


@torch.no_grad()
def evaluate(model, dataloader, device):

    model.eval()

    correct = 0
    total = 0

    for imgs, labels in tqdm(
        dataloader,
        desc="验证",
        leave=False
    ):
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs, labels)

        correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()

        total += labels.size(0)

    model.train()

    return correct / total if total > 0 else 0.0


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    val_acc,
    num_classes,
    backbone
):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_acc': val_acc,
        'num_classes': num_classes,
        'embedding_dim': Config.EMBEDDING_DIM,
        'backbone': backbone
    }, path)


def train(args):

    device = torch.device(Config.DEVICE)

    use_pin_memory = device.type == 'cuda'

    labels_map = load_labels()

    num_classes = len(labels_map)

    print(f"[训练] device={device}")
    print(f"[训练] classes={num_classes}")

    base_dataset = AugmentedDataset(
        Config.AUGMENTED_DIR
    )

    if len(base_dataset) == 0:
        raise RuntimeError("增强数据为空")

    class_indices = defaultdict(list)

    for i, sample in enumerate(base_dataset.samples):
        class_indices[sample["class_idx"]].append(i)

    train_indices = []
    val_indices = []

    rng = np.random.default_rng(42)

    for indices in class_indices.values():

        indices = np.array(indices)

        rng.shuffle(indices)

        split = int(len(indices) * 0.85)

        train_indices.extend(indices[:split].tolist())
        val_indices.extend(indices[split:].tolist())

    train_transform = transforms.Compose([
        transforms.Resize((Config.AUGMENT_SIZE, Config.AUGMENT_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((Config.AUGMENT_SIZE, Config.AUGMENT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = TransformWrapper(
        base_dataset,
        train_indices,
        train_transform
    )

    val_dataset = TransformWrapper(
        base_dataset,
        val_indices,
        val_transform
    )

    print(f"[训练] train={len(train_dataset)}")
    print(f"[训练] val={len(val_dataset)}")

    loader_args = dict(
        batch_size=args.batch,
        num_workers=args.workers,
        pin_memory=use_pin_memory,
        persistent_workers=args.workers > 0
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **loader_args
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_args
    )

    model = ArcFaceModel(
        num_classes=num_classes,
        embedding_dim=Config.EMBEDDING_DIM,
        s=args.arcface_s,
        m=args.arcface_m,
        backbone_name=args.backbone
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )

    criterion = nn.CrossEntropyLoss()

    start_epoch = 0

    if args.resume:

        checkpoint = torch.load(
            args.resume,
            map_location=device
        )

        model.load_state_dict(
            checkpoint['model_state_dict']
        )

        optimizer.load_state_dict(
            checkpoint['optimizer_state_dict']
        )

        scheduler.load_state_dict(
            checkpoint['scheduler_state_dict']
        )

        start_epoch = checkpoint.get('epoch', 0) + 1

        print(f"[训练] resume epoch={start_epoch}")

    Config.MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    log_file = Config.MODEL_DIR / "training_log.csv"

    if not log_file.exists():
        with open(log_file, "w") as f:
            f.write("epoch,train_loss,val_acc,lr\n")

    best_acc = 0.0

    best_model_path = Config.MODEL_DIR / "best.pth"

    for epoch in range(start_epoch, args.epochs):

        model.train()

        total_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}"
        )

        for imgs, labels in pbar:

            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(imgs, labels)

            loss = criterion(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=2.0
            )

            optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}'
            })

        scheduler.step()

        avg_loss = total_loss / len(train_loader)

        val_acc = evaluate(
            model,
            val_loader,
            device
        )

        current_lr = scheduler.get_last_lr()[0]

        print(
            f"[Epoch {epoch+1}] "
            f"loss={avg_loss:.4f} "
            f"acc={val_acc:.4f}"
        )

        with open(log_file, "a") as f:
            f.write(
                f"{epoch+1},"
                f"{avg_loss:.4f},"
                f"{val_acc:.4f},"
                f"{current_lr:.2e}\n"
            )

        if val_acc > best_acc:

            best_acc = val_acc

            save_checkpoint(
                best_model_path,
                model,
                optimizer,
                scheduler,
                epoch,
                val_acc,
                num_classes,
                args.backbone
            )

            print(f"[保存] best={val_acc:.4f}")

        if (epoch + 1) % 20 == 0:

            save_checkpoint(
                Config.MODEL_DIR /
                f"checkpoint_epoch{epoch+1}.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                val_acc,
                num_classes,
                args.backbone
            )

    print(f"[训练] best_acc={best_acc:.4f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="ArcFace 训练"
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=Config.NUM_EPOCHS
    )

    parser.add_argument(
        '--batch',
        type=int,
        default=Config.BATCH_SIZE
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=Config.LEARNING_RATE
    )

    parser.add_argument(
        '--arcface-s',
        type=float,
        default=30.0
    )

    parser.add_argument(
        '--arcface-m',
        type=float,
        default=0.5
    )

    parser.add_argument(
        '--backbone',
        type=str,
        default='resnet50',
        choices=['resnet50', 'efficientnet']
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=2
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None
    )

    args = parser.parse_args()

    train(args)
```
