"""
train.py - ArcFace 度量学习训练脚本（内存优化版）

修复：
  - 数据集只加载一次，用 Subset + 不同 transform wrapper 代替三次实例化
  - pin_memory 根据设备自动决定
  - 修复 args.resume=None 时 checkpoint 未定义导致的 NameError
  - pretrained=True 改为 weights= 新 API
"""

import os
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
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b3, EfficientNet_B3_Weights
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from utils import Config, load_labels


# ============================================================
# 数据集
# ============================================================
class AugmentedDataset(Dataset):
    """
    加载增强后的图纸数据集。
    transform 在 __getitem__ 中按需应用，不预加载图像到内存。
    """

    def __init__(self, augment_dir, transform=None):
        index_file = Path(augment_dir) / "augment_index.json"

        if not index_file.exists():
            print("[数据集] 未找到 augment_index.json，直接扫描 png 文件...")
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
        else:
            with open(index_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.samples = data["samples"]

        self.transform = transform
        print(f"[数据集] 共加载 {len(self.samples)} 张增强图像（元数据）")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["path"]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, sample["class_idx"]


class TransformWrapper(Dataset):
    """
    ★ 核心修复：让同一份 samples 列表支持不同 transform，
    避免为 train/val 各实例化一次完整数据集。
    """

    def __init__(self, base_dataset, indices, transform):
        self.base = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        sample = self.base.samples[real_idx]
        img = Image.open(sample["path"]).convert('RGB')
        img = self.transform(img)
        return img, sample["class_idx"]


# ============================================================
# ArcFace 模型
# ============================================================
class ArcFaceModel(nn.Module):
    """ResNet-50 / EfficientNet-B3 + 嵌入层 + ArcFace 分类头"""

    def __init__(self, num_classes, embedding_dim=512, s=30.0, m=0.5,
                 backbone_name='resnet50'):
        super().__init__()

        if backbone_name == 'resnet50':
            backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif backbone_name == 'efficientnet':
            backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"未知 backbone: {backbone_name}")

        self.backbone = backbone
        self.embedding = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        self.arcface = ArcFaceHead(embedding_dim, num_classes, s, m)

    def forward(self, x, labels=None):
        x = self.backbone(x)
        x = self.embedding(x)
        x_norm = F.normalize(x, p=2, dim=1)
        if labels is not None:
            return self.arcface(x_norm, labels)
        return x_norm


class ArcFaceHead(nn.Module):
    """Additive Angular Margin Loss 分类头"""

    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        cosine = F.linear(embeddings, F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = one_hot * torch.cos(theta + self.m) + (1 - one_hot) * cosine
        return output * self.s


# ============================================================
# 评估
# ============================================================
@torch.no_grad()
def evaluate(model, dataloader, device):
    """计算验证集准确率，结束后恢复 train 模式"""
    model.eval()
    correct = total = 0
    for imgs, labels in tqdm(dataloader, desc="验证", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs, labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    model.train()
    return correct / total if total > 0 else 0.0


# ============================================================
# 训练主循环
# ============================================================
def train(args):
    device = torch.device(Config.DEVICE)
    # pin_memory 仅在真正用 CUDA 时开启，CPU 训练开启反而浪费内存
    use_pin_memory = device.type == 'cuda'
    print(f"[训练] 设备: {device}  pin_memory: {use_pin_memory}")

    labels_map = load_labels()
    num_classes = len(labels_map)
    print(f"[训练] 类别数: {num_classes}")

    # ★ 数据集只实例化一次（只读 samples 元数据，不加载图像）
    base_dataset = AugmentedDataset(Config.AUGMENTED_DIR, transform=None)

    if len(base_dataset) == 0:
        raise RuntimeError(
            f"增强数据集为空！请先运行 augment.py。\n预期数据在: {Config.AUGMENTED_DIR}"
        )

    # 按类别分层划分 train / val 索引
    class_indices = defaultdict(list)
    for i, s in enumerate(base_dataset.samples):
        class_indices[s["class_idx"]].append(i)

    train_indices, val_indices = [], []
    rng = np.random.default_rng(42)
    for indices in class_indices.values():
        indices = np.array(indices)
        rng.shuffle(indices)
        split = int(len(indices) * 0.85)
        train_indices.extend(indices[:split].tolist())
        val_indices.extend(indices[split:].tolist())

    train_transform = transforms.Compose([
        transforms.Resize((Config.AUGMENT_SIZE, Config.AUGMENT_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((Config.AUGMENT_SIZE, Config.AUGMENT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ★ TransformWrapper：复用同一份 samples，只换 transform，零额外内存
    train_dataset = TransformWrapper(base_dataset, train_indices, train_transform)
    val_dataset   = TransformWrapper(base_dataset, val_indices,   val_transform)

    print(f"[训练] 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=16,
        shuffle=True, num_workers=2,
        pin_memory=use_pin_memory, drop_last=True,
        persistent_workers=True   # 避免每 epoch 重建 worker 进程
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16,
        shuffle=False, num_workers=2,
        pin_memory=use_pin_memory,
        persistent_workers=True
    )

    # 模型
    model = ArcFaceModel(
        num_classes=num_classes,
        embedding_dim=Config.EMBEDDING_DIM,
        s=args.arcface_s,
        m=args.arcface_m,
        backbone_name=args.backbone
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=60, eta_min=args.lr * 0.01)
    criterion = nn.CrossEntropyLoss()

    # ★ 修复：resume 和 start_epoch 统一处理，不再有未定义变量风险
    start_epoch = 0
    if args.resume:
        print(f"[训练] 从断点恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"[训练] 从 epoch {start_epoch} 继续")

    # 日志
    Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log_file = Config.MODEL_DIR / "training_log.csv"
    if not log_file.exists():
        with open(log_file, "w") as f:
            f.write("epoch,train_loss,val_acc,lr\n")

    best_acc = 0.0
    best_model_path = Config.MODEL_DIR / "best.pth"

    def save_checkpoint(path, epoch, val_acc):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'num_classes': num_classes,
            'embedding_dim': Config.EMBEDDING_DIM,
            'backbone': args.backbone
        }, path)

    print(f"\n{'='*50}")
    print(f"开始训练: {60} epochs, lr={args.lr}, batch={16}")
    print(f"{'='*50}\n")

    for epoch in range(start_epoch, 60):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/60")

        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs, labels), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, device)
        current_lr = scheduler.get_last_lr()[0]

        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}, lr={current_lr:.2e}")
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{avg_loss:.4f},{val_acc:.4f},{current_lr:.2e}\n")

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(best_model_path, epoch, val_acc)
            print(f"  ✅ 保存最佳模型 (val_acc={val_acc:.4f})")

        if (epoch + 1) % 20 == 0:
            save_checkpoint(
                Config.MODEL_DIR / f"checkpoint_epoch{epoch+1}.pth",
                epoch, val_acc
            )

    print(f"\n{'='*50}")
    print(f"训练完成！最佳 val_acc: {best_acc:.4f}")
    print(f"模型保存在: {best_model_path}")
    print(f"训练日志: {log_file}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ArcFace 度量学习训练")
    parser.add_argument('--epochs',     type=int,   default=Config.NUM_EPOCHS)
    parser.add_argument('--batch',      type=int,   default=Config.BATCH_SIZE)
    parser.add_argument('--lr',         type=float, default=Config.LEARNING_RATE)
    parser.add_argument('--arcface-s',  type=float, default=30.0)
    parser.add_argument('--arcface-m',  type=float, default=0.5)
    parser.add_argument('--backbone',   type=str,   default='resnet50',
                        choices=['resnet50', 'efficientnet'])
    parser.add_argument('--workers',    type=int,   default=2)
    parser.add_argument('--resume',     type=str,   default=None)
    args = parser.parse_args()
    train(args)