"""
RDM 铆钉图纸检索 - 公共工具模块
包含：路径配置、标签管理、特征提取器、FAISS 索引
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# ============================================================
# 全局配置（你的路径在这里改）
# ============================================================
class Config:
    # 项目根目录
    PROJECT_DIR = Path(r"D:\Project\RDM")

    # 图纸数据
    DATA_DIR = PROJECT_DIR / "data"            # 原始 PNG 图纸
    LABEL_DIR = PROJECT_DIR / "labels"          # 标签文件夹
    AUGMENTED_DIR = PROJECT_DIR / "augmented"   # 增强后数据
    MODEL_DIR = PROJECT_DIR / "models"          # 模型保存

    # 图像参数
    IMAGE_SIZE = 518          # DINOv2 推荐尺寸；ArcFace 训练可用 224/256
    AUGMENT_SIZE = 256        # 训练时增强图像的大小

    # DINOv2 模型名（可选：dinov2_vits14 / dinov2_vitb14 / dinov2_vitl14 / dinov2_vitg14）
    DINOV2_MODEL = "dinov2_vitl14"

    # ArcFace 训练参数
    EMBEDDING_DIM = 512
    NUM_EPOCHS = 60
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4

    # 检索参数
    TOP_K = 5   # 返回前 K 个匹配结果

    # 设备
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def ensure_dirs(cls):
        """确保所有目录存在"""
        for d in [cls.LABEL_DIR, cls.AUGMENTED_DIR, cls.MODEL_DIR]:
            d.mkdir(parents=True, exist_ok=True)

# 确保目录存在
Config.ensure_dirs()


# ============================================================
# 标签管理
# ============================================================
def build_labels(data_dir=None, label_dir=None):
    """
    扫描 data 目录，为每张图纸分配类别 ID
    生成 labels.json 和 labels.csv
    返回: {图纸ID: 类别索引} 的字典
    """
    if data_dir is None:
        data_dir = Config.DATA_DIR
    if label_dir is None:
        label_dir = Config.LABEL_DIR

    data_path = Path(data_dir)
    png_files = sorted(data_path.glob("*.png"))

    if not png_files:
        raise FileNotFoundError(f"在 {data_dir} 中没有找到 PNG 文件！")

    mapping = {}
    for i, f in enumerate(png_files):
        drawing_id = f.stem  # 去掉 .png 后缀，如 "D000001"
        mapping[drawing_id] = i

    # 保存 JSON
    label_json = {
        "description": "铆钉图纸标签映射：图纸ID → 类别索引",
        "total_classes": len(mapping),
        "mapping": mapping
    }
    with open(Path(label_dir) / "labels.json", "w", encoding="utf-8") as f:
        json.dump(label_json, f, ensure_ascii=False, indent=2)

    # 保存 CSV（方便 Excel 打开）
    with open(Path(label_dir) / "labels.csv", "w", encoding="utf-8") as f:
        f.write("drawing_id,class_index\n")
        for did, idx in mapping.items():
            f.write(f"{did},{idx}\n")

    print(f"[标签] 共 {len(mapping)} 个类别，已保存到 {label_dir}")
    return mapping


def load_labels(label_dir=None):
    """从标签文件加载映射。支持三种格式：
      1. labels.json（自动生成的标准格式）
      2. classes.txt + XXX.txt（用户现有的标签格式）
      3. 都没有 → 自动扫描 data 目录生成
    """
    if label_dir is None:
        label_dir = Config.LABEL_DIR

    label_path = Path(label_dir)

    # 1. 优先读 labels.json
    json_file = label_path / "labels.json"
    if json_file.exists():
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[标签] 从 labels.json 加载: {len(data['mapping'])} 个类别")
        return data["mapping"]

    # 2. 读用户现有的 classes.txt + 单个 .txt 标签
    classes_file = label_path / "classes.txt"
    if classes_file.exists():
        mapping = {}
        with open(classes_file, "r", encoding="utf-8") as f:
            drawing_ids = [line.strip() for line in f if line.strip()]

        for did in drawing_ids:
            txt_file = label_path / f"{did}.txt"
            if txt_file.exists():
                try:
                    with open(txt_file, "r", encoding="utf-8") as f:
                        class_idx = int(f.read().strip())
                    mapping[did] = class_idx
                except (ValueError, FileNotFoundError):
                    continue

        if mapping:
            print(f"[标签] 从 classes.txt + txt 文件加载: {len(mapping)} 个类别")

            # 顺便生成 labels.json 方便以后用
            label_json = {
                "description": "从现有标签格式迁移",
                "total_classes": len(mapping),
                "mapping": mapping
            }
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(label_json, f, ensure_ascii=False, indent=2)
            print(f"[标签] 已同步生成 labels.json")

            return mapping

    # 3. 都没有 → 自动生成
    print("[标签] 未找到标签文件，自动扫描 data 目录生成...")
    return build_labels(label_dir=label_dir)


# ============================================================
# DINOv2 特征提取器
# ============================================================
class DINOv2Extractor:
    """DINOv2 零样本特征提取器"""

    def __init__(self, model_name=None, device=None):
        self.device = device or Config.DEVICE
        self.model_name = model_name or Config.DINOV2_MODEL

        print(f"[DINOv2] 加载模型 {self.model_name}（首次运行会下载，约 1GB）...")
        self.model = torch.hub.load('facebookresearch/dinov2', self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # 获取特征维度
        dummy = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            self.feat_dim = self.model(dummy).shape[1]
        print(f"[DINOv2] 特征维度: {self.feat_dim}")

        self.transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract(self, image_or_path):
        """提取单张图像的特征向量"""
        if isinstance(image_or_path, (str, Path)):
            img = Image.open(image_or_path).convert('RGB')
        elif isinstance(image_or_path, np.ndarray):
            img = Image.fromarray(image_or_path).convert('RGB')
        else:
            img = image_or_path.convert('RGB')

        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
        return feat.cpu().numpy().squeeze()

    def extract_batch(self, image_paths, verbose=True):
        """批量提取特征"""
        features = []
        iterator = image_paths
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(image_paths, desc="提取特征")

        for path in iterator:
            feat = self.extract(path)
            features.append(feat)

        return np.stack(features)


# ============================================================
# FAISS / 向量检索
# ============================================================
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("[警告] faiss 未安装，将使用 NumPy 暴力搜索。安装: pip install faiss-cpu")


class VectorIndex:
    """向量检索索引"""

    def __init__(self, features=None, labels=None, use_faiss=True):
        """
        features: [N, D] 特征矩阵
        labels: [N] 对应的标签列表
        """
        self.use_faiss = use_faiss and HAS_FAISS
        self.features = None
        self.labels = None
        self.faiss_index = None

        if features is not None:
            self.build(features, labels)

    def build(self, features, labels):
        """构建索引"""
        self.features = features.astype('float32')
        self.labels = np.array(labels)

        if self.use_faiss:
            # L2 归一化后内积 = 余弦相似度
            faiss.normalize_L2(self.features)

            # 暴力搜索（1000 张图完全够用）
            self.faiss_index = faiss.IndexFlatIP(self.features.shape[1])
            self.faiss_index.add(self.features)
            print(f"[索引] FAISS 索引已构建，包含 {len(labels)} 条记录")
        else:
            # NumPy fallback：预归一化
            norms = np.linalg.norm(self.features, axis=1, keepdims=True)
            self.features = self.features / (norms + 1e-8)
            print(f"[索引] NumPy 暴力搜索已就绪，包含 {len(labels)} 条记录")

    def search(self, query_feat, k=5):
        """
        查询最相似的 k 个结果
        query_feat: [D] 或 [1, D]
        返回: [(label, similarity_score), ...]
        """
        if query_feat.ndim == 1:
            query_feat = query_feat.reshape(1, -1)
        query_feat = query_feat.astype('float32')

        if self.use_faiss:
            faiss.normalize_L2(query_feat)
            scores, indices = self.faiss_index.search(query_feat, k)
            results = []
            for i in range(k):
                idx = indices[0][i]
                results.append((self.labels[idx], float(scores[0][i])))
        else:
            # NumPy 暴力余弦相似度
            norms = np.linalg.norm(query_feat, axis=1, keepdims=True)
            query_feat = query_feat / (norms + 1e-8)
            sims = np.dot(query_feat, self.features.T).squeeze()
            top_k = np.argsort(sims)[::-1][:k]
            results = [(self.labels[i], float(sims[i])) for i in top_k]

        return results

    def save(self, path):
        """保存索引到磁盘"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_path,
                 features=self.features,
                 labels=self.labels)
        print(f"[索引] 已保存到 {save_path}.npz")

    @classmethod
    def load(cls, path):
        """从磁盘加载索引"""
        data = np.load(f"{path}.npz", allow_pickle=True)
        index = cls(data['features'], data['labels'])
        return index


# ============================================================
# 图纸库管理
# ============================================================
class DrawingGallery:
    """图纸库：管理所有图纸的特征索引"""

    def __init__(self, data_dir=None, label_dir=None, model_dir=None,
                 use_dinov2=True, arcface_model_path=None):
        self.data_dir = Path(data_dir or Config.DATA_DIR)
        self.model_dir = Path(model_dir or Config.MODEL_DIR)
        self.labels = load_labels(label_dir)
        self.label_list = sorted(self.labels.keys(), key=lambda k: self.labels[k])

        # 初始化特征提取器
        if use_dinov2:
            self.extractor = DINOv2Extractor()
            self.index_path = self.model_dir / "dinov2_index"
        elif arcface_model_path:
            self.extractor = ArcFaceExtractor(arcface_model_path)
            self.index_path = self.model_dir / "arcface_index"
        else:
            raise ValueError("必须指定 use_dinov2=True 或提供 arcface_model_path")

        self.index = None

    def build_index(self, force_rebuild=False):
        """构建/加载图纸特征索引"""
        idx_file = Path(f"{self.index_path}.npz")

        if idx_file.exists() and not force_rebuild:
            print(f"[图纸库] 从缓存加载索引: {idx_file}")
            self.index = VectorIndex.load(self.index_path)
            return

        print("[图纸库] 正在提取所有图纸特征...")
        image_paths = [self.data_dir / f"{lid}.png" for lid in self.label_list]

        features = self.extractor.extract_batch(image_paths)
        labels = list(range(len(self.label_list)))

        self.index = VectorIndex(features, labels)
        self.index.save(self.index_path)
        print(f"[图纸库] 索引构建完成！共 {len(labels)} 张图纸")

    def query(self, image_path, k=None):
        """查询单张照片对应的图纸"""
        if self.index is None:
            self.build_index()

        k = k or Config.TOP_K
        feat = self.extractor.extract(image_path)
        results = self.index.search(feat, k=k)

        # 转换为可读格式
        output = []
        for label_idx, score in results:
            drawing_id = self.label_list[label_idx]
            drawing_path = self.data_dir / f"{drawing_id}.png"
            output.append({
                "drawing_id": drawing_id,
                "class_index": int(label_idx),
                "similarity": round(float(score), 4),
                "drawing_path": str(drawing_path)
            })
        return output


# ============================================================
# ArcFace 模型（训练/推理用）
# ============================================================
class ArcFaceHead(torch.nn.Module):
    """ArcFace 角度间隔分类头"""

    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        cosine = F.linear(embeddings, F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        theta_m = theta + self.m
        cosine_m = torch.cos(theta_m)
        output = one_hot * cosine_m + (1 - one_hot) * cosine
        return output * self.s


class ArcFaceExtractor:
    """加载训练好的 ArcFace 模型做特征提取"""

    def __init__(self, model_path, num_classes=None, device=None):
        self.device = device or Config.DEVICE

        # 加载模型
        from torchvision.models import resnet50
        self.backbone = resnet50(pretrained=False)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Identity()

        self.embedding = torch.nn.Linear(in_features, Config.EMBEDDING_DIM)

        checkpoint = torch.load(model_path, map_location=self.device)
        # 兼容两种保存格式
        if 'backbone' in checkpoint:
            self.backbone.load_state_dict(checkpoint['backbone'])
            self.embedding.load_state_dict(checkpoint['embedding'])
        else:
            self.backbone.load_state_dict({k.replace('backbone.', ''): v
                                           for k, v in checkpoint.items()
                                           if k.startswith('backbone.')})
            self.embedding.load_state_dict({k.replace('embedding.', ''): v
                                            for k, v in checkpoint.items()
                                            if k.startswith('embedding.')})

        self.backbone.to(self.device).eval()
        self.embedding.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((Config.AUGMENT_SIZE, Config.AUGMENT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        print(f"[ArcFace] 模型已加载: {model_path}")

    def extract(self, image_or_path):
        if isinstance(image_or_path, (str, Path)):
            img = Image.open(image_or_path).convert('RGB')
        elif isinstance(image_or_path, np.ndarray):
            img = Image.fromarray(image_or_path).convert('RGB')
        else:
            img = image_or_path.convert('RGB')

        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            x = self.backbone(tensor)
            x = self.embedding(x)
            x = F.normalize(x, p=2, dim=1)
        return x.cpu().numpy().squeeze()

    def extract_batch(self, image_paths, verbose=True):
        features = []
        iterator = image_paths
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(image_paths, desc="ArcFace提取特征")
        for path in iterator:
            features.append(self.extract(path))
        return np.stack(features)


if __name__ == "__main__":
    # 测试：扫描并建立标签
    print("=" * 50)
    print("RDM 工具模块测试")
    print(f"设备: {Config.DEVICE}")
    print(f"数据目录: {Config.DATA_DIR}")
    print(f"标签目录: {Config.LABEL_DIR}")

    labels = build_labels()
    print(f"标签示例: {list(labels.items())[:3]}")
    print("✅ utils.py 就绪")
