"""
augment.py - 图纸数据增强（内存优化版）

优化点：
  - 多进程结果改为增量写入 JSON，不在主进程堆积图像数据
  - chunksize 控制，避免进程间传输过多数据
  - 默认并行数限制为 2，防止笔记本内存爆炸
  - 增量写入索引文件，崩溃后可续跑
  - 单张图像处理完立即释放，不累积
"""

import os
import sys
import cv2
import json
import numpy as np
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).parent))
from utils import Config, load_labels

# ============================================================
# 配置
# ============================================================
OUTPUT_SIZE = Config.AUGMENT_SIZE
AUG_PER_DRAWING = 50

P_ROTATE = 0.9
P_PERSPECTIVE = 0.7
P_BLUR = 0.5
P_NOISE = 0.7
P_TEXTURE = 0.8
P_RUST = 0.6
P_SCRATCH = 0.5
P_BRIGHTNESS = 0.8
P_ERASE = 0.4
P_BG_CHANGE = 0.9


# ============================================================
# 增强函数（与原版相同，略作内存友好调整）
# ============================================================
def geometric_augment(img):
    h, w = img.shape[:2]
    if random.random() < P_ROTATE:
        angle = random.uniform(-25, 25)
        scale = random.uniform(0.75, 1.25)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
        img = cv2.warpAffine(img, M, (w, h),
                             borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    if random.random() < P_PERSPECTIVE:
        offset = 0.06 * max(w, h)
        src_pts = np.float32([[0,0],[w-1,0],[0,h-1],[w-1,h-1]])
        dst_pts = np.float32([
            [random.uniform(0, offset), random.uniform(0, offset)],
            [w - random.uniform(0, offset), random.uniform(0, offset)],
            [random.uniform(0, offset), h - random.uniform(0, offset)],
            [w - random.uniform(0, offset), h - random.uniform(0, offset)]
        ])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        img = cv2.warpPerspective(img, M, (w, h),
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return img


def blur_augment(img):
    if random.random() < P_BLUR:
        ksize = random.choice([3, 5, 7])
        if random.random() < 0.3:
            k = random.randint(5, 15)
            angle = random.uniform(0, 360)
            kernel = np.zeros((k, k))
            center = k // 2
            for i in range(k):
                x = int(center + (i - center) * np.cos(np.radians(angle)))
                y = int(center + (i - center) * np.sin(np.radians(angle)))
                if 0 <= x < k and 0 <= y < k:
                    kernel[y, x] = 1
            s = kernel.sum()
            if s > 0:
                kernel /= s
            img = cv2.filter2D(img, -1, kernel)
        else:
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return img


def noise_augment(img):
    if random.random() < P_NOISE:
        noise_type = random.choice(['gaussian', 'salt_pepper', 'speckle'])
        if noise_type == 'gaussian':
            sigma = random.uniform(3, 18)
            noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        elif noise_type == 'salt_pepper':
            amount = random.uniform(0.001, 0.02)
            mask = np.random.random(img.shape[:2]) < amount
            img[mask] = random.randint(0, 60)
            mask = np.random.random(img.shape[:2]) < amount
            img[mask] = random.randint(200, 255)
        elif noise_type == 'speckle':
            noise = np.random.normal(0, random.uniform(5, 15), img.shape)
            img = np.clip(img.astype(np.float32) + img.astype(np.float32) * noise / 255.0,
                          0, 255).astype(np.uint8)
    return img


def texture_augment(img):
    if random.random() < P_TEXTURE and img.ndim == 3:
        h, w = img.shape[:2]
        texture_size = max(h, w) // random.randint(2, 6)
        tex_h, tex_w = texture_size, texture_size
        texture = np.random.randint(160, 230, (tex_h, tex_w), dtype=np.uint8)
        texture = cv2.resize(texture, (w, h))
        if random.random() < 0.4:
            kernel_size = random.choice([15, 21, 31])
            angle = random.uniform(0, 180)
            kernel = np.zeros((kernel_size, kernel_size))
            center = kernel_size // 2
            for i in range(kernel_size):
                x = int(center + (i - center) * np.cos(np.radians(angle)))
                y = int(center + (i - center) * np.sin(np.radians(angle)))
                if 0 <= x < kernel_size and 0 <= y < kernel_size:
                    kernel[y, x] = 1
            s = kernel.sum()
            kernel /= max(s, 1)
            texture = cv2.filter2D(texture, -1, kernel)
        texture = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
        texture = cv2.GaussianBlur(texture, (11, 11), 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = (gray < 240).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        blend = random.uniform(0.15, 0.4)
        img = (img.astype(np.float32) * (1 - blend * mask[:,:,np.newaxis]) +
               texture.astype(np.float32) * blend * mask[:,:,np.newaxis])
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def rust_augment(img):
    if random.random() < P_RUST and img.ndim == 3:
        h, w = img.shape[:2]
        rust_mask = np.zeros((h, w), dtype=np.float32)
        num_blobs = random.randint(1, 8)
        for _ in range(num_blobs):
            cx = random.randint(0, w-1)
            cy = random.randint(0, h-1)
            radius = random.randint(8, min(h, w)//6)
            blob = np.zeros((h, w), dtype=np.float32)
            for _ in range(random.randint(1, 4)):
                ox = random.randint(-radius//2, radius//2)
                oy = random.randint(-radius//2, radius//2)
                r = random.randint(radius//2, radius)
                cv2.circle(blob, (cx+ox, cy+oy), r, 1.0, -1)
            blob = cv2.GaussianBlur(blob, (radius*2+1, radius*2+1), 0)
            rust_mask += blob * random.uniform(0.25, 0.7)
        rust_mask = np.clip(rust_mask, 0, 1)
        rust_b = random.randint(30, 80)
        rust_g = random.randint(20, 60)
        rust_r = random.randint(60, 160)
        rust_color = np.array([rust_b, rust_g, rust_r], dtype=np.float32)
        blend = random.uniform(0.3, 0.7)
        img = (img.astype(np.float32) * (1 - rust_mask[:,:,np.newaxis] * blend) +
               rust_color * rust_mask[:,:,np.newaxis] * blend)
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def scratch_augment(img):
    if random.random() < P_SCRATCH and img.ndim == 3:
        num_scratches = random.randint(0, 6)
        for _ in range(num_scratches):
            x1 = random.randint(0, img.shape[1]-1)
            y1 = random.randint(0, img.shape[0]-1)
            length = random.randint(15, 100)
            angle = random.uniform(0, 2 * np.pi)
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))
            thickness = random.randint(1, 3)
            color = (random.randint(0,60), random.randint(0,60), random.randint(0,60))
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img


def brightness_augment(img):
    if random.random() < P_BRIGHTNESS:
        alpha = random.uniform(0.6, 1.8)
        beta = random.uniform(-40, 40)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img


def erase_augment(img):
    if random.random() < P_ERASE:
        h, w = img.shape[:2]
        num_erases = random.randint(1, 4)
        for _ in range(num_erases):
            erase_area = random.uniform(0.02, 0.12) * h * w
            aspect = random.uniform(0.3, 3.3)
            eh = int(np.sqrt(erase_area / aspect))
            ew = int(np.sqrt(erase_area * aspect))
            if eh >= h or ew >= w:
                continue
            x = random.randint(0, w - ew)
            y = random.randint(0, h - eh)
            img[y:y+eh, x:x+ew] = random.randint(180, 255)
    return img


def background_augment(img):
    if random.random() < P_BG_CHANGE and img.ndim == 3:
        h, w = img.shape[:2]
        bg_type = random.choice(['solid', 'gradient', 'texture', 'noise'])
        if bg_type == 'solid':
            bg = np.full((h, w, 3), random.randint(60, 220), dtype=np.uint8)
        elif bg_type == 'gradient':
            bg = np.zeros((h, w, 3), dtype=np.uint8)
            for c in range(3):
                g = np.linspace(random.randint(40,200), random.randint(40,200), w)
                bg[:, :, c] = np.tile(g, (h, 1))
        elif bg_type == 'texture':
            bg_small = np.random.randint(40, 200, (h//8, w//8, 3), dtype=np.uint8)
            bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)
            bg = cv2.GaussianBlur(bg, (31, 31), 0)
        else:
            bg = np.random.randint(80, 180, (h, w, 3), dtype=np.uint8)
            bg = cv2.GaussianBlur(bg, (15, 15), 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fg_mask = (gray < 235).astype(np.float32)
        fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
        img = (img.astype(np.float32) * fg_mask[:,:,np.newaxis] +
               bg.astype(np.float32) * (1 - fg_mask[:,:,np.newaxis]))
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# ============================================================
# 核心：单张图纸的全部变体（在子进程完成，只返回轻量元数据）
# ============================================================
def augment_single(drawing_path):
    img = cv2.imread(str(drawing_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = OUTPUT_SIZE / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h))
    canvas = np.full((OUTPUT_SIZE, OUTPUT_SIZE, 3), 255, dtype=np.uint8)
    y_off = (OUTPUT_SIZE - new_h) // 2
    x_off = (OUTPUT_SIZE - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = img
    img = canvas

    img = geometric_augment(img)
    img = texture_augment(img)
    img = rust_augment(img)
    img = scratch_augment(img)
    img = blur_augment(img)
    img = noise_augment(img)
    img = background_augment(img)
    img = erase_augment(img)
    img = brightness_augment(img)
    return img


def process_one_drawing(args):
    """
    子进程任务：生成所有变体、直接写入磁盘。
    主进程只收到轻量的元数据列表，不接收任何图像数据。
    """
    drawing_path, class_idx, num_variants, output_dir = args
    drawing_id = Path(drawing_path).stem
    meta = []   # 只收集 (相对路径字符串, class_idx)，不含图像数据

    for variant in range(num_variants):
        aug_img = augment_single(drawing_path)
        if aug_img is None:
            continue
        out_name = f"{drawing_id}_aug{variant:03d}.png"
        out_path = Path(output_dir) / out_name
        cv2.imwrite(str(out_path), aug_img)
        # ★ 关键：aug_img 在这里离开作用域，子进程立即释放
        del aug_img
        meta.append((out_name, class_idx))   # 只传文件名，不传图像

    return meta   # 返回给主进程的数据极小


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="铆钉图纸数据增强（内存优化版）")
    parser.add_argument('--num', type=int, default=AUG_PER_DRAWING,
                        help=f'每张图纸生成的变体数量（默认 {AUG_PER_DRAWING}）')
    parser.add_argument('--workers', type=int, default=2,
                        help='并行进程数（默认 2，笔记本建议不超过 4）')
    parser.add_argument('--data', type=str, default=str(Config.DATA_DIR))
    parser.add_argument('--output', type=str, default=str(Config.AUGMENTED_DIR))
    parser.add_argument('--limit', type=int, default=0,
                        help='只处理前 N 张图纸（0=全部）')
    args = parser.parse_args()

    labels = load_labels()
    num_classes = len(labels)
    print(f"[增强] 总类别数: {num_classes}")
    print(f"[增强] 每类生成: {args.num} 张变体")
    print(f"[增强] 预计总共: {num_classes * args.num} 张增强图像")
    print(f"[增强] 并行进程: {args.workers}")

    drawing_ids = sorted(labels.keys())
    if args.limit > 0:
        drawing_ids = drawing_ids[:args.limit]
        print(f"[增强] 仅处理前 {args.limit} 张")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ★ 断点续跑：跳过已完成的图纸
    index_path = output_dir / "augment_index.json"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        done_ids = {
            Path(s["path"]).stem.rsplit("_aug", 1)[0]
            for s in existing.get("samples", [])
        }
        drawing_ids = [d for d in drawing_ids if d not in done_ids]
        print(f"[增强] 已有进度，跳过 {len(done_ids)} 张，剩余 {len(drawing_ids)} 张")
        all_samples = existing.get("samples", [])
    else:
        all_samples = []

    tasks = []
    for did in drawing_ids:
        drawing_path = Path(args.data) / f"{did}.png"
        if not drawing_path.exists():
            print(f"[警告] 文件不存在，跳过: {drawing_path}")
            continue
        tasks.append((str(drawing_path), labels[did], args.num, str(output_dir)))

    if not tasks:
        print("[信息] 没有需要处理的图纸，已全部完成。")
        return

    # ★ 增量写入：每处理完一张图纸就更新索引，防止崩溃丢失进度
    def flush_index():
        index = {
            "total_images": len(all_samples),
            "total_classes": num_classes,
            "aug_per_drawing": args.num,
            "samples": all_samples
        }
        tmp_path = index_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False)
        tmp_path.replace(index_path)   # 原子替换，防止写到一半崩溃

    FLUSH_EVERY = 10   # 每完成 10 张图纸写一次索引

    if args.workers > 1 and len(tasks) > 1:
        print(f"[增强] 使用 {args.workers} 个进程并行（chunksize=1 控制内存）...")
        with Pool(args.workers) as pool:
            for i, meta in enumerate(
                tqdm(pool.imap_unordered(process_one_drawing, tasks, chunksize=1),
                     total=len(tasks), desc="增强进度")
            ):
                # meta 只含文件名 + class_idx，极轻量
                for out_name, class_idx in meta:
                    all_samples.append({
                        "path": str(output_dir / out_name),
                        "class_idx": class_idx
                    })
                if (i + 1) % FLUSH_EVERY == 0:
                    flush_index()
    else:
        print("[增强] 单进程处理...")
        for i, task in enumerate(tqdm(tasks, desc="增强进度")):
            meta = process_one_drawing(task)
            for out_name, class_idx in meta:
                all_samples.append({
                    "path": str(output_dir / out_name),
                    "class_idx": class_idx
                })
            if (i + 1) % FLUSH_EVERY == 0:
                flush_index()

    flush_index()
    print(f"\n[增强] ✅ 完成！共生成 {len(all_samples)} 张增强图像")
    print(f"[增强] 索引文件: {index_path}")
    print(f"[增强] 输出目录: {output_dir}")


if __name__ == "__main__":
    main()