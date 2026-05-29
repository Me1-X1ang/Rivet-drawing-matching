"""
inference.py - 铆钉图纸检索推理脚本
输入一张损坏铆钉的照片，返回最匹配的图纸。

两种模式:
  1. dinov2（默认）：零样本，不需要训练，直接使用 DINOv2 预训练模型
  2. arcface：需先运行 train.py 训练 ArcFace 模型

用法:
  # 单张照片查询
  python inference.py --query photo.jpg

  # 指定模型模式
  python inference.py --query photo.jpg --mode arcface --model models/best.pth

  # 重建索引（图纸库更新后）
  python inference.py --rebuild

  # 交互模式
  python inference.py --interactive
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    Config, load_labels, build_labels,
    DrawingGallery, DINOv2Extractor, ArcFaceExtractor
)


def print_banner():
    print(r"""
    ╔══════════════════════════════════════╗
    ║     🔩 铆钉图纸检索系统 RDM v1.0      ║
    ║     Rivet Drawing Matcher            ║
    ╚══════════════════════════════════════╝
    """)


def print_results(results, query_path, show_paths=False):
    """格式化打印检索结果"""
    print(f"\n{'='*60}")
    print(f"📷 查询照片: {query_path}")
    print(f"{'='*60}")
    print(f"{'排名':<6}{'图纸编号':<16}{'相似度':<12}")
    print(f"{'-'*34}")

    for rank, r in enumerate(results, 1):
        sim_pct = r['similarity'] * 100
        bar = '█' * int(sim_pct / 5) + '░' * (20 - int(sim_pct / 5))
        print(f"#{rank:<5}{r['drawing_id']:<16}{sim_pct:>6.1f}%  {bar}")

        if show_paths:
            print(f"      📁 {r['drawing_path']}")

    print(f"\n💡 最佳匹配: {results[0]['drawing_id']} (置信度 {results[0]['similarity']*100:.1f}%)")


def cmd_query(args):
    """单张照片查询"""
    print_banner()

    # 初始化图纸库
    if args.mode == 'dinov2':
        print("[模式] DINOv2 零样本检索")
        gallery = DrawingGallery(use_dinov2=True)
    elif args.mode == 'arcface':
        if not args.model:
            args.model = Config.MODEL_DIR / "best.pth"
        if not Path(args.model).exists():
            print(f"[错误] ArcFace 模型不存在: {args.model}")
            print("请先运行 train.py 训练模型，或使用 --mode dinov2")
            return
        print(f"[模式] ArcFace 检索 (模型: {args.model})")
        gallery = DrawingGallery(use_dinov2=False, arcface_model_path=args.model)
    else:
        print(f"[错误] 未知模式: {args.mode}")
        return

    # 构建/加载索引
    gallery.build_index(force_rebuild=args.rebuild)

    # 查询
    results = gallery.query(args.query, k=args.top_k)
    print_results(results, args.query, show_paths=args.verbose)


def cmd_interactive(args):
    """交互模式：持续输入照片路径"""
    print_banner()

    # 初始化
    if args.mode == 'dinov2':
        print("[模式] DINOv2 零样本检索")
        gallery = DrawingGallery(use_dinov2=True)
    else:
        if not args.model:
            args.model = Config.MODEL_DIR / "best.pth"
        if not Path(args.model).exists():
            print(f"[错误] 模型不存在: {args.model}")
            return
        print(f"[模式] ArcFace 检索")
        gallery = DrawingGallery(use_dinov2=False, arcface_model_path=args.model)

    gallery.build_index(force_rebuild=args.rebuild)

    print("\n📸 交互模式：输入照片路径进行检索")
    print("   输入 'q' 或 'exit' 退出")
    print("   输入 'rebuild' 重建索引")
    print("   拖拽图片到终端窗口即可获取路径\n")

    while True:
        try:
            query_path = input("🔍 照片路径 > ").strip().strip('"').strip("'")
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if query_path.lower() in ('q', 'exit', 'quit'):
            print("👋 再见！")
            break

        if query_path.lower() == 'rebuild':
            print("[索引] 正在重建...")
            gallery.build_index(force_rebuild=True)
            continue

        if not query_path:
            continue

        if not Path(query_path).exists():
            print(f"[错误] 文件不存在: {query_path}")
            continue

        results = gallery.query(query_path, k=args.top_k)
        print_results(results, query_path)


def cmd_build(args):
    """仅构建索引"""
    print_banner()
    if args.mode == 'dinov2':
        gallery = DrawingGallery(use_dinov2=True)
    else:
        if not args.model:
            args.model = Config.MODEL_DIR / "best.pth"
        gallery = DrawingGallery(use_dinov2=False, arcface_model_path=args.model)

    gallery.build_index(force_rebuild=True)
    print("\n✅ 索引构建完成！现在可以运行 inference.py --query photo.jpg 进行检索")


def main():
    parser = argparse.ArgumentParser(
        description="铆钉图纸检索 - 输入损坏照片，输出匹配图纸",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python inference.py --query damaged_rivet.jpg
  python inference.py --query photo.jpg --top_k 10 --verbose
  python inference.py --interactive
  python inference.py --rebuild --mode arcface --model models/best.pth
        """
    )

    # 子命令
    parser.add_argument('--query', '-q', type=str, default=None,
                        help='要查询的照片路径')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='交互模式')
    parser.add_argument('--rebuild', '-r', action='store_true',
                        help='强制重建索引')

    # 参数
    parser.add_argument('--mode', '-m', type=str, default='dinov2',
                        choices=['dinov2', 'arcface'],
                        help='检索模式: dinov2=零样本(默认), arcface=训练后模型')
    parser.add_argument('--model', type=str, default=None,
                        help='ArcFace 模型路径（仅 --mode arcface 时需要）')
    parser.add_argument('--top_k', '-k', type=int, default=Config.TOP_K,
                        help=f'返回前 K 个结果（默认 {Config.TOP_K}）')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='显示详细路径')

    args = parser.parse_args()

    # 先确保标签存在
    if not (Config.LABEL_DIR / "labels.json").exists():
        print("[初始化] 首次运行，正在扫描图纸并生成标签...")
        build_labels()

    # 路由
    if args.interactive:
        cmd_interactive(args)
    elif args.query:
        cmd_query(args)
    elif args.rebuild:
        cmd_build(args)
    else:
        # 没有 query 也没有 interactive：显示帮助
        parser.print_help()
        print("\n💡 快速开始：")
        print("  1. 首次运行（自动建索引）: python inference.py --query 你的照片.jpg")
        print("  2. 交互模式:              python inference.py --interactive")
        print("  3. 重建索引:              python inference.py --rebuild")


if __name__ == "__main__":
    main()
