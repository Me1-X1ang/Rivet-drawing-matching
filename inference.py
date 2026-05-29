```python id="rx7t95"
"""
inference.py - 图纸检索

模式：
  - dinov2：零样本检索
  - arcface：训练后检索
"""

import sys
import argparse

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    Config,
    build_labels,
    DrawingGallery
)


def create_gallery(args):

    if args.mode == "dinov2":

        return DrawingGallery(
            use_dinov2=True
        )

    model_path = args.model or (
        Config.MODEL_DIR / "best.pth"
    )

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"模型不存在: {model_path}"
        )

    return DrawingGallery(
        use_dinov2=False,
        arcface_model_path=model_path
    )


def print_results(results, query_path, show_paths=False):

    print(f"\nquery: {query_path}\n")

    print(f"{'rank':<6}{'drawing_id':<18}{'score':<10}")

    for rank, result in enumerate(results, 1):

        score = result["similarity"] * 100

        print(
            f"{rank:<6}"
            f"{result['drawing_id']:<18}"
            f"{score:>6.1f}%"
        )

        if show_paths:
            print(f"       {result['drawing_path']}")

    best = results[0]

    print(
        f"\nbest: "
        f"{best['drawing_id']} "
        f"({best['similarity']*100:.1f}%)"
    )


def run_query(args):

    gallery = create_gallery(args)

    gallery.build_index(
        force_rebuild=args.rebuild
    )

    results = gallery.query(
        args.query,
        k=args.top_k
    )

    print_results(
        results,
        args.query,
        args.verbose
    )


def run_interactive(args):

    gallery = create_gallery(args)

    gallery.build_index(
        force_rebuild=args.rebuild
    )

    print("[交互模式]")
    print("输入 q / exit 退出")
    print("输入 rebuild 重建索引\n")

    while True:

        try:
            query_path = input("> ")

        except (EOFError, KeyboardInterrupt):
            print()
            break

        query_path = (
            query_path
            .strip()
            .strip('"')
            .strip("'")
        )

        if not query_path:
            continue

        if query_path.lower() in {
            "q",
            "quit",
            "exit"
        }:
            break

        if query_path.lower() == "rebuild":

            print("[索引] rebuilding...")

            gallery.build_index(
                force_rebuild=True
            )

            continue

        if not Path(query_path).exists():

            print(f"[错误] 文件不存在")

            continue

        results = gallery.query(
            query_path,
            k=args.top_k
        )

        print_results(results, query_path)


def rebuild_index(args):

    gallery = create_gallery(args)

    gallery.build_index(
        force_rebuild=True
    )

    print("[索引] 完成")


def main():

    parser = argparse.ArgumentParser(
        description="铆钉图纸检索"
    )

    parser.add_argument(
        "--query",
        "-q",
        type=str
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true"
    )

    parser.add_argument(
        "--rebuild",
        "-r",
        action="store_true"
    )

    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="dinov2",
        choices=["dinov2", "arcface"]
    )

    parser.add_argument(
        "--model",
        type=str
    )

    parser.add_argument(
        "--top_k",
        "-k",
        type=int,
        default=Config.TOP_K
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true"
    )

    args = parser.parse_args()

    if not (
        Config.LABEL_DIR / "labels.json"
    ).exists():

        print("[初始化] build labels")

        build_labels()

    if args.interactive:

        run_interactive(args)

    elif args.query:

        run_query(args)

    elif args.rebuild:

        rebuild_index(args)

    else:

        parser.print_help()


if __name__ == "__main__":
    main()
```
