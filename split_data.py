#!/usr/bin/env python3
"""
Split a root that contains only class subfolders into train/ and val/ (stratified per-class).

Example original layout:
  root/
    class1folder/
    class2folder/
    class3folder/

Usage examples:
  # recommended: move files (saves disk)
  python split_root_to_train_val.py --root /path/to/root --val-size 0.2 --move

  # safe: copy files
  python split_root_to_train_val.py --root /path/to/root --val-size 0.2 --copy

  # dry-run to preview
  python split_root_to_train_val.py --root /path/to/root --val-size 0.2 --dry-run
"""

import argparse
from pathlib import Path
import random
import shutil
import sys

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder: Path):
    return [
        p
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def parse_val_size(s: str):
    # accept fraction like "0.2" or integer like "10"
    if "." in s:
        v = float(s)
        if not (0.0 < v < 1.0):
            raise argparse.ArgumentTypeError("Fraction val-size phải nằm trong (0,1).")
        return ("frac", v)
    else:
        v = int(s)
        if v < 0:
            raise argparse.ArgumentTypeError("Integer val-size phải >= 0.")
        return ("abs", v)


def split_list(imgs, val_spec, seed):
    n = len(imgs)
    if n == 0:
        return [], []
    rnd = random.Random(seed)
    imgs_shuffled = imgs.copy()
    rnd.shuffle(imgs_shuffled)
    mode, v = val_spec
    if mode == "frac":
        k = int(round(n * v))
    else:
        k = int(min(v, n))
    if n >= 2 and k == 0:
        k = 1
    if k >= n and n >= 2:
        k = n - 1
    val = imgs_shuffled[:k]
    train = imgs_shuffled[k:]
    return train, val


def copy_or_move(paths, dst_dir: Path, move=False):
    for p in paths:
        dst = dst_dir / p.name
        if move:
            shutil.move(str(p), str(dst))
        else:
            shutil.copy2(str(p), str(dst))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default=r"C:\Users\thith\Downloads\dataset_image",
        help="Path to dataset root (contains class subfolders).",
    )
    p.add_argument(
        "--val-size",
        default="0.2",
        type=parse_val_size,
        help="Fraction (0.2) or absolute count per-class (e.g. 10).",
    )
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--move", action="store_true", help="Move files (recommended).")
    grp.add_argument("--copy", action="store_true", help="Copy files.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--force",
        action="store_true",
        help="Remove existing train/ val/ before creating.",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Show actions without performing them."
    )
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        print(f"[ERROR] Root không tồn tại hoặc không phải thư mục: {root}")
        sys.exit(1)

    train_dir = root / "train"
    val_dir = root / "val"
    if (train_dir.exists() or val_dir.exists()) and not args.force:
        print("[ERROR] Đã tồn tại train/ hoặc val/ trong root. Dùng --force để ghi đè.")
        sys.exit(1)
    if args.force:
        if train_dir.exists():
            shutil.rmtree(train_dir)
        if val_dir.exists():
            shutil.rmtree(val_dir)

    # discover classes: direct subfolders of root EXCLUDING train/ val/
    classes = [
        d
        for d in sorted(root.iterdir())
        if d.is_dir() and d.name not in ("train", "val") and not d.name.startswith(".")
    ]
    if not classes:
        print("[ERROR] Không tìm thấy thư mục lớp nào dưới root.")
        sys.exit(1)

    # prepare output dirs for each class
    for c in classes:
        ensure_dir(train_dir / c.name)
        ensure_dir(val_dir / c.name)

    summary = {}
    for c in classes:
        imgs = list_images(c)
        train_list, val_list = split_list(imgs, args.val_size, args.seed)
        summary[c.name] = {
            "total": len(imgs),
            "train": len(train_list),
            "val": len(val_list),
        }
        if args.dry_run:
            print(
                f"[DRY] {c.name}: total={len(imgs)} -> train={len(train_list)} val={len(val_list)}"
            )
        else:
            copy_or_move(train_list, train_dir / c.name, move=args.move)
            copy_or_move(val_list, val_dir / c.name, move=args.move)

    # print summary
    total = sum(v["total"] for v in summary.values())
    t_train = sum(v["train"] for v in summary.values())
    t_val = sum(v["val"] for v in summary.values())
    print(
        f"\nSummary: classes={len(summary)} total_images={total} -> train={t_train}, val={t_val}"
    )
    for k, v in summary.items():
        print(
            f"  {k:20s} total={v['total']:4d} train={v['train']:4d} val={v['val']:4d}"
        )
    if args.dry_run:
        print("\n[DRY RUN] No files copied/moved.")
    else:
        print(f"\nDone. Created:\n  {train_dir}\n  {val_dir}")


if __name__ == "__main__":
    main()
