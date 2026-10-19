#!/usr/bin/env python3
"""
Register train and val YOLO splits with 3LC.

Does not register a test split: participants have test images on disk only and run
inference via predict.py; a 3LC test table is not part of the competition workflow.

Idempotent: if tables already exist, skips creation and prints .latest() URLs.

Prerequisites (see README): 3lc installed, `3lc login`, and data under data/.
All names and paths come from config.yaml.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*from_yolo.*deprecated.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*from_yolo.*deprecated.*", category=UserWarning)
logging.getLogger("3lc").setLevel(logging.ERROR)

import tlc  # noqa: E402

WORK_DIR = Path(__file__).resolve().parent


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (WORK_DIR / p).resolve()


# ---------------------------------------------------------------------------
# 本地适配（2026-10-15）：3LC 3.0 移除了 Table.from_yolo（官方 breaking change），
# 改用 Table.from_yolo_url。后者不再接收 dataset.yaml，改为接收【图片目录】
# +【显式的类别映射】。故从 dataset.yaml 的 names 字段读取类别映射，
# 保持 kit 原本“配置驱动”的设计：改数据集只需改 dataset.yaml。
# ---------------------------------------------------------------------------
def _read_categories(dataset_yaml: Path) -> dict[int, str]:
    """从 dataset.yaml 的 names 字段读出 {class_id(int): 类别名}。"""
    data = _load_yaml(dataset_yaml)
    names = data.get("names")
    if not isinstance(names, dict) or not names:
        raise SystemExit(
            f"dataset.yaml 缺少可用的 names 映射: {dataset_yaml}\n"
            "3LC 3.x 的 Table.from_yolo_url 需要显式类别映射，无法自动推导。"
        )
    return {int(k): str(v) for k, v in names.items()}


def main() -> int:
    os.chdir(WORK_DIR)
    cfg_path = WORK_DIR / "config.yaml"
    if not cfg_path.is_file():
        print(f"ERROR: Missing {cfg_path}", file=sys.stderr)
        return 1

    cfg = _load_yaml(cfg_path)
    paths = cfg.get("paths", {})
    tlc_cfg = cfg.get("tlc", {})

    dataset_yaml = _resolve(str(paths.get("dataset_yaml", "dataset.yaml")))
    if not dataset_yaml.is_file():
        print(f"ERROR: dataset yaml not found: {dataset_yaml}", file=sys.stderr)
        return 1

    project_name = str(tlc_cfg.get("project_name", "ua_detrac_vehicle_detection"))
    dataset_name = str(tlc_cfg.get("dataset_name", "ua_detrac_10k"))
    train_name = str(tlc_cfg.get("train_table_name", f"{dataset_name}-train"))
    val_name = str(tlc_cfg.get("val_table_name", f"{dataset_name}-val"))

    for sub in ("data/train/images", "data/train/labels", "data/val/images", "data/val/labels"):
        p = WORK_DIR / sub
        if not p.is_dir():
            print(f"ERROR: Expected directory missing: {p}", file=sys.stderr)
            return 1

    categories = _read_categories(dataset_yaml)

    print("=" * 70)
    print("REGISTER 3LC TABLES (train + val only)")
    print("=" * 70)
    print(f"  dataset_yaml: {dataset_yaml}")
    print(f"  project: {project_name}, dataset: {dataset_name}")
    print(f"  categories: {categories}")

    tables_exist = False
    try:
        tlc.Table.from_names(
            project_name=project_name, dataset_name=dataset_name, table_name=train_name
        )
        tlc.Table.from_names(
            project_name=project_name, dataset_name=dataset_name, table_name=val_name
        )
        tables_exist = True
    except Exception:
        pass

    if tables_exist:
        print("\n  Tables already exist — skipping creation (idempotent).")
        train_t = tlc.Table.from_names(
            project_name=project_name, dataset_name=dataset_name, table_name=train_name
        ).latest()
        val_t = tlc.Table.from_names(
            project_name=project_name, dataset_name=dataset_name, table_name=val_name
        ).latest()
        print(f"\n  Train: {train_t.url}")
        print(f"  Val:   {val_t.url}")
    else:
        print("\n  Creating train table from YOLO...")
        train_t = tlc.Table.from_yolo_url(
            images_url=WORK_DIR / "data/train/images",
            categories=categories,
            task="detect",
            dataset_name=dataset_name,
            project_name=project_name,
            table_name=train_name,
        )
        print(f"  Train samples: {len(train_t)}")

        print("\n  Creating val table from YOLO...")
        val_t = tlc.Table.from_yolo_url(
            images_url=WORK_DIR / "data/val/images",
            categories=categories,
            task="detect",
            dataset_name=dataset_name,
            project_name=project_name,
            table_name=val_name,
        )
        print(f"  Val samples:   {len(val_t)}")

        print(f"\n  Train: {train_t.url}")
        print(f"  Val:   {val_t.url}")

    print("\n  Test images: use data/test/images with predict.py (not registered in 3LC).")
    print("\n" + "=" * 70)
    print("OK — next: python train.py")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
