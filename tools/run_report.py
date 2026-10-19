#!/usr/bin/env python3
"""本地读取训练指标 —— 完全不需要 3LC Dashboard 或 Object Service。

用法（在项目根目录执行）：
    python tools/run_report.py                  # 汇总所有 run
    python tools/run_report.py yolov8n_baseline # 只看一个 run

数据来源（全部在本机磁盘上）：
    1) Ultralytics 逐 epoch 指标  <kit>/runs/detect/<run>/results.csv
    2) 3LC 逐类别聚合指标         <LOCALAPPDATA>/3LC/3LC/projects/<proj>/runs/<run>/metrics_*.parquet
    3) 3LC 逐样本指标（后续分析用）  同上，行数 = 图片数

说明：
    - 3LC 的三张聚合表通过 "label=4(all) 那行的 num_images" 与数据集两张表的 row_count
      匹配来判定是 train 还是 val，不依赖硬编码的表编号。
    - 本脚本只读取文件，不训练、不推理、不联网。
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

# ---- 路径配置（如有变动只改这里）-----------------------------------------
KIT = Path(
    r"E:\kaggle\3LC_Multi_Vehicle_detection_Challenge\multi-vehicle_kaggle_competition_starter_kit_files"
)
TLC_PROJECTS = Path(os.environ.get("LOCALAPPDATA", "")) / "3LC" / "3LC" / "projects"
TLC_PROJECT = "ua_detrac_vehicle_detection"
TLC_DATASET = "ua_detrac_10k"
# ------------------------------------------------------------------------


def read_class_names() -> dict[int, str]:
    """从 dataset.yaml 读类别名。"""
    import yaml

    p = KIT / "dataset.yaml"
    if not p.is_file():
        return {}
    names = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("names") or {}
    return {int(k): str(v) for k, v in names.items()}


def read_table_row_counts() -> dict[str, int]:
    """读出 train/val 两张表的行数，用于给 3LC 聚合表判定 split。"""
    out: dict[str, int] = {}
    base = TLC_PROJECTS / TLC_PROJECT / "datasets" / TLC_DATASET / "tables"
    if not base.is_dir():
        return out
    for d in sorted(base.iterdir()):
        f = d / "object.3lc.json"
        if not f.is_file():
            continue
        try:
            out[d.name] = int(json.loads(f.read_text(encoding="utf-8")).get("row_count") or 0)
        except Exception:
            continue
    return out


def read_aggregate_tables(run_dir: Path) -> dict[str, list[dict]]:
    """读取 3LC 的聚合指标表（含 mAP 列的那些）。返回 {目录名: 行列表}。"""
    import pyarrow.parquet as pq

    out: dict[str, list[dict]] = {}
    if not run_dir.is_dir():
        return out
    for d in sorted(run_dir.glob("metrics_*")):
        f = d / f"{d.name}.parquet"
        if not f.is_file():
            continue
        try:
            rows = pq.read_table(f).to_pylist()
        except Exception:
            continue
        if rows and "mAP" in rows[0]:
            out[d.name] = rows
    return out


def read_results_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str, default: float = 0.0) -> float:
    v = row.get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def print_per_epoch(rows: list[dict]) -> None:
    if not rows:
        print("  (没有 results.csv)")
        return
    hdr = f"  {'epoch':>5} {'mAP50':>8} {'mAP50-95':>9} {'P':>7} {'R':>7} {'box_loss':>9} {'time(s)':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(
            f"  {r.get('epoch', '?'):>5} "
            f"{_f(r, 'metrics/mAP50(B)'):>8.4f} "
            f"{_f(r, 'metrics/mAP50-95(B)'):>9.4f} "
            f"{_f(r, 'metrics/precision(B)'):>7.4f} "
            f"{_f(r, 'metrics/recall(B)'):>7.4f} "
            f"{_f(r, 'train/box_loss'):>9.4f} "
            f"{_f(r, 'time'):>8.1f}"
        )


def print_per_class(rows: list[dict], split: str, names: dict[int, str]) -> None:
    """打印某个 split 的逐类别指标。label 0..N-1 是类别，label=4(最大) 是汇总。"""
    if not rows:
        return
    # 汇总行 = num_instances 最大的那行
    all_row = max(rows, key=lambda r: r.get("num_instances") or 0)
    print(f"\n  [{split}] epoch={all_row.get('epoch')}  逐类别指标(3LC 聚合表)")
    hdr = f"  {'label':>5} {'类别':<8} {'images':>7} {'inst':>7} {'P':>7} {'R':>7} {'mAP':>8} {'mAP50-95':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in sorted(rows, key=lambda x: x.get("label") or 0):
        lab = int(r.get("label") or 0)
        name = names.get(lab, "ALL(all)" if r is all_row else "?")
        mark = "*" if r is all_row else " "
        print(
            f"  {lab:>5}{mark}{name:<8} "
            f"{int(r.get('num_images') or 0):>7} "
            f"{int(r.get('num_instances') or 0):>7} "
            f"{_f(r, 'precision'):>7.4f} "
            f"{_f(r, 'recall'):>7.4f} "
            f"{_f(r, 'mAP'):>8.4f} "
            f"{_f(r, 'mAP50-95'):>9.4f}"
        )


def report_run(run_name: str, names: dict[int, str], row_counts: dict[str, int]) -> None:
    print("=" * 78)
    print(f"RUN: {run_name}")
    print("=" * 78)

    run_csv_dir = KIT / "runs" / "detect" / run_name
    epochs = read_results_csv(run_csv_dir / "results.csv")
    if epochs:
        best = max(epochs, key=lambda r: _f(r, "metrics/mAP50-95(B)"))
        print(
            f"  最优 epoch: {best.get('epoch')}  "
            f"mAP50={_f(best, 'metrics/mAP50(B)'):.4f}  "
            f"mAP50-95={_f(best, 'metrics/mAP50-95(B)'):.4f}"
        )
        print()
        print_per_epoch(epochs)
        print(f"\n  权重: {run_csv_dir / 'weights' / 'best.pt'}")
        for extra in ("results.png", "confusion_matrix.png", "confusion_matrix_normalized.png"):
            if (run_csv_dir / extra).is_file():
                print(f"  图表: {run_csv_dir / extra}")
    else:
        print("  (未找到 ultralytics 的 results.csv)")

    # 3LC 聚合表
    tlc_run = TLC_PROJECTS / TLC_PROJECT / "runs" / run_name
    aggs = read_aggregate_tables(tlc_run)
    if not aggs:
        print("\n  (未找到 3LC 聚合指标表 —— 训练时是否启用了 3LC 指标收集？)")
        return

    # 给每张聚合表判定 split：用 label=4 那行的 num_images 去匹配表 row_count
    for tab, rows in aggs.items():
        all_row = max(rows, key=lambda r: r.get("num_instances") or 0)
        n_img = int(all_row.get("num_images") or 0)
        split = "unknown"
        for name, cnt in row_counts.items():
            if cnt == n_img:
                split = name
                break
        print_per_class(rows, split, names)

    # 逐样本表
    for d in sorted(tlc_run.glob("metrics_*")):
        f = d / f"{d.name}.parquet"
        if not f.is_file():
            continue
        try:
            import pyarrow.parquet as pq

            schema_names = pq.read_schema(f).names
        except Exception:
            continue
        if "mAP" in schema_names:
            continue  # 聚合表已打印
        n = pq.read_metadata(f).num_rows
        print(f"\n  逐样本表: {f}  ({n} 行, 列: {', '.join(schema_names)})")


def main() -> int:
    names = read_class_names()
    row_counts = read_table_row_counts()

    runs_root = KIT / "runs" / "detect"
    if len(sys.argv) > 1:
        targets = sys.argv[1:]
    else:
        targets = sorted([p.name for p in runs_root.iterdir() if p.is_dir()])

    if not targets:
        print("没有找到任何 run。先跑一次 train.py。")
        return 1

    print(f"ultralytics runs 根目录: {runs_root}")
    print(f"3LC 项目根目录        : {TLC_PROJECTS / TLC_PROJECT}")
    print()
    for name in targets:
        report_run(name, names, row_counts)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
