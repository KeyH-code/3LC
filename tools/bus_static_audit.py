#!/usr/bin/env python3
"""Mechanical bus-position audit. No model, network, annotation writing or training.

build: exclusive-create evidence; verify: read-only independent recomputation;
summarize: validate user feedback, print statistics (never changes annotations).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

import numpy as np
from PIL import Image, __version__ as PIL_VERSION

ROOT = Path(__file__).resolve().parents[1]
KIT = "multi-vehicle_kaggle_competition_starter_kit_files"
OUT = ROOT / "reports/bus-static-review"
VAL_DIR = ROOT / "reports/data-audit/random-label-review-20261024"
VAL_LIST = VAL_DIR / "val_use_list_excluding_16.relative.txt"
MANIFEST = ROOT / "reports/data_manifest.json"
BASELINE = ROOT / "reports/baseline-starter-20261024/verification.json"
TEMPLATE = ROOT / "tools/bus_static_review.template.html"
SIZE = 640
NAMES = ["truck", "car", "van", "bus"]
STATUSES = ("unreviewed", "missing", "already_labeled", "no_bus", "uncertain", "scene_mismatch")
LIMITS = [
    "仅发现候选，不是漏标概率，不生成新框或修改原标签。",
    "MVI/帧号仅为命名线索，机位及时间未核实；同位置不等于同一辆静止车辆。",
    "val 信息参与检索；不把同一 val 上后续改善当作独立泛化证据。竞赛训练用途另行确认。",
    "Recall 只针对已有标注，差距不能估计漏标率或确定根因；P 接近不能排除错漏标。",
    "归档没有证明 train/val P/R 使用同一置信度工作点。",
    "本轮没有人工结论；不外推全量，不自动补标、训练、推理或推广类别。",
]


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(obj):
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def safe(path):
    p = Path(path).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError(f"Outside project: {p}")
    parts = p.relative_to(ROOT).parts
    if any(x in (".worktrees", ".git", "test") for x in parts):
        raise ValueError(f"Forbidden input: {p}")
    return p


def sha(path):
    return hashlib.sha256(safe(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(safe(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    with safe(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def git_snapshot():
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                              encoding="utf-8").stdout.strip()
    return {"branch": run("branch", "--show-current"), "head": run("rev-parse", "HEAD"),
            "status": run("status", "--short", "--untracked-files=normal")}


def manifest_entries():
    rows = read_json(MANIFEST)
    result = {}
    for row in rows:
        rel = row["relative"].replace("\\", "/")
        q = PurePosixPath(rel)
        if q.is_absolute() or len(q.parts) != 3 or q.parts[0] not in ("train", "val") or q.parts[1] not in ("images", "labels"):
            raise ValueError("Unexpected manifest path")
        if rel in result:
            raise ValueError("Duplicate manifest path")
        safe(ROOT / KIT / "data" / rel)
        result[rel] = row
    return result


def integrity():
    """Read/hash all original train/val files, including excluded val, solely for integrity."""
    expected = manifest_entries()
    base = safe(ROOT / KIT / "data")
    actual = {}
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            for p in safe(base / split / kind).iterdir():
                p = safe(p)
                if not p.is_file():
                    raise ValueError(f"Unexpected nested data entry: {p}")
                actual[p.relative_to(base).as_posix()] = p
    if set(actual) != set(expected):
        raise ValueError("Original data file set differs from manifest")
    bad = []
    for rel, p in actual.items():
        row = expected[rel]
        if p.stat().st_size != row["bytes"] or sha(p) != row["sha256"]:
            bad.append(rel)
    if bad:
        raise ValueError(f"Original data size/hash mismatch: {bad[:5]}")
    summary = read_json(ROOT / "reports/data_summary.json")
    if sha(MANIFEST) != summary["manifest_sha256"] or len(expected) != 17668:
        raise ValueError("Unexpected manifest identity")
    return {"files": len(expected), "size_sha256_matched": len(actual), "missing": 0,
            "extra": 0, "mismatched": 0, "manifest_sha256": sha(MANIFEST)}


def val_paths():
    prov = read_json(VAL_DIR / "val_use_list_relative.provenance.json")
    if sha(VAL_LIST) != prov["relative_file_sha256"]:
        raise ValueError("val966 list hash mismatch")
    lines = VAL_LIST.read_text(encoding="utf-8").splitlines()
    if len(lines) != 966 or len(set(lines)) != 966:
        raise ValueError("val966 must contain 966 unique paths")
    paths = []
    for line in lines:
        if not line.startswith("data/val/images/") or len(PurePosixPath(line).parts) != 4:
            raise ValueError("Unexpected val966 path")
        paths.append(safe(ROOT / KIT / line))
    entries = manifest_entries()
    tr_sha = {v["sha256"] for k, v in entries.items() if k.startswith("train/images/")}
    all_val = {k for k in entries if k.startswith("val/images/")}
    omitted = {k for k in all_val if entries[k]["sha256"] in tr_sha}
    selected = {p.relative_to(ROOT / KIT / "data").as_posix() for p in paths}
    if len(omitted) != 16 or selected != all_val - omitted:
        raise ValueError("val966 selection is not exactly original val minus 16 known identical images")
    return paths


def parse_labels(text):
    labels = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError("YOLO row needs 5 fields")
        cls = int(parts[0])
        values = [Decimal(s) for s in parts[1:]]
        if cls not in range(4) or any(not v.is_finite() for v in values):
            raise ValueError("Invalid class/non-finite field")
        if any(v < 0 or v > 1 for v in values) or values[2] <= 0 or values[3] <= 0:
            raise ValueError("Invalid normalized xywh")
        # Preserve original decimal strings; do not silently rewrite coordinates.
        labels.append([cls, *parts[1:]])
    return labels


def pixel_bounds(box, size=SIZE):
    """Half-open rectangle: pixel center (i+.5) inside [left,right), [top,bottom)."""
    _, xs, ys, ws, hs = box
    x, y, w, h = map(Decimal, (xs, ys, ws, hs))
    half = Decimal("0.5")
    def bound(v):
        return min(size, max(0, int((v * size - half).to_integral_value(rounding=ROUND_CEILING))))
    return bound(x-w/2), bound(y-h/2), bound(x+w/2), bound(y+h/2)


def occupancy(labels, cls=None, size=SIZE):
    mask = np.zeros((size, size), dtype=bool)
    for b in labels:
        if cls is not None and b[0] != cls:
            continue
        x0, y0, x1, y1 = pixel_bounds(b, size)
        mask[y0:y1, x0:x1] = True
    return mask


def frequency(frames, size=SIZE):
    """All frames are denominator. Identical image hashes count once in unique map.

    If duplicate image files disagree in labels, union their masks for unique
    support and report this policy; never select a 'correct' annotation version.
    """
    counts = np.zeros((size, size), dtype=np.uint16)
    by_hash = {}
    for f in frames:
        m = occupancy(f["labels"], 3, size)
        counts += m
        key = f["image_sha256"]
        by_hash[key] = np.logical_or(by_hash.get(key, np.zeros_like(m)), m)
    unique = np.zeros_like(counts)
    for m in by_hash.values():
        unique += m
    return counts, unique, len(frames), len(by_hash)


def score(labels, counts, unique, size=SIZE):
    bus = occupancy(labels, 3, size)
    any_class = occupancy(labels, None, size)
    result = {}
    for prefix, weights in (("", counts), ("unique_", unique)):
        denominator = int(weights.sum())
        for name, mask in (("bus", bus), ("any_class", any_class)):
            result[prefix + name + "_uncovered"] = (
                float(int(weights[~mask].sum()) / denominator) if denominator else None)
    return result


def support_state(counts, unique, n, u):
    if not n:
        return "no_val_reference"
    if not counts.any():
        return "no_bus_reference"
    if u < 2:
        return "single_unique_image_not_repeat_evidence"
    if int(unique.max()) < 2:
        return "no_repeated_position"
    return "repeated_annotation_region_not_confirmed_static"


def load_inventory():
    entries = manifest_entries()
    trains = sorted(safe(ROOT / KIT / "data/train/images").iterdir())
    paths = [("train", p) for p in trains] + [("val966", p) for p in val_paths()]
    images, scenes = {}, {}
    for split, path in paths:
        path = safe(path)
        rel = path.relative_to(ROOT).as_posix()
        match = re.match(r"^(MVI_\d+)_img(\d+)_", path.stem)
        if not match:
            raise ValueError(f"Unknown image naming: {rel}")
        scene, frame = match.group(1), int(match.group(2))
        label_path = safe(path.parent.parent / "labels" / (path.stem + ".txt"))
        label_rel = label_path.relative_to(ROOT).as_posix()
        labels = parse_labels(label_path.read_text(encoding="utf-8"))
        image_sha = entries[path.relative_to(ROOT / KIT / "data").as_posix()]["sha256"]
        label_sha = entries[label_path.relative_to(ROOT / KIT / "data").as_posix()]["sha256"]
        ident = hashlib.sha256(f"{rel}\n{image_sha}\n{label_sha}".encode()).hexdigest()
        with Image.open(path) as img:
            if img.size != (SIZE, SIZE):
                raise ValueError(f"Non-640 input: {rel}")
        images[ident] = {"id": ident, "split": split, "scene": scene, "frame": frame,
                         "image_path": rel, "label_path": label_rel,
                         "image_sha256": image_sha, "label_sha256": label_sha,
                         "labels": labels, "bus_boxes": sum(b[0] == 3 for b in labels)}
        s = scenes.setdefault(scene, {"id": scene, "train_ids": [], "val_ids": []})
        s["train_ids" if split == "train" else "val_ids"].append(ident)
    for s in scenes.values():
        for field in ("train_ids", "val_ids"):
            s[field].sort(key=lambda i: (images[i]["frame"], images[i]["image_path"]))
        s["reference_set_sha256"] = digest([
            [i, images[i]["image_sha256"], images[i]["label_sha256"]] for i in s["val_ids"]])
        s["alignment"] = "unverified_no_registration_applied"
        s["train_images"] = len(s["train_ids"])
        s["val_images"] = len(s["val_ids"])
        for split, ids in (("train", s["train_ids"]), ("val", s["val_ids"])):
            s[split + "_bus_images"] = sum(images[i]["bus_boxes"] > 0 for i in ids)
            s[split + "_bus_boxes"] = sum(images[i]["bus_boxes"] for i in ids)
            groups = defaultdict(list)
            for i in ids:
                groups[images[i]["image_sha256"]].append(i)
            s[split + "_duplicate_hash_groups"] = [
                {"sha256": h, "ids": group,
                 "label_hashes": sorted({images[i]["label_sha256"] for i in group})}
                for h, group in sorted(groups.items()) if len(group) > 1]
    dataset_id = digest({"images": [[i, images[i]["image_sha256"], images[i]["label_sha256"]]
                                     for i in sorted(images)], "val_list_sha256": sha(VAL_LIST),
                         "algorithm": "bus-frequency-v1-pixel-center-half-open"})
    return {"schema": "bus-scene-v1", "dataset_id": dataset_id,
            "limits": LIMITS, "images": images, "scenes": [scenes[k] for k in sorted(scenes)]}


def encode_counts(counts):
    return np.stack([(counts & 255).astype(np.uint8), (counts >> 8).astype(np.uint8),
                     np.zeros_like(counts, dtype=np.uint8)], axis=-1)


def heat_image(counts, n):
    f = counts.astype(np.float64) / n if n else np.zeros_like(counts, dtype=float)
    rgba = np.zeros((*counts.shape, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = np.rint(185 * (1-f)).astype(np.uint8)
    rgba[..., 2] = 40
    # Purely visual sqrt alpha improves visibility; all score math uses raw counts.
    rgba[..., 3] = np.ceil(200 * np.sqrt(f)).astype(np.uint8)
    return rgba


def scene_arrays(inv, scene):
    return frequency([inv["images"][i] for i in scene["val_ids"]])


def describe_scene(s, counts, unique, n, u):
    s["val_unique_images"] = u
    s["max_pixel_support_frames"] = int(counts.max())
    s["max_pixel_support_unique_images"] = int(unique.max())
    s["reference_mass_pixel_frames"] = int(counts.sum())
    s["support_state"] = support_state(counts, unique, n, u)
    s["duplicate_unique_policy"] = "OR union of labels per identical image hash; not adjudicated truth"
    s["maps"] = ({k: f"reference_maps/{s['id']}.{suffix}.png" for k, suffix in
                  (("heat", "heat"), ("counts", "counts"), ("unique", "unique-counts"))} if n else {})


def candidate(inv, s, ident, counts, unique):
    im = inv["images"][ident]
    return {"id": ident, "train_path": im["image_path"], "scene": s["id"],
            "image_sha256": im["image_sha256"], "label_sha256": im["label_sha256"],
            "reference_set_sha256": s["reference_set_sha256"],
            "reference_scene_id": s["id"], "alignment": s["alignment"],
            "support_state": s["support_state"], **score(im["labels"], counts, unique)}


def recall_analysis():
    baseline = read_json(BASELINE)
    rows = []
    for t, v in zip(baseline["class_metrics"]["train"], baseline["class_metrics"]["val"], strict=True):
        if t["label"] != v["label"]:
            raise ValueError("Class streams misaligned")
        name = NAMES[t["label"]] if t["label"] < 4 else "all"
        rows.append({"class": name, "train": t, "val966": v,
                     "recall_gap_percentage_points": 100*(t["recall"]-v["recall"]),
                     "AP50_gap_percentage_points": 100*(t["mAP"]-v["mAP"])})
    return {"source": BASELINE.relative_to(ROOT).as_posix(), "source_sha256": sha(BASELINE),
            "phase": "final train collection / final best val966, not epoch CSV",
            "selected_class": "bus", "selected_by": "learner", "limits": LIMITS, "rows": rows}


def feedback_identity(c):
    return {k: c[k] for k in ("id", "train_path", "image_sha256", "label_sha256", "reference_set_sha256")}


def validate_feedback(payload, candidates, dataset_id):
    if not isinstance(payload, dict) or payload.get("schema") != "bus-feedback-v1" or payload.get("dataset_id") != dataset_id:
        raise ValueError("Feedback schema/dataset identity mismatch")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("Feedback records must be a list")
    allowed = {c["id"]: c for c in candidates}
    seen = set()
    for r in records:
        if not isinstance(r, dict) or r.get("id") not in allowed or r["id"] in seen:
            raise ValueError("Unknown or duplicate review ID")
        seen.add(r["id"])
        if any(r.get(k) != v for k, v in feedback_identity(allowed[r["id"]]).items()):
            raise ValueError("Feedback source identity mismatch")
        status, count, note = r.get("status"), r.get("missing_count"), r.get("note")
        if status not in STATUSES or not isinstance(note, str) or len(note) > 10000:
            raise ValueError("Invalid feedback status/note")
        if count is not None and (type(count) is not int or count < 1 or count > 9007199254740991 or status != "missing"):
            raise ValueError("Optional count must be a positive safe integer, only for confirmed missing")
        if "missing_count" not in r:
            raise ValueError("Explicit null required for unknown missing count")
    return records


def summarize_feedback(payload, candidates, dataset_id):
    records = validate_feedback(payload, candidates, dataset_id)
    by_id = {r["id"]: r for r in records}
    groups = defaultdict(Counter)
    totals = Counter()
    for c in candidates:
        r = by_id.get(c["id"], {"status": "unreviewed", "missing_count": None})
        for stats in (totals, groups[c["scene"]]):
            stats["total"] += 1
            stats[r["status"]] += 1
            if r["status"] != "unreviewed":
                stats["reviewed"] += 1
            if r["status"] == "missing":
                if r["missing_count"] is None:
                    stats["confirmed_images_without_box_count"] += 1
                else:
                    stats["confirmed_images_with_box_count"] += 1
                    stats["reported_missing_boxes_partial"] += r["missing_count"]
    def finish(s):
        result = {k: s[k] for k in (*STATUSES, "total", "reviewed", "confirmed_images_without_box_count",
                                    "confirmed_images_with_box_count", "reported_missing_boxes_partial")}
        result["missing_over_reviewed"] = s["missing"] / s["reviewed"] if s["reviewed"] else None
        result["ratio_denominator_includes_uncertain_and_mismatch"] = True
        return result
    return {"dataset_id": dataset_id, "unique_train_path_unit": True,
            "total": finish(totals), "by_scene": {k: finish(v) for k, v in sorted(groups.items())},
            "limits": LIMITS, "missing_box_total_is_unknown_if_any_confirmed_count_blank": True}


def render_page(inv, candidates):
    packed = canonical({"inventory": inv, "candidates": candidates, "statuses": STATUSES,
                        "names": NAMES}).replace("<", "\\u003c").replace("&", "\\u0026")
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count("__AUDIT_DATA__") != 1:
        raise ValueError("Expected one template data placeholder")
    return template.replace("__AUDIT_DATA__", packed)


def source_hashes():
    paths = [ROOT / "tools/bus_static_audit.py", TEMPLATE, ROOT / "tests/test_bus_static_audit.py"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def artifact_hashes():
    # Only generated files; user feedback JSON in this directory is not an artifact.
    paths = [OUT / name for name in ("README.md", "scene_manifest.json", "candidates.json",
             "build_config.json", "recall_analysis.json", "RECALL_ANALYSIS.md", "review.html")]
    paths += list((OUT / "reference_maps").glob("*.png"))
    return {p.relative_to(OUT).as_posix(): sha(p) for p in sorted(paths)}


def summary_inventory(inv, candidates):
    scenes = inv["scenes"]
    return {"train_images": len(candidates), "val966_images": sum(s["val_images"] for s in scenes),
            "MVI_groups": len(scenes), "shared_MVI_groups": sum(bool(s["train_images"] and s["val_images"]) for s in scenes),
            "val_bus_images": sum(s["val_bus_images"] for s in scenes),
            "val_bus_boxes": sum(s["val_bus_boxes"] for s in scenes),
            "computed_train_scores": sum(c["bus_uncovered"] is not None for c in candidates),
            "score_not_computable": sum(c["bus_uncovered"] is None for c in candidates),
            "support_states": dict(Counter(s["support_state"] for s in scenes)),
            "human_reviews_created": 0}


def build():
    if OUT.exists():
        raise FileExistsError("Output exists; refusing overwrite. Use verify or a separately reviewed edit.")
    before_git = git_snapshot()
    if before_git["branch"] != "main":
        raise ValueError("This approved build is main-only")
    before = integrity()
    print("Original data integrity before: 17668/17668", flush=True)
    inv = load_inventory()
    OUT.mkdir()
    (OUT / "reference_maps").mkdir()
    candidates = []
    for s in inv["scenes"]:
        counts, unique, n, u = scene_arrays(inv, s)
        describe_scene(s, counts, unique, n, u)
        if n:
            for key, arr in (("heat", heat_image(counts, n)), ("counts", encode_counts(counts)), ("unique", encode_counts(unique))):
                Image.fromarray(arr).save(OUT / s["maps"][key])
        candidates.extend(candidate(inv, s, i, counts, unique) for i in s["train_ids"])
    recall = recall_analysis()
    config = {"schema": "bus-build-v1", "dataset_id": inv["dataset_id"], "canvas": [SIZE, SIZE],
              "bus_class_id": 3, "class_names": NAMES, "val_list_sha256": sha(VAL_LIST),
              "frame_frequency": "bus mask count / ALL val966 scene frames, including frames without bus",
              "unique_frequency": "union masks within identical image SHA256; denominator all unique scene images",
              "raster_rule": "pixel center in half-open xyxy, Decimal arithmetic; clip only rendering bounds",
              "score": "sum(reference_count * NOT train_coverage) / sum(reference_count); null if zero mass",
              "maps_count_encoding": "RGB: R + 256*G; B=0; counts and unique maps separate",
              "heat_rendering": "R=255,G=round(185*(1-F)),B=40,alpha=ceil(200*sqrt(F)); visual only",
              "no_hard_thresholds": True, "no_registration": True, "no_sampling": True,
              "initial_order": "MVI then frame number then path", "input_sources": [
                  MANIFEST.relative_to(ROOT).as_posix(), VAL_LIST.relative_to(ROOT).as_posix(),
                  BASELINE.relative_to(ROOT).as_posix(), f"{KIT}/data/train", f"{KIT}/data/val (val966 only for analysis; all val hashes for integrity)"],
              "versions": {"python": sys.version.split()[0], "executable": sys.executable,
                           "Pillow": PIL_VERSION, "numpy": np.__version__}, "limits": LIMITS}
    for name, obj in (("scene_manifest.json", inv), ("candidates.json", {"dataset_id": inv["dataset_id"], "records": candidates}),
                      ("recall_analysis.json", recall), ("build_config.json", config)):
        write_json(OUT / name, obj)
    lines = ["# Baseline Recall 对比", "", "同一模型收尾 train 收集与 best val966 验证；不是训练期 loss。", "",
             "| 类别 | Train Recall | Val966 Recall | 差距（百分点） |", "|---|---:|---:|---:|"]
    lines += [f"| {r['class']} | {r['train']['recall']:.8f} | {r['val966']['recall']:.8f} | {r['recall_gap_percentage_points']:.6f} |" for r in recall["rows"]]
    lines += ["", "差值按原始精度计算；Recall 两列为0–1，差值为百分点。", "", *["- " + x for x in LIMITS]]
    (OUT / "RECALL_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "review.html").write_text(render_page(inv, candidates), encoding="utf-8")
    readme = r'''# Bus 位置频次审核工具

## 打开与审核

打开同目录 `review.html`（建议使用 PI-Desktop 的本地预览）。保留本项目目录结构，原图通过相对路径加载，页面没有复制原图或上传功能。

1. 自行选择 MVI 场景，默认按编号排列，未预选可疑组。MVI/帧号只是命名线索。
2. 先看 train 原图与四类原框。需要时勾选热图、打开全部 val966 参照或 train 联系图；可放大。
3. 选择结论、可选缺框数与备注，点击“保存本张意见”。导航时如有未保存输入会提示保存。
4. 意见存到当前浏览器本地存储，刷新可以恢复；定期手动“导出审核 JSON”备份并交回。移动页面/更换浏览器可能改变存储环境；可手动导入旧反馈恢复。
5. 不要求审核全部图片。没有 bus 原标签的 train 图片没有被过滤，零参考图片保留为不可计算。

`?test=1` 是独立存储的测试模式，不用于真实审核。正式页面初始没有 AI 预填结论。

## 证据含义

- `scene_manifest.json`：全部场景、每图原坐标文本、类别、帧号及图片/标签SHA256；val只取固定966张。
- `candidates.json`：7852个唯一train路径的记录，稳定ID由路径与图片/标签哈希组成；每条引用场景参考集合哈希，不是建议新增框。
- `reference_maps/`：每场景热图、支持帧计数图、不同图片哈希支持计数图；没有val的组不生成图。
- F(p)分母为该场景全部val966帧，包括无bus帧；同一帧重叠只计一次。
- 像素中心`(x+.5,y+.5)`位于原框半开区域`[left,right) × [top,bottom)`才计入。使用Decimal计算边界；仅画布裁切，不回写框。
- 不同图支持：相同图片SHA256归为一个身份，重复图标签若冲突则取并集，仅用于支持描述，不选择正确标签。场景中保存重复组与其标签哈希。
- 计数PNG为RGB，像素计数=`R+256*G`，B=0；heat PNG透明度为`ceil(200*sqrt(F))`，只是显示增强。页面热图开启后悬停显示实际计数、分母、频次及不同图支持。某些浏览器直接用file打开会禁止Canvas读像素，届时页面明确提示；可改用PI本地预览，仍可看原图/热图及场景级计数。
- 分数=`sum(参考计数 × 未覆盖mask)/sum(参考计数)`，保留bus/任意类别覆盖及不同图支持的四个值；参考质量为0时为null，不是0。按分数排序不截取前N，不存在静态频次/聚类/配准阈值。
- 重复位置不是同一辆静止车的证明；机位均未核实，没有自动配准。你可以标记机位/位置不匹配。

## 汇总规则与限制

按唯一train路径计数，确认漏标/已有标注/无可确认bus/不确定/机位不匹配分别报告，未审不算无问题。命中比例分母为全部已审图片（含不确定及机位不匹配），同时显示明确分子分母。缺框数留空是未知，只报告已填数小计，不补成0。

本工具验证的是“这种检索是否能找到漏标”，不能证明比随机抽查更高效、不能证明漏标导致Recall差距，也不估计全量漏标率。val信息参与检索，不把后续相同val改善解释成独立泛化证据；实际修标及竞赛训练用途另行确认。无模型、网络调用、训练/推理或标签回写。

## 可复现命令（项目根目录，PowerShell）

```powershell
$py = 'E:\anaconda\envs\dl-3lc\python.exe'
& $py -I -B -m unittest discover -s .\tests -p test_bus_static_audit.py -v
& $py -I -B .\tools\bus_static_audit.py verify
# 仅第一次构建用：输出目录已存在时会拒绝覆盖；不删除现有产物或意见。
& $py -I -B .\tools\bus_static_audit.py build
# 若首次构建已产出完整文件但在写verification前中断：仅创建缺失核验，不覆盖产物。
& $py -I -B .\tools\bus_static_audit.py finalize
# 审核后，把你明确指定的反馈放在项目内；以下只向stdout汇总，不改文件。
& $py -I -B .\tools\bus_static_audit.py summarize .\reports\bus-static-review\你的反馈.json
```

verify核验原train/val17668文件集合/尺寸/SHA256，并重算全部场景地图、候选分数、输入身份、页面与Recall。被排除的16张val只参与完整性哈希核对，不参与内容分析或页面。

反馈导入先全量校验批次、ID、路径及来源哈希，重复ID/非法状态或缺框数导致整批拒绝；有冲突需你确认才替换，不自动去重掩盖错误。没有执行任何实际修标。

核验结果见`verification.json`；浏览器检查完成情况另见`browser_checks.json`（如存在），不把未执行检查视为通过。
'''
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    after = integrity()
    if before != after:
        raise ValueError("Data changed during build")
    report = {"schema": "bus-verification-v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "git_before": before_git, "git_after": git_snapshot(), "integrity_before": before,
              "integrity_after": after, "summary": summary_inventory(inv, candidates),
              "source_sha256": source_hashes(), "artifact_sha256": artifact_hashes(),
              "browser_checks": "not_run_yet", "unit_tests": "recorded separately after actual execution",
              "no_model_execution": True, "no_label_writes": True, "original_data_copied": False}
    write_json(OUT / "verification.json", report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


def verify(finalize=False):
    before_git = git_snapshot()
    if finalize:
        if (OUT / "verification.json").exists():
            raise FileExistsError("Verification exists; finalize will not overwrite it")
        if before_git["branch"] != "main":
            raise ValueError("This approved finalization is main-only")
        preserved = artifact_hashes()
        report = None
    else:
        report = read_json(OUT / "verification.json")
        if artifact_hashes() != report["artifact_sha256"]:
            raise ValueError("Generated artifact set/hash differs from verification snapshot")
        if source_hashes() != report["source_sha256"]:
            raise ValueError("Source hash differs from verification snapshot")
    current_integrity = integrity()
    print("Original data integrity: 17668/17668", flush=True)
    if report and (current_integrity != report["integrity_before"] or current_integrity != report["integrity_after"]):
        raise ValueError("Original data no longer matches build")
    inv = load_inventory()
    expected_candidates = []
    maps_checked = 0
    for s in inv["scenes"]:
        counts, unique, n, u = scene_arrays(inv, s)
        describe_scene(s, counts, unique, n, u)
        if n:
            for key, arr in (("heat", heat_image(counts, n)), ("counts", encode_counts(counts)), ("unique", encode_counts(unique))):
                with Image.open(OUT / s["maps"][key]) as img:
                    if not np.array_equal(np.array(img), arr):
                        raise ValueError("Map content differs from source labels")
                maps_checked += 1
        expected_candidates.extend(candidate(inv, s, i, counts, unique) for i in s["train_ids"])
    if inv != read_json(OUT / "scene_manifest.json"):
        raise ValueError("Scene manifest differs from sources")
    if {"dataset_id": inv["dataset_id"], "records": expected_candidates} != read_json(OUT / "candidates.json"):
        raise ValueError("Candidates differ from recomputation")
    if recall_analysis() != read_json(OUT / "recall_analysis.json"):
        raise ValueError("Recall analysis differs from archived source")
    if (OUT / "review.html").read_text(encoding="utf-8") != render_page(inv, expected_candidates):
        raise ValueError("Review page differs from source template/data")
    ids = [c["id"] for c in expected_candidates]
    paths = [c["train_path"] for c in expected_candidates]
    if len(ids) != len(set(ids)) or len(paths) != len(set(paths)) or len(ids) != 7852:
        raise ValueError("Candidate uniqueness/coverage error")
    result = {"verified": True, "integrity": current_integrity, "maps_recomputed": maps_checked,
              "candidates_recomputed": len(ids), "summary": summary_inventory(inv, expected_candidates)}
    if finalize:
        after = integrity()
        if current_integrity != after or preserved != artifact_hashes():
            raise ValueError("Original data/artifacts changed during interrupted-build finalization")
        report = {"schema": "bus-verification-v1", "created_at": datetime.now(timezone.utc).isoformat(),
                  "operation": "finalize_preserved_outputs_after_300_second_build_timeout",
                  "initial_attempt": "Timed out after artifacts were generated, no verification written; no original files changed by tool",
                  "git_before": before_git, "git_after": git_snapshot(),
                  "integrity_before": current_integrity, "integrity_after": after,
                  "summary": result["summary"], "recomputation": result,
                  "source_sha256": source_hashes(), "artifact_sha256": artifact_hashes(),
                  "browser_checks": "not_finished_yet", "unit_tests": "record after actual run",
                  "no_model_execution": True, "no_label_writes": True, "original_data_copied": False}
        write_json(OUT / "verification.json", report)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    sub.add_parser("verify")
    sub.add_parser("finalize", help="Recover interrupted build: recompute and preserve outputs, create missing verification only")
    p = sub.add_parser("summarize")
    p.add_argument("feedback", help="Explicit in-project user feedback JSON; no labels are written")
    args = parser.parse_args()
    if args.command == "build":
        build()
    elif args.command == "verify":
        verify()
    elif args.command == "finalize":
        verify(finalize=True)
    else:
        c = read_json(OUT / "candidates.json")
        report = summarize_feedback(read_json(safe(args.feedback)), c["records"], c["dataset_id"])
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
