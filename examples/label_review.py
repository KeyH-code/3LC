"""Run the label review workflow on supplied synthetic boxes."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from tools import oof_label_cleaning as cleaning


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("demo-output"))
    args = parser.parse_args()

    example = Path(__file__).resolve().parent
    original_label = example / "sample.txt"
    raw = original_label.read_bytes()
    lines = raw.decode("utf-8").splitlines()
    fixture = json.loads((example / "sample.json").read_text(encoding="utf-8"))
    source = {
        "id": "sample",
        "scene": "synthetic",
        "frame": 1,
        "image_path": "sample.jpg",
        "image_sha256": None,
        "label_path": original_label.name,
        "label_sha256": hashlib.sha256(raw).hexdigest(),
        "labels": [line.split() for line in lines],
    }
    predictions = {"candidates": fixture["predictions"], "fold": 0, "model_sha256": None}
    candidates = cleaning.signals(source, predictions, fixture["confidence_min"])

    review = fixture["review"]
    if {row["prediction_id"] for row in candidates} != review.keys():
        raise ValueError("Generated candidates do not match the example review decisions.")
    decisions = []
    for candidate in candidates:
        if "gt_index" in candidate:
            candidate["original_label_line"] = lines[candidate["gt_index"]]
        decisions.append({
            **review[candidate["prediction_id"]],
            "key": candidate["key"],
            "source": "synthetic_review",
            "evidence": [],
        })

    changes, deferred = cleaning.proposal(candidates, decisions)
    revised = cleaning.edited_label(raw, changes)
    try:
        args.output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("Choose a new output directory with --output.")

    (args.output / "revised.txt").write_bytes(revised)
    counts = Counter(change["action"] for change in changes)
    summary = {
        "candidates": len(candidates),
        "changed_classes": counts["change_class"],
        "added_boxes": counts["add_box"],
        "kept": sum(row["decision"] == "keep" for row in decisions),
        "deferred": len(deferred),
        "original_unchanged": original_label.read_bytes() == raw,
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
