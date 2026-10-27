"""Synthetic tests only. No original data/model/network operations."""
import ast
import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bus_static_audit", ROOT / "tools/bus_static_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
FULL_BUS = [3, "0.5", "0.5", "1", "1"]
LEFT_BUS = [3, "0.25", "0.5", "0.5", "1"]
RIGHT_BUS = [3, "0.75", "0.5", "0.5", "1"]


def frame(labels, key):
    return {"labels": labels, "image_sha256": key}


def fake_candidate(i, scene="MVI_FAKE"):
    return {"id": str(i), "train_path": f"train/{i}.jpg", "image_sha256": f"im{i}",
            "label_sha256": f"label{i}", "reference_set_sha256": "ref", "scene": scene}


def record(c, status="missing", count=None, note=""):
    return {**audit.feedback_identity(c), "status": status, "missing_count": count, "note": note}


def payload(records):
    return {"schema": "bus-feedback-v1", "dataset_id": "fixture", "records": records}


class FrequencyTests(unittest.TestCase):
    def test_overlapping_boxes_count_once_in_frame(self):
        c, u, n, z = audit.frequency([frame([FULL_BUS, LEFT_BUS, FULL_BUS], "a")], 4)
        np.testing.assert_array_equal(c, np.ones((4, 4), dtype=np.uint16))
        self.assertEqual((n, z), (1, 1))
        np.testing.assert_array_equal(c, u)

    def test_denominator_includes_no_bus_frame(self):
        c, u, n, z = audit.frequency([frame([FULL_BUS], "a"), frame([], "b")], 4)
        self.assertEqual((n, z), (2, 2))
        self.assertEqual(float(c[0, 0] / n), .5)

    def test_duplicate_image_not_independent_support(self):
        c, u, n, z = audit.frequency([frame([FULL_BUS], "same"), frame([FULL_BUS], "same")], 4)
        self.assertEqual((int(c.max()), int(u.max()), n, z), (2, 1, 2, 1))
        self.assertEqual(audit.support_state(c, u, n, z), "single_unique_image_not_repeat_evidence")

    def test_duplicate_label_conflict_unique_uses_union(self):
        c, u, n, z = audit.frequency([frame([LEFT_BUS], "same"), frame([RIGHT_BUS], "same")], 4)
        np.testing.assert_array_equal(u, np.ones((4, 4), dtype=np.uint16))
        self.assertEqual(z, 1)

    def test_single_frame_not_repeat_evidence(self):
        c, u, n, z = audit.frequency([frame([FULL_BUS], "a")], 4)
        self.assertEqual(audit.support_state(c, u, n, z), "single_unique_image_not_repeat_evidence")

    def test_many_frames_but_disjoint_positions_not_repeat(self):
        c, u, n, z = audit.frequency([frame([LEFT_BUS], "a"), frame([RIGHT_BUS], "b")], 4)
        self.assertEqual(audit.support_state(c, u, n, z), "no_repeated_position")

    def test_repeated_annotation_not_asserted_static(self):
        c, u, n, z = audit.frequency([frame([FULL_BUS], "a"), frame([FULL_BUS], "b")], 4)
        self.assertEqual(audit.support_state(c, u, n, z), "repeated_annotation_region_not_confirmed_static")

    def test_no_reference_states(self):
        c, u, n, z = audit.frequency([], 4)
        self.assertEqual(audit.support_state(c, u, n, z), "no_val_reference")
        c, u, n, z = audit.frequency([frame([], "a")], 4)
        self.assertEqual(audit.support_state(c, u, n, z), "no_bus_reference")

    def test_pixel_center_half_open_exact_boundary(self):
        # x interval [.5,2.5): centers .5,1.5 included, 2.5 excluded.
        b = [3, ".375", ".5", ".5", "1"]
        self.assertEqual(audit.pixel_bounds(b, 4), (0, 0, 2, 4))

    def test_render_clipping_does_not_change_original(self):
        b = [3, ".99", ".5", ".1", "1"]
        old = b.copy()
        self.assertEqual(audit.pixel_bounds(b, 640)[2], 640)
        self.assertEqual(b, old)

    def test_count_encoding_is_lossless(self):
        c = np.array([[0, 1, 256, 1000, 65535]], dtype=np.uint16)
        rgb = audit.encode_counts(c)
        decoded = rgb[..., 0].astype(np.uint16) + 256 * rgb[..., 1].astype(np.uint16)
        np.testing.assert_array_equal(decoded, c)


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.c, self.u, _, _ = audit.frequency([frame([FULL_BUS], "a")], 4)

    def test_empty_train_labels_retained_as_candidate(self):
        c = fake_candidate(1)
        im = {"image_path": c["train_path"], "image_sha256": c["image_sha256"],
              "label_sha256": c["label_sha256"], "labels": []}
        # Public scorer exercises empty-label semantics; inventory never filters on bus count.
        s = audit.score(im["labels"], self.c, self.u, 4)
        self.assertEqual(s["bus_uncovered"], 1)
        self.assertEqual(s["any_class_uncovered"], 1)

    def test_other_class_not_misrepresented_as_unlabeled(self):
        other = [1, *FULL_BUS[1:]]
        s = audit.score([other], self.c, self.u, 4)
        self.assertEqual(s["bus_uncovered"], 1)
        self.assertEqual(s["any_class_uncovered"], 0)

    def test_score_zero_one_half_and_null(self):
        self.assertEqual(audit.score([FULL_BUS], self.c, self.u, 4)["bus_uncovered"], 0)
        self.assertEqual(audit.score([LEFT_BUS], self.c, self.u, 4)["bus_uncovered"], .5)
        empty = np.zeros_like(self.c)
        self.assertTrue(all(v is None for v in audit.score([], empty, empty, 4).values()))

    def test_weighted_score_uses_counts_not_binary_union(self):
        c = np.array([[3, 1], [3, 1]], dtype=np.uint16)
        s = audit.score([[3, ".25", ".5", ".5", "1"]], c, c, 2)
        self.assertEqual(s["bus_uncovered"], .25)


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.cs = [fake_candidate(i, "MVI_A" if i < 3 else "MVI_B") for i in range(5)]

    def test_empty_feedback_has_no_fake_negative(self):
        s = audit.summarize_feedback(payload([]), self.cs, "fixture")["total"]
        self.assertEqual((s["reviewed"], s["unreviewed"], s["no_bus"]), (0, 5, 0))
        self.assertIsNone(s["missing_over_reviewed"])

    def test_unknown_count_is_not_zero(self):
        p = payload([record(self.cs[0]), record(self.cs[1], "uncertain"), record(self.cs[2], "scene_mismatch"),
                     record(self.cs[3], "missing", 2), record(self.cs[4], "unreviewed", note="draft")])
        s = audit.summarize_feedback(p, self.cs, "fixture")
        self.assertEqual(s["total"]["missing_over_reviewed"], .5)
        self.assertEqual(s["total"]["confirmed_images_without_box_count"], 1)
        self.assertEqual(s["total"]["reported_missing_boxes_partial"], 2)
        self.assertEqual(s["total"]["reviewed"], 4)
        self.assertEqual(s["by_scene"]["MVI_A"]["reviewed"], 3)

    def test_duplicate_review_rejected_not_double_counted(self):
        r = record(self.cs[0])
        with self.assertRaises(ValueError):
            audit.validate_feedback(payload([r, r]), self.cs, "fixture")

    def test_bad_batch_and_wrong_source_rejected(self):
        p = payload([record(self.cs[0])])
        p["dataset_id"] = "other"
        with self.assertRaises(ValueError):
            audit.validate_feedback(p, self.cs, "fixture")
        p = payload([record(self.cs[0])])
        p["records"][0]["label_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            audit.validate_feedback(p, self.cs, "fixture")

    def test_counts_validation(self):
        for bad in (0, -1, True, 1.5, "2", 9007199254740992):
            with self.subTest(count=bad), self.assertRaises(ValueError):
                audit.validate_feedback(payload([record(self.cs[0], count=bad)]), self.cs, "fixture")
        with self.assertRaises(ValueError):
            audit.validate_feedback(payload([record(self.cs[0], "uncertain", 2)]), self.cs, "fixture")

    def test_whole_batch_validation_has_no_mutation(self):
        p = payload([record(self.cs[0]), record(self.cs[1])])
        p["records"][1]["id"] = "unknown"
        old = copy.deepcopy(p)
        with self.assertRaises(ValueError):
            audit.validate_feedback(p, self.cs, "fixture")
        self.assertEqual(p, old)

    def test_all_statuses_and_chinese_note_roundtrip(self):
        c = self.cs[0]
        for status in audit.STATUSES:
            p = payload([record(c, status, note="遮挡/边界：待人工确认")])
            self.assertEqual(audit.validate_feedback(p, self.cs, "fixture"), p["records"])


class SafetyTests(unittest.TestCase):
    def test_forbidden_paths_rejected_without_read(self):
        for p in (ROOT.parent, ROOT / ".worktrees/anything", ROOT / audit.KIT / "data/test/images"):
            with self.subTest(path=p), self.assertRaises(ValueError):
                audit.safe(p)

    def test_parse_preserves_decimal_text(self):
        self.assertEqual(audit.parse_labels("3 0.500000 0.5 0.25 0.100000\n")[0][1:],
                         ["0.500000", "0.5", "0.25", "0.100000"])
        self.assertEqual(audit.parse_labels("\n"), [])
        for line in ("4 .5 .5 .1 .1", "3 NaN .5 .1 .1", "3 .5 .5 0 .1", "3 .5 .5 .1"):
            with self.assertRaises((ValueError, ArithmeticError)):
                audit.parse_labels(line)

    def test_no_model_or_network_imports(self):
        source = (ROOT / "tools/bus_static_audit.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imports.extend(a.name.split(".")[0] for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                imports.append(n.module.split(".")[0])
        self.assertFalse(set(imports) & {"torch", "tlc", "ultralytics", "requests", "http", "socket", "urllib"})
        template = audit.TEMPLATE.read_text(encoding="utf-8")
        self.assertEqual(template.count("__AUDIT_DATA__"), 1)
        self.assertIn("connect-src 'none'", template)
        self.assertNotIn("fetch(", template)
        self.assertNotIn("XMLHttpRequest", template)


if __name__ == "__main__":
    unittest.main()
