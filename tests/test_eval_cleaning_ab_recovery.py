import importlib.util
from pathlib import Path
import tempfile
import unittest
import json
import hashlib

ROOT=Path(__file__).resolve().parents[1]


class RecoveryTests(unittest.TestCase):
    def tool(self):
        spec=importlib.util.spec_from_file_location('recovery_test',ROOT/'tools/eval_cleaning_ab_recovery.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

    def test_weight_uses_recorded_path_and_checks_bytes(self):
        m=self.tool()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root=Path(temp);m.SOURCE=root
            weight=root/'A/yolov8n/weights/best.pt';weight.parent.mkdir(parents=True);weight.write_bytes(b'fixture')
            result=root/'A/train/result.json';result.parent.mkdir(parents=True)
            result.write_text(json.dumps({'state':'completed','epochs':28,'best':str(weight),'best_sha256':hashlib.sha256(b'fixture').hexdigest()}))
            self.assertEqual(m.weight_path('A'),weight)
            weight.write_bytes(b'changed')
            with self.assertRaises(RuntimeError):m.weight_path('A')

    def test_legacy_source_is_read_only(self):
        m=self.tool()
        self.assertTrue(m.allowed_file(m.SOURCE/'A/train/result.json',False))
        self.assertFalse(m.allowed_file(m.SOURCE/'A/yolov8n/weights/best.pt',True))
        self.assertTrue(m.allowed_file(m.OUT/'runtime/cache',True))
        self.assertFalse(m.allowed_file(ROOT/'unrelated.txt',False))

    def test_options_match_shared_fp32_evaluation(self):
        m=self.tool();opts=m.evaluation_options()
        self.assertEqual((opts['imgsz'],opts['batch'],opts['workers']),(640,32,2))
        self.assertEqual((opts['conf'],opts['iou'],opts['max_det']),(.001,.7,300))
        self.assertIsNone(opts['quantize']);self.assertFalse(opts['rect']);self.assertFalse(opts['augment'])

    def test_local_asset_resolver_never_downloads_missing_or_unapproved_weights(self):
        m=self.tool()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            p=Path(temp)/'best.pt';p.write_bytes(b'fixture')
            resolver=m.local_assets({str(p.resolve()):m.sha(p)})
            self.assertEqual(resolver(str(p)),str(p.resolve()))
            with self.assertRaises(RuntimeError):resolver(str(p.parent/'absent.pt'))
            p.write_bytes(b'drift')
            with self.assertRaises(RuntimeError):resolver(str(p))

    def test_owned_source_loader_does_not_probe_pycache(self):
        from unittest.mock import patch
        m=self.tool()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            path=Path(temp)/'owned.py';path.write_text('value = 7\n')
            with patch('importlib.machinery.SourceFileLoader.get_code', side_effect=AssertionError('No cache loader')):
                self.assertEqual(m.module('owned',path).value,7)

    def test_guard_rejects_source_mutations_before_syscall(self):
        m=self.tool()
        for event,args in [('os.remove',(str(m.SOURCE/'data/x.jpg'),-1)),
                           ('os.rename',(str(m.OUT/'x'),str(m.SOURCE/'x'),-1,-1)),
                           ('os.mkdir',(str(m.SOURCE/'x'),511,-1)),
                           ('os.rmdir',(str(m.SOURCE/'data'),-1))]:
            with self.subTest(event=event),self.assertRaises(PermissionError):m.check_mutation(event,args)
        m.check_mutation('os.rename',(str(m.OUT/'x'),str(m.OUT/'y'),-1,-1))


if __name__=='__main__':unittest.main()
