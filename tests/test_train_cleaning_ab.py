"""Synthetic tests only: no torch imports, models, datasets, or GPU work."""
import importlib.util
from pathlib import Path
import unittest
import tempfile
import copy

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'tools/train_cleaning_ab.py'


class Contract(unittest.TestCase):
    def tool(self):
        self.assertTrue(PATH.is_file(), 'Approved paired experiment executor not implemented yet')
        spec = importlib.util.spec_from_file_location('cleaning_ab_test', PATH)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module

    def test_ss04_paired_configuration(self):
        m = self.tool(); a = m.training_options()
        self.assertEqual((a['epochs'], a['batch'], a['imgsz'], a['seed']), (28, 32, 640, 0))
        self.assertEqual((a['patience'], a['mosaic'], a['mixup'], a['copy_paste']), (100, 1., .05, .1))
        self.assertFalse(a['pretrained']); self.assertFalse(a['resume'])
        self.assertEqual(a['close_mosaic'], 10)

    def test_val_requires_exactly_966_unique_entries(self):
        m = self.tool()
        m.validate_val([str(i) for i in range(966)])
        for vals in [[], [str(i) for i in range(965)], ['same'] * 966,
                     [str(i) for i in range(966)] + ['0']]:
            with self.subTest(length=len(vals)), self.assertRaises(ValueError):
                m.validate_val(vals)

    def test_train_pair_rejects_identity_or_bytes_drift(self):
        m = self.tool()
        source = [dict(id=str(i), image_sha256=str(i)) for i in range(7852)]
        self.assertEqual(len(m.validate_pair(source, copy.deepcopy(source))), 7852)
        for kind in ['short', 'duplicate', 'bytes']:
            other = copy.deepcopy(source)
            if kind == 'short': other.pop()
            elif kind == 'duplicate': other[-1] = other[0].copy()
            else: other[-1]['image_sha256'] = 'changed'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                m.validate_pair(source, other)

    def test_artifacts_cannot_be_overwritten(self):
        m = self.tool()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            p = Path(temp)/'evidence.json'
            m.save(p, {'first': True})
            with self.assertRaises(FileExistsError): m.save(p, {'first': False})
            self.assertEqual(m.load(p), {'first': True})

    def test_stage_budget_never_exceeds_total_deadline(self):
        m = self.tool()
        self.assertEqual(m.stage_timeout('train', 1000, 990), 10)
        self.assertEqual(m.stage_timeout('train', 10000, 0), 5400)
        self.assertEqual(m.stage_timeout('evaluate', 10000, 0), 3600)
        with self.assertRaises(TimeoutError): m.stage_timeout('train', 100, 100)

    def test_fixed_metrics_counts_empty_and_populated_classes(self):
        import numpy as np
        m = self.tool()
        stats = {'tp': [np.array([[True]*10, [False]*10])],
                 'conf': [np.array([.9, .3])], 'pred_cls': [np.array([1, 1])],
                 'target_cls': [np.array([1, 1, 2])]}
        result = m.fixed_metrics(stats)
        car = result['classes'][1]
        self.assertEqual((car['TP'], car['FP'], car['FN']), (1, 1, 1))
        self.assertEqual((car['precision'], car['recall']), (.5, .5))
        self.assertEqual(result['classes'][2]['FN'], 1)

    def test_scored_validator_factory_captures_instance(self):
        m = self.tool(); capture = {}
        class Base:
            def __init__(self, **kwargs): self.kwargs = kwargs
        factory = m.scored_validator_factory(Base, capture)
        instance = factory(args={'plots': False})
        self.assertIs(capture['validator'], instance)
        self.assertEqual(instance.kwargs['args'], {'plots': False})

    def test_effective_options_reject_optimizer_or_amp_drift(self):
        m = self.tool(); args = m.training_options()
        args['warmup_bias_lr']=0.
        m.assert_training_contract(args, 'AdamW', [.00125], (.9, .999), True)
        for key, value in [('batch', 16), ('mixup', 0.), ('box', 3.)]:
            changed = args.copy(); changed[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                m.assert_training_contract(changed, 'AdamW', [.00125], (.9, .999), True)
        for opt, lr, beta, amp in [('SGD', [.00125], (.9, .999), True),
                                   ('AdamW', [.01], (.9, .999), True),
                                   ('AdamW', [.00125], (.8, .999), True),
                                   ('AdamW', [.00125], (.9, .999), False)]:
            with self.assertRaises(RuntimeError):
                m.assert_training_contract(args, opt, lr, beta, amp)

    def test_guard_read_write_boundaries(self):
        m = self.tool()
        self.assertTrue(m.allowed_file(m.OUT/'A/model/weights/best.pt', False))
        self.assertTrue(m.allowed_file(m.OUT/'runtime/cache', True))
        self.assertTrue(m.allowed_file(m.PYTHON.parent/'Lib/site-packages/x.py', False))
        self.assertFalse(m.allowed_file(m.PYTHON.parent/'Lib/site-packages/x.py', True))
        self.assertFalse(m.allowed_file(ROOT/'unrelated.txt', False))
        self.assertFalse(m.allowed_file(ROOT/'tests/test_train_cleaning_ab.py', False))
        self.assertTrue(m.allowed_file(PATH, False))

    def test_stage_paths_do_not_confuse_preflight_and_model_runs(self):
        m = self.tool()
        self.assertEqual(m.stage_path('preflight', None, None), m.OUT/'preflight')
        self.assertEqual(m.stage_path('train', 'A', None), m.OUT/'A/train')
        self.assertEqual(m.stage_path('evaluate', 'B', 'val966'), m.OUT/'B/eval_val966')

    def test_auto_optimizer_effective_warmup_is_zero(self):
        m=self.tool(); actual=m.training_options();actual['warmup_bias_lr']=0.
        m.assert_training_contract(actual, 'AdamW', [.00125], (.9,.999), True)
        self.assertEqual(m.training_options()['warmup_bias_lr'], .1)

    def test_manifest_byte_check_rejects_same_count_label_edit(self):
        m=self.tool()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            p=Path(temp)/'label.txt';p.write_text('1 .5 .5 .2 .2\n')
            row={'label':str(p),'label_sha256':m.sha(p)}
            m.verify_rows([row], kinds=['label'])
            p.write_text('2 .5 .5 .2 .2\n')
            with self.assertRaises(RuntimeError):m.verify_rows([row], kinds=['label'])

    def test_loader_checks_membership_and_box_count(self):
        from types import SimpleNamespace
        m=self.tool()
        dataset=SimpleNamespace(im_files=['a.jpg'],labels=[{'cls':[1,1]}])
        m.assert_loader(dataset,[{'image':'a.jpg'}],2)
        with self.assertRaises(RuntimeError):m.assert_loader(dataset,[{'image':'b.jpg'}],2)
        with self.assertRaises(RuntimeError):m.assert_loader(dataset,[{'image':'a.jpg'}],3)

    def test_rectangular_validation_can_reorder_but_not_change_membership(self):
        from types import SimpleNamespace
        m=self.tool(); dataset=SimpleNamespace(im_files=['b.jpg','a.jpg'],labels=[{'cls':[1]},{'cls':[2]}])
        rows=[{'image':'a.jpg'},{'image':'b.jpg'}]
        m.assert_loader(dataset,rows,2,ordered=False)
        with self.assertRaises(RuntimeError):m.assert_loader(dataset,rows,2)
        dataset.im_files=['b.jpg','b.jpg']
        with self.assertRaises(RuntimeError):m.assert_loader(dataset,rows,2,ordered=False)


if __name__ == '__main__':
    unittest.main()
