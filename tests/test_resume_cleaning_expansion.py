"""Synthetic recovery contracts; no torch imports or model execution."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class ResumeContracts(unittest.TestCase):
    def tool(self):
        p=ROOT/'tools/resume_cleaning_expansion.py'
        self.assertTrue(p.is_file(),'Missing scoped recovery executor')
        spec=importlib.util.spec_from_file_location('resume_expansion_test',p)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        return m

    def test_checkpoint_requires_epoch_optimizer_ema_scaler_and_frozen_args(self):
        m=self.tool()
        args=dict(m.core().training_options(),device='0',warmup_bias_lr=0.)
        ckpt=dict(epoch=3,optimizer={'state':{1:{}},'param_groups':[{}]},ema=object(),
                  scaler={'scale':65536},updates=100,best_fitness=.45293,train_args=args)
        self.assertEqual(m.check_checkpoint_metadata(ckpt)['next_epoch'],5)
        for key,value in [('epoch',-1),('optimizer',None),('ema',None),('scaler',{}),('updates',None)]:
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                m.check_checkpoint_metadata(dict(ckpt,**{key:value}))
        with self.assertRaises(RuntimeError):
            m.check_checkpoint_metadata(dict(ckpt,train_args=dict(ckpt['train_args'],epochs=32)))

    def test_recovery_guard_preserves_source_and_only_new_output_is_writable(self):
        m=self.tool();self.assertTrue(hasattr(m,'configured'),'Missing isolated runtime configuration')
        c=m.configured()
        self.assertEqual(c.OUT,m.OUT);self.assertEqual(c.SELF,m.SELF)
        self.assertTrue(c.allowed_file(m.SOURCE/'D080/yolov8n/weights/last.pt',False))
        self.assertFalse(c.allowed_file(m.SOURCE/'D080/yolov8n/weights/last.pt',True))
        self.assertTrue(c.allowed_file(m.OUT/'runtime/cache',True))
        self.assertFalse(c.allowed_file(ROOT.parent/'other.txt',False))

    def test_merge_requires_exact_original_four_and_remaining_twenty_four(self):
        m=self.tool();self.assertTrue(hasattr(m,'merge_epochs'),'Missing epoch continuity gate')
        prior=[dict(epoch=i,best_updated=True) for i in range(1,5)]
        new=[dict(epoch=i,best_updated=False) for i in range(5,29)]
        self.assertEqual(len(m.merge_epochs(prior,new)),28)
        for wrong in [new[1:],new+new[:1],[dict(epoch=4)]+new[1:]]:
            with self.assertRaises(RuntimeError):m.merge_epochs(prior,wrong)

    def test_recovery_resume_options_only_relocate_paths_and_set_resume(self):
        m=self.tool();self.assertTrue(hasattr(m,'check_resume_args'),'Missing effective resume contract')
        baseline=dict(epochs=28,batch=32,seed=0,resume=False,model='old',data='old',project='old',name='old',save_dir='old')
        actual=dict(baseline,resume=str(m.OUT/'checkpoint/last.pt'),model='new',data='new',project='new',name='new',save_dir='new')
        m.check_resume_args(actual,baseline)
        with self.assertRaises(RuntimeError):m.check_resume_args(dict(actual,batch=16),baseline)
        with self.assertRaises(RuntimeError):m.check_resume_args(dict(actual,resume=False),baseline)

    def test_resume_summary_does_not_claim_three_uninterrupted_from_scratch_runs(self):
        p=ROOT/'tools/summarize_cleaning_expansion_resume.py'
        self.assertTrue(p.is_file(),'Missing recovery-aware report')
        s=importlib.util.spec_from_file_location('resume_summary_test',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
        text=m.disclosure()
        self.assertIn('D080',text);self.assertIn('resume',text);self.assertIn('RNG',text)

    def test_resume_trainer_relocates_native_paths_without_changing_epochs(self):
        m=self.tool();self.assertTrue(hasattr(m,'relocate_resume_args'),'Missing native-resume path relocation')
        from types import SimpleNamespace
        args=SimpleNamespace(resume='old',model='old',data='old',project='old',name='old',save_dir='old',epochs=28,batch=32)
        m.relocate_resume_args(args)
        self.assertEqual(args.epochs,28);self.assertEqual(args.batch,32)
        self.assertEqual(args.data,str(m.OUT/'D080.yaml'))
        self.assertEqual(args.resume,str(m.OUT/'checkpoint/last.pt'))
        self.assertTrue(Path(args.save_dir).is_relative_to(m.OUT))

    def test_recovery_is_single_use_without_deleting_original_markers(self):
        m=self.tool();self.assertTrue(hasattr(m,'ready'),'Missing recovery once-only gate')
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            m.OUT=Path(temp);c=m.configured()
            with self.assertRaises(RuntimeError):m.ready(c)
            c.save(m.OUT/'supervision_started.json',{'started':True})
            with self.assertRaisesRegex(RuntimeError,'repeat'):m.ready(c)

    def test_manifest_relocation_keeps_hashes_and_maps_only_source_data(self):
        m=self.tool();self.assertTrue(hasattr(m,'relocate_row'),'Missing verified independent copy mapping')
        row=dict(image=str(m.SOURCE/'data/D080/images/train/x.jpg'),label=str(m.SOURCE/'data/D080/labels/train/x.txt'),image_sha256='image',label_sha256='label')
        r=m.relocate_row(row)
        self.assertEqual(r['image'],str(m.OUT/'data/D080/images/train/x.jpg'))
        self.assertEqual(r['label_sha256'],'label')
        with self.assertRaises(ValueError):m.relocate_row(dict(row,label=str(ROOT/'other.txt')))

    def test_final_reports_require_manifest_hashes_and_resume_disclosure(self):
        m=self.tool();self.assertTrue(hasattr(m,'verify_reports'),'Missing final delivery report validation')
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            m.OUT=Path(temp);c=m.configured()
            protocol=dict(D080_resume=True,interrupted_completed_epochs=4,additional_epochs=24,uninterrupted_equivalence=False)
            c.save(m.OUT/'comparison.json',dict(recovery_protocol=protocol))
            (m.OUT/'RESULT.md').write_text('D080 resume=true; RNG streams differ; not uninterrupted.',encoding='utf-8')
            for name in m.REPORT_FILES:
                if not (m.OUT/name).exists():(m.OUT/name).write_text('fixture',encoding='utf-8')
            marker=dict(D080_resume=True,models=3,evaluations=12,original_attempt_unchanged=True,
                        files={name:c.sha(m.OUT/name) for name in m.REPORT_FILES})
            c.save(m.OUT/'success.json',marker)
            self.assertTrue(m.verify_reports(c)['reports_verified'])
            (m.OUT/'metrics.csv').write_text('modified',encoding='utf-8')
            with self.assertRaises(RuntimeError):m.verify_reports(c)
            marker['files']['metrics.csv']=c.sha(m.OUT/'metrics.csv')
            (m.OUT/'RESULT.md').write_text('all from scratch',encoding='utf-8')
            marker['files']['RESULT.md']=c.sha(m.OUT/'RESULT.md')
            with self.assertRaises(RuntimeError):m.verify_reports(c,marker)
            (m.OUT/'RESULT.md').write_text('D080 resume=true; RNG; not uninterrupted',encoding='utf-8')
            marker['files']['RESULT.md']=c.sha(m.OUT/'RESULT.md')
            protocol['uninterrupted_equivalence']=True
            (m.OUT/'comparison.json').write_text(__import__('json').dumps(dict(recovery_protocol=protocol)))
            marker['files']['comparison.json']=c.sha(m.OUT/'comparison.json')
            with self.assertRaises(RuntimeError):m.verify_reports(c,marker)
            (m.OUT/'RESULT.md').unlink()
            with self.assertRaises(RuntimeError):m.verify_reports(c,marker)


if __name__=='__main__':unittest.main()
