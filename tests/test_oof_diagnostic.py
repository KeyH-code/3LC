"""Synthetic and mock checks only. Never construct a real model or invoke GPU inference."""
import ast
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('oof_unit',ROOT/'tools/oof_diagnostic.py')
o=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(o)


def samples(groups=30):
    return [dict(id=f'g{g:02d}-i{i}',scene=f'MVI_{g:05d}',frame=i,image_sha256=f'image-{g}-{i}',label_sha256=f'label-{g}-{i}',image_path=f'train/images/{g}-{i}.jpg',label_path=f'train/labels/{g}-{i}.txt',labels=[[c,'.5','.5','.2','.2'] for c in range(4)]) for g in range(groups) for i in range(1+g%3)]


def record(s,cc=None):
    return dict({k:s[k] for k in ['id','scene','frame','image_path','label_path','image_sha256','label_sha256']},fold=1,model_sha256='model',effective_params=o.cfg()['prediction'],input_shape=[1,3,640,640],original_shape=[640,640],candidates=[] if cc is None else cc)


class ScratchCase(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='oof-unit-',dir=o.SCRATCH);self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()


class Splitting(unittest.TestCase):
    def setUp(self):self.ss=samples();self.ff=o.split_folds(self.ss)
    def test_deterministic(self):self.assertEqual(self.ff,o.split_folds(self.ss));self.assertEqual(len(self.ff),5)
    def test_outer_unique(self):
        ids=[i for f in self.ff for i in f['outer']];self.assertEqual(len(ids),len(self.ss));self.assertEqual(len(ids),len(set(ids)))
    def test_role_coverage(self):o.validate_splits(self.ss,self.ff)
    def test_cross_group_hash_components(self):
        ss=copy.deepcopy(self.ss);ss[1]['image_sha256']=ss[0]['image_sha256'];ff=o.split_folds(ss)
        for f in ff:
            self.assertEqual(next(r for r in ('fit','monitor','outer') if ss[0]['id'] in f[r]),next(r for r in ('fit','monitor','outer') if ss[1]['id'] in f[r]))
    def test_transitive_components(self):
        ss=copy.deepcopy(self.ss);ss[1]['image_sha256']=ss[0]['image_sha256'];ss[2]['image_sha256']=ss[3]['image_sha256']
        cc=o.components(ss);self.assertTrue(any(len(c['scenes'])==3 for c in cc))
    def test_too_few_groups(self):
        with self.assertRaises(ValueError):o.split_folds(samples(3))
    def test_missing_class(self):
        ss=copy.deepcopy(self.ss)
        for s in ss:s['labels']=s['labels'][:3]
        with self.assertRaises(ValueError):o.split_folds(ss)
    def test_duplicate_id(self):
        ss=copy.deepcopy(self.ss);ss[1]['id']=ss[0]['id']
        with self.assertRaises(ValueError):o.validate_splits(ss,self.ff)
    def test_duplicate_role_id(self):
        ff=copy.deepcopy(self.ff);ff[0]['fit'].append(ff[0]['fit'][0])
        with self.assertRaises(ValueError):o.validate_splits(self.ss,ff)
    def test_missing_image(self):
        ff=copy.deepcopy(self.ff);ff[0]['fit'].pop()
        with self.assertRaises(ValueError):o.validate_splits(self.ss,ff)
    def test_role_hash_leakage(self):
        ss=copy.deepcopy(self.ss);by={s['id']:s for s in ss};by[self.ff[0]['fit'][0]]['image_sha256']=by[self.ff[0]['outer'][0]]['image_sha256']
        with self.assertRaises(ValueError):o.validate_splits(ss,self.ff)
    def test_duplicate_path(self):
        ss=copy.deepcopy(self.ss);by={s['id']:s for s in ss};by[self.ff[0]['fit'][0]]['image_path']=by[self.ff[0]['fit'][1]]['image_path']
        with self.assertRaises(ValueError):o.validate_splits(ss,self.ff)


class ConfigAndRecords(ScratchCase):
    def test_approved_config(self):
        c=o.cfg();self.assertEqual((c['training']['epochs'],c['training']['patience'],c['training']['seed'],c['split_seed']),(24,8,0,42));self.assertFalse(c['training']['pretrained']);self.assertEqual(c['budget']['training_attempts'],5);self.assertTrue(c['safety']['allow_retry'])
    def test_config_drift(self):
        text=o.CONFIG.read_text(encoding='utf-8').replace('patience: 8','patience: 9');p=self.root/'bad.yaml';p.write_text(text,encoding='utf-8')
        with patch.object(o,'CONFIG',p),self.assertRaises(ValueError):o.cfg()
    def test_device_canonicalization(self):
        o.assert_effective({'device':'cuda:0'},{'device':0})
        with self.assertRaises(RuntimeError):o.assert_effective({'device':'cpu'},{'device':0})
    def test_empty_prediction_retained(self):s=samples()[0];o.validate_prediction(record(s),s,1,'model',o.cfg()['prediction'])
    def test_valid_candidate(self):
        s=samples()[0];c=o.candidate(s['id'],0,[0,0,640,640,.001,3]);o.validate_prediction(record(s,[c]),s,1,'model',o.cfg()['prediction']);self.assertEqual(c['class_name'],'bus')
    def test_invalid_candidates(self):
        for row in ([0,0,640,640,float('nan'),3],[0,0,641,640,.5,3],[0,0,640,640,.5,4],[0,0,640,640,.5,1.1],[0,0,0,640,.5,1],[0,0,640,640,.0009,1]):
            with self.subTest(row=row),self.assertRaises(ValueError):o.candidate('id',0,row)
    def test_identity_model_shape_and_params(self):
        s=samples()[0]
        for key,value in [('fold',2),('model_sha256','wrong'),('image_path','test/a.jpg'),('image_sha256','wrong'),('original_shape',[320,320]),('input_shape',[1,3,320,320])]:
            r=record(s);r[key]=value
            with self.subTest(key=key),self.assertRaises((ValueError,RuntimeError)):o.validate_prediction(r,s,1,'model',o.cfg()['prediction'])
        r=record(s);r['effective_params']['conf']=.25
        with self.assertRaises(RuntimeError):o.validate_prediction(r,s,1,'model',o.cfg()['prediction'])
    def test_candidate_identity(self):
        s=samples()[0];c=o.candidate(s['id'],0,[1,1,2,2,.5,1]);c['id']='bad'
        with self.assertRaises(ValueError):o.validate_prediction(record(s,[c]),s,1,'model',o.cfg()['prediction'])
    def test_max_det(self):
        s=samples()[0];rr=record(s,[o.candidate(s['id'],i,[1,1,2,2,.5,1]) for i in range(301)])
        with self.assertRaises(ValueError):o.validate_prediction(rr,s,1,'model',o.cfg()['prediction'])
    def test_json_and_exclusive_write(self):
        p=self.root/'record.json';o.write(p,{'x':1})
        with self.assertRaises(FileExistsError):o.write(p,{'x':2})
        self.assertEqual(o.read(p),{'x':1})
        with self.assertRaises(ValueError):o.canonical({'x':float('inf')})
    def test_bad_jsonl(self):
        p=self.root/'bad.jsonl'
        for text in ('{}\n\n','{bad}\n'):
            p.write_text(text,encoding='utf-8')
            with self.assertRaises(ValueError):list(o.rows(p))
    def test_deadline(self):
        with self.assertRaises(TimeoutError):o.time_left({'deadline_unix':0},2700)
        with patch.object(o.time,'time',return_value=100):self.assertEqual(o.time_left({'deadline_unix':150},2700),50)
    def test_deadline_supervises_nonmodel_stages(self):
        with patch.object(o.threading,'Timer') as timer,patch.object(o.time,'time',return_value=100),patch.object(o,'OUT',self.root),patch.object(o.os,'_exit') as exit_fn:
            o.arm_deadline(150);self.assertEqual(timer.call_args.args[0],50);timer.return_value.start.assert_called_once();timer.call_args.args[1]();exit_fn.assert_called_once_with(124);self.assertEqual(o.read(self.root/'budget_exceeded.json')['state'],'paused_budget_exhausted')
        o.DEADLINE_TIMER=None
    def test_hash_drift(self):
        p=self.root/'a';p.write_text('old');h=o.sha(p);p.write_text('new')
        with self.assertRaises(ValueError):o.verify_hashes({'a':h},self.root)
    def test_cli_dispatch(self):
        for command,target in [('prepare','prepare'),('run','run'),('verify','verify'),('_worker','worker')]:
            with self.subTest(command=command),patch.object(sys,'argv',['oof',command]),patch.object(o,target) as fn:o.main();fn.assert_called_once_with()
    def test_supervisor_imports_no_models(self):
        tree=ast.parse(Path(o.__file__).read_text(encoding='utf-8'))
        for node in tree.body:
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module]
                self.assertFalse(any(n.split('.')[0] in ('torch','ultralytics','tlc') for n in names))
    def test_template_contract(self):
        t=o.TEMPLATE.read_text(encoding='utf-8');self.assertEqual(t.count('__OOF_DATA__'),1)
        for marker in ('token!==state.token','clear();state.loaded=null','原始零预测图片','model_sha256','prediction_ids'):self.assertIn(marker,t)
        for marker in ('fetch(','localStorage','XMLHttpRequest','<form'):self.assertNotIn(marker,t)
    def test_real_scaled_architecture_copy(self):
        path=Path(o.materialize_architecture(self.root));self.assertTrue(path.is_file());self.assertEqual(path.name,'yolov8n.yaml');self.assertEqual(o.sha(path),o.sha(o.ARCH))
        with self.assertRaises(FileExistsError):o.materialize_architecture(self.root)
    def test_architecture_parser_regression_without_importing_models(self):
        # Run the installed parser function alone with synthetic dependencies: no torch/YOLO import.
        text=(o.PKG/'nn/tasks.py').read_text(encoding='utf-8');tree=ast.parse(text);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='yaml_model_load');calls=[]
        import re,yaml
        def check_yaml(path,hard=True):
            calls.append(str(path));p=Path(path)
            if p.is_file():return str(p)
            if hard:raise FileNotFoundError(str(p))
            return None
        ns={'Path':Path,'re':re,'check_yaml':check_yaml,'YAML':types.SimpleNamespace(load=lambda p:yaml.safe_load(Path(p).read_text(encoding='utf-8'))),'guess_model_scale':lambda p:'n','LOGGER':Mock()}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<installed-parser-only>','exec'),ns)
        path=Path(o.materialize_architecture(self.root));result=ns['yaml_model_load'](str(path));self.assertEqual(result['scale'],'n');self.assertEqual(calls,[str(path)])


class GuardAndLaunch(ScratchCase):
    def config(self):
        return dict(work=str(self.root),read_roots=[],read_files=[str(self.root.parent/'exact-oof.json')],write_roots=[],write_files=[],allowed_weights=[],config_path=str(self.root/'probe_worker_config.json'),token='secret',role='probe')
    def hook(self):
        c=self.config()
        with patch.object(sys,'addaudithook') as add,patch.object(sys,'meta_path',sys.meta_path.copy()):o.install_guard(c);hook=add.call_args.args[0];finder=sys.meta_path[0]
        return hook,finder
    def test_read_write_boundaries(self):
        hook,_=self.hook();hook('open',(str(self.root/'inside.txt'),'w',0));hook('open',(str(self.root.parent/'exact-oof.json'),'r',0))
        for path,mode in [(self.root.parent/'denied.txt','r'),(self.root.parent/'exact-oof.json','w'),(self.root/'old.pt','r')]:
            with self.subTest(path=path,mode=mode),self.assertRaises(PermissionError):hook('open',(str(path),mode,0))
    def test_exact_write_permissions(self):
        c=self.config();p=self.root.parent/'write-only-oof';c['write_files']=[str(p)]
        with patch.object(sys,'addaudithook') as add,patch.object(sys,'meta_path',sys.meta_path.copy()):o.install_guard(c)
        hook=add.call_args.args[0];hook('open',(str(p),'w',0))
        with self.assertRaises(PermissionError):hook('open',(str(p),'r',0))
    def test_network_subprocess_and_3lc(self):
        hook,finder=self.hook()
        for event in ('socket.connect','socket.getaddrinfo','socket.bind','socket.sendto','subprocess.Popen','os.system'):
            with self.subTest(event=event),self.assertRaises(PermissionError):hook(event,())
        for name in ('tlc','tlc_ultralytics.foo','tlcconfig'):
            with self.assertRaises(PermissionError):finder.find_spec(name)
    def test_worker_identity(self):
        p=self.root/'probe_worker_config.json';c=self.config();o.write(p,c)
        with patch.object(o,'OUT',self.root),patch.dict(os.environ,{'OOF_CONFIG':str(p),'OOF_TOKEN':'wrong'}),self.assertRaises(RuntimeError):o.bootstrap()
    def test_worker_cwd_and_bootstrap(self):
        p=self.root/'probe_worker_config.json';c=self.config();o.write(p,c)
        with patch.object(o,'OUT',self.root),patch.dict(os.environ,{'OOF_CONFIG':str(p),'OOF_TOKEN':'secret'}),patch.object(o.Path,'cwd',return_value=self.root),patch.object(o,'install_guard') as guard:
            self.assertEqual(o.bootstrap()['role'],'probe');guard.assert_called_once()
    def test_private_worker_not_public(self):
        with patch.object(o,'BOOT_CONFIG',None),self.assertRaises(RuntimeError):o.worker()
    def test_environment_strips_keys(self):
        with patch.dict(os.environ,{'OPENAI_API_KEY':'do-not-inherit','TLC_API_KEY':'do-not-inherit'}):env=o.environment(self.root)
        self.assertNotIn('OPENAI_API_KEY',env);self.assertNotIn('TLC_API_KEY',env);self.assertEqual(env['YOLO_OFFLINE'],'true');self.assertFalse(o.read(self.root/'config/Ultralytics/settings.json')['sync'])
    def test_no_prepare_overwrite(self):
        with patch.object(o,'OUT',self.root),self.assertRaises(FileExistsError):o.prepare()
    def test_no_run_retry(self):
        o.write(self.root/'run_started.json',{})
        with patch.object(o,'OUT',self.root),self.assertRaises(FileExistsError):o.run()
    def test_no_stage_retry(self):
        o.write(self.root/'started.json',{})
        with self.assertRaises(FileExistsError):o.launch('train',self.root)
    def test_prepare_entire_copy_freeze_flow_without_runtime(self):
        ss=samples();out=self.root/'evidence'
        for s in ss:
            for kind in ('image','label'):
                p=self.root/s[kind+'_path'];p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s['id']+kind,encoding='utf-8');s[kind+'_sha256']=o.sha(p)
        audit=types.SimpleNamespace(git_snapshot=lambda:dict(branch='main',head=o.HEAD))
        with patch.object(o,'ROOT',self.root),patch.object(o,'OUT',out),patch.object(o,'PYTHON',Path(sys.executable)),patch.object(o,'module',return_value=audit),patch.object(o,'originals_integrity',return_value={'files':17668}),patch.object(o,'protected_files',return_value={}),patch.object(o,'inventory',return_value=ss),patch.object(o,'dependency_hashes',return_value={}),patch.object(o,'source_hashes',return_value={}),patch.object(o,'arm_deadline'),patch.object(o,'launch',return_value={'runtime':{}}) as launch,patch('builtins.print'):
            o.prepare();self.assertEqual(launch.call_args.args[0],'probe');self.assertEqual(launch.call_count,1);self.assertTrue((out/'preparation_verified.json').is_file());self.assertTrue((out/'preflight_snapshot.json').is_file())
            f=o.read(out/'fold_manifest.json')[0];by={s['id']:s for s in ss};s=by[f['fit'][0]];derived=out/'fold_01/data/fit/images'/Path(s['image_path']).name;self.assertEqual(o.sha(derived),s['image_sha256']);self.assertNotEqual(derived.stat().st_ino,(self.root/s['image_path']).stat().st_ino)
    def test_launch_mock_success(self):
        stage=self.root/'stage';proc=Mock();proc.wait.return_value=0;proc.poll.return_value=0
        def spawn(*a,**kw):
            self.assertEqual(a[0][1:4],['-I','-B','-u']);o.write(stage/'worker_result.json',dict(state='completed',guard_events=[]));return proc
        with patch.object(o.subprocess,'Popen',side_effect=spawn),patch.object(o,'WindowsJob') as job:
            result=o.launch('probe',stage);self.assertEqual(result['state'],'completed');job.return_value.close.assert_called_once()
        self.assertEqual(o.read(stage/'status.json')['exit_code'],0)
    def test_timeout_preserves_lock_and_evidence(self):
        stage=self.root/'timeout';proc=Mock();proc.wait.side_effect=[subprocess.TimeoutExpired('mock',1),9];proc.poll.return_value=9;proc.returncode=9
        with patch.object(o.subprocess,'Popen',return_value=proc),patch.object(o,'WindowsJob'),self.assertRaises(RuntimeError):o.launch('train',stage,timeout=1)
        self.assertTrue(o.read(stage/'status.json')['timed_out']);self.assertTrue((stage/'started.json').exists());self.assertTrue((stage/'train_worker_config.json').exists());self.assertTrue((stage/'console.log').exists())
    def tearDown(self):o.GUARD_EVENTS.clear();super().tearDown()


class MockTraining(ScratchCase):
    """Intercept train() before it can run, then exercise the registered closures."""
    class Intercept(Exception):pass
    def setUp(self):
        super().setUp();self.callbacks={};self.kw=None
        test=self
        class Base:
            def get_model(self,**kw):return kw
            def save_model(self):return True
        class FakeYOLO:
            ckpt={};ckpt_path=None
            def __init__(self,*a,**kw):pass
            def add_callback(self,event,callback):test.callbacks[event]=callback
            def train(self,**kw):test.kw=kw;raise test.Intercept()
        names=['ultralytics','ultralytics.models','ultralytics.models.yolo','ultralytics.models.yolo.detect','ultralytics.engine','ultralytics.engine.trainer','ultralytics.data','ultralytics.data.utils','ultralytics.utils','ultralytics.utils.torch_utils']
        mods={n:types.ModuleType(n) for n in names}
        for n,m in mods.items():
            m.__path__=[]
            if '.' in n:setattr(mods[n.rsplit('.',1)[0]],n.rsplit('.',1)[1],m)
        mods['ultralytics.models.yolo.detect'].DetectionTrainer=Base;mods['ultralytics.utils.torch_utils'].init_seeds=Mock()
        self.torch=types.SimpleNamespace(isfinite=lambda v:types.SimpleNamespace(all=lambda:math.isfinite(float(v))))
        architecture=o.materialize_architecture(self.root)
        c=dict(runroot=str(self.root),data_yaml=str(self.root/'data.yaml'),training=o.cfg()['training'],architecture=architecture,fold=1,fit_manifest=str(self.root/'fit.json'),monitor_manifest=str(self.root/'monitor.json'),data_root=str(self.root/'data'))
        self.c=c
        with patch.dict(sys.modules,mods),patch.object(o,'load_runtime',return_value=(self.torch,FakeYOLO,{})),self.assertRaises(self.Intercept):o.train_worker(c)
    def test_exact_training_wiring(self):
        self.assertEqual(self.kw['patience'],8);self.assertEqual(self.kw['epochs'],24);self.assertFalse(self.kw['resume']);self.assertFalse(self.kw['pretrained']);self.assertNotIn('outer',self.kw['data']);self.assertEqual(self.kw['workers'],2)
    def test_random_model_only(self):
        trainer=self.kw['trainer']();self.assertIsNone(trainer.get_model(cfg='arch')['weights'])
        with self.assertRaises(RuntimeError):trainer.get_model(weights='old.pt')
    def test_nan_loss_and_fitness_stop(self):
        tr=self.kw['trainer']();tr.loss=1.;tr.tloss={'box':1.,'cls':float('nan')};tr.fitness=.1
        with self.assertRaises(RuntimeError):tr._handle_nan_recovery(0)
        tr.tloss={'box':1.};tr.fitness=float('inf')
        with self.assertRaises(RuntimeError):tr._handle_nan_recovery(0)
        tr.fitness=.1;self.assertFalse(tr._handle_nan_recovery(0))
    def test_batch_nan_stop(self):
        with self.assertRaises(RuntimeError):self.callbacks['on_train_batch_end'](types.SimpleNamespace(loss=1.,tloss={'box':float('nan')}))
    def test_oom_retry_is_disabled_each_epoch(self):
        t=types.SimpleNamespace(_oom_retries=0);self.callbacks['on_train_epoch_start'](t);self.assertGreaterEqual(t._oom_retries,3)
    def test_epoch_trace_dict_loss(self):
        t=types.SimpleNamespace(fitness=.2,metrics={'metrics/mAP50-95(B)':.2},epoch=0,stopper=types.SimpleNamespace(best_epoch=1,best_fitness=.2,possible_stop=False,patience=8),stop=False,best_fitness=.2,tloss={'box_loss':1.,'cls_loss':2.})
        self.callbacks['on_fit_epoch_end'](t);r=list(o.rows(self.root/'stopper_trace.jsonl'))[0];self.assertEqual(r['loss'],{'box_loss':1.,'cls_loss':2.});self.assertTrue(r['best_checkpoint_updated'])
        t.oof_final_eval=True;self.callbacks['on_fit_epoch_end'](t);self.assertEqual(len(list(o.rows(self.root/'stopper_trace.jsonl'))),1)
    def test_native_fitness_checked(self):
        t=types.SimpleNamespace(fitness=.2,metrics={'metrics/mAP50-95(B)':.3})
        with self.assertRaises(RuntimeError):self.callbacks['on_fit_epoch_end'](t)
    def test_initialized_actual_args_optimizer_and_loaders(self):
        ss=samples();fit=[ss[0]];mon=[ss[1]];o.write(self.root/'fit.json',fit);o.write(self.root/'monitor.json',mon)
        model=types.SimpleNamespace(names=o.NAMES,yaml={'scale':'n'},parameters=lambda:[types.SimpleNamespace(numel=lambda:3011628)],state_dict=lambda:{})
        optimizer=type('AdamW',(),{'param_groups':[{'lr':.00125,'betas':(.9,.999)}]})()
        def loader(role,ss):return types.SimpleNamespace(dataset=types.SimpleNamespace(im_files=[str(self.root/'data'/role/'images'/Path(s['image_path']).name) for s in ss],labels=[{'cls':[0]*len(s['labels'])} for s in ss]))
        args=dict(self.c['training'],device='cuda:0');t=types.SimpleNamespace(args=types.SimpleNamespace(**args),device='cuda:0',amp=True,model=model,optimizer=optimizer,train_loader=loader('fit',fit),test_loader=loader('monitor',mon))
        self.callbacks['on_train_start'](t);rt=o.read(self.root/'training_runtime.json');self.assertFalse(rt['external_amp_check_executed']);self.assertFalse(rt['oom_recovery_enabled']);self.assertEqual(rt['effective_args']['patience'],8)
        t.args.seed=42
        with self.assertRaises(RuntimeError):self.callbacks['on_train_start'](t)


class SavedEvaluation(unittest.TestCase):
    def test_iou(self):self.assertEqual(o.iou([0,0,1,1],[0,0,1,1]),1);self.assertEqual(o.iou([0,0,1,1],[2,2,3,3]),0)
    def test_empty(self):
        s=samples()[0];m=o.evaluate([record(s)],[s]);self.assertEqual(m['mAP50'],0);self.assertEqual([r['ground_truth'] for r in m['classes']],[1]*4)
    def test_exact_four_class_predictions(self):
        s=samples()[0];cc=[o.candidate(s['id'],c,[256,256,384,384,.9,c]) for c in range(4)];m=o.evaluate([record(s,cc)],[s]);self.assertGreater(m['mAP50_95'],.98)
    def test_wrong_class_not_matched(self):
        s=samples()[0];s['labels']=[[3,'.5','.5','.2','.2']];m=o.evaluate([record(s,[o.candidate(s['id'],0,[256,256,384,384,.9,1])])],[s]);self.assertEqual(m['mAP50'],0)
    def test_unique_match(self):
        s=samples()[0];s['labels']=s['labels'][:1];cc=[o.candidate(s['id'],i,[256,256,384,384,.9-i*.1,0]) for i in range(2)];m=o.evaluate([record(s,cc)],[s]);self.assertEqual(m['classes'][0]['detections'],2);self.assertLessEqual(m['classes'][0]['AP50'],1)


if __name__=='__main__':unittest.main()
