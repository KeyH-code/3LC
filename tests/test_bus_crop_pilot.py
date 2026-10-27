"""Synthetic preparation tests only; no model import, inference or human feedback."""
import ast
import copy
import importlib.util
import math
import os
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('bus_crop_pilot', ROOT/'tools/bus_crop_pilot.py')
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)
SCRATCH = Path(os.environ.get('PI_SCRATCH_DIR', r'E:\pi-desktop\data\data\scratch\112b9029-dd22-4fc5-9132-2dab3956e6cc'))

def fixtures():
    samples = []
    for f in pilot.FRAMES:
        name = f'MVI_20011_img{f:05d}_jpg.rf.synthetic'
        samples.append(dict(id=f'synthetic-{f}', frame=f, scene='MVI_20011', split='train',
                            image_path=f'multi-vehicle_kaggle_competition_starter_kit_files/data/train/images/{name}.jpg',
                            label_path=f'multi-vehicle_kaggle_competition_starter_kit_files/data/train/labels/{name}.txt'))
    views = [dict(id=s['id']+':'+c, sample_id=s['id'], frame=s['frame'], condition=c,
                  roi=list(roi), scale=640/(roi[2]-roi[0]),
                  input_file=None if c=='A' else f'inputs/{s["frame"]:05d}_{c}.png')
             for s in samples for c, roi in pilot.CONDITIONS.items()]
    return samples, views


class Geometry(unittest.TestCase):
    def test_full_canvas_maps_to_roi_at_all_scales(self):
        for roi in pilot.CONDITIONS.values():
            self.assertEqual(pilot.map_box([0.,0.,640.,640.],roi),roi)

    def test_round_trip_fractional_and_edges(self):
        for roi in pilot.CONDITIONS.values():
            for box in ([0.,0.,640.,640.],[.25,13.75,411.125,600.5],[640.,640.,640.,640.]):
                restored=pilot.inverse_box(pilot.map_box(box,roi),roi)
                for a,b in zip(box,restored): self.assertAlmostEqual(a,b,places=10)

    def test_known_two_and_four_times_coordinates(self):
        self.assertEqual(pilot.map_box([20.,40.,100.,200.],pilot.CONDITIONS['B']),[196.,20.,236.,100.])
        self.assertEqual(pilot.map_box([20.,40.,100.,200.],pilot.CONDITIONS['C']),[271.,10.,291.,50.])

    def test_independent_x_y_not_transposed(self):
        self.assertEqual(pilot.map_box([64.,128.,320.,512.],[100,20,420,180]),[132.,52.,260.,148.])

    def test_bad_input_boxes(self):
        for box in ([0,0,641,640],[-1,0,3,4],[3,0,2,4],[0,4,2,3],[0,0,math.nan,4],[0,0,math.inf,4],[0,0,1]):
            with self.subTest(box=box), self.assertRaises(ValueError): pilot.map_box(box,pilot.CONDITIONS['A'])

    def test_bad_roi(self):
        for roi in ([0,0,0,3],[3,0,2,3],[0,0,math.nan,3],[-1,0,4,4],[0,0,641,640]):
            with self.subTest(roi=roi), self.assertRaises(ValueError): pilot.map_box([0,0,1,1],roi)

    def test_inverse_rejects_outside_or_nonfinite(self):
        for box in ([0,0,640,640],[266,0,math.inf,10],[266,0,265,10]):
            with self.assertRaises(ValueError): pilot.inverse_box(box,pilot.CONDITIONS['C'])


class FixedInputs(unittest.TestCase):
    def setUp(self): self.samples,self.views=fixtures()
    def rejects(self):
        with self.assertRaises(ValueError): pilot.validate_fixed(self.samples,self.views)

    def test_exact_13_and_39(self):
        pilot.validate_fixed(self.samples,self.views)
        self.assertEqual((len(self.samples),len(self.views)),(13,39))

    def test_duplicate_sample(self):
        self.samples[1]=copy.deepcopy(self.samples[0]);self.rejects()

    def test_extra_sample(self):
        self.samples.append(copy.deepcopy(self.samples[-1]));self.rejects()

    def test_extra_view(self):
        self.views.append(copy.deepcopy(self.views[0]));self.rejects()

    def test_duplicate_view_id(self):
        self.views[1]['id']=self.views[0]['id'];self.rejects()

    def test_changed_condition(self):
        self.views[1]['condition']='D';self.rejects()

    def test_changed_roi(self):
        self.views[1]['roi'][0]+=1;self.rejects()

    def test_changed_scale(self):
        self.views[1]['scale']=3;self.rejects()

    def test_reordered_views(self):
        self.views.reverse();self.rejects()

    def test_validation_split(self):
        self.samples[0]['split']='val';self.rejects()

    def test_wrong_scene(self):
        self.samples[0]['scene']='MVI_20012';self.rejects()

    def test_outside_image_path(self):
        self.samples[0]['image_path']='../other/MVI_20011_img00005_any.jpg';self.rejects()

    def test_wrong_frame_filename(self):
        self.samples[0]['image_path']=self.samples[0]['image_path'].replace('img00005','img00006');self.rejects()

    def test_outside_label_path(self):
        self.samples[0]['label_path']='../other.txt';self.rejects()

    def test_extra_input_path(self):
        self.views[1]['input_file']='../other.png';self.rejects()

    def test_full_image_must_not_be_copied(self):
        self.views[0]['input_file']='inputs/full.png';self.rejects()


class Confidence(unittest.TestCase):
    def test_bands(self):
        for conf,expected in ((.05,'weak'),(.249999,'weak'),(.25,'high'),(1.,'high')):
            self.assertEqual(pilot.band(conf),expected)

    def test_invalid(self):
        for conf in (.0499,-1.,1.001,math.nan,math.inf):
            with self.assertRaises(ValueError): pilot.band(conf)


class Safety(unittest.TestCase):
    def test_run_refuses_existing_directory_before_checks(self):
        rt=pilot.runtime()
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp, patch.object(rt,'directory',return_value=Path(tmp)), \
             patch.object(pilot,'verify_preparation',side_effect=AssertionError('checked before lock')):
            with self.assertRaises(FileExistsError):rt.run(pilot)

    def test_worker_cannot_start_without_private_configuration(self):
        rt=pilot.runtime()
        with patch.object(pilot,'read',return_value={'worker_token':'required','scratch_work':'missing'}):
            with self.assertRaises(RuntimeError):rt.worker(pilot)

    def test_no_model_imports(self):
        tree=ast.parse(Path(pilot.__file__).read_text(encoding='utf-8'))
        forbidden={'torch','tlc','tlc_ultralytics','ultralytics'}
        for n in ast.walk(tree):
            names=[a.name for a in n.names] if isinstance(n,ast.Import) else [n.module or ''] if isinstance(n,ast.ImportFrom) else []
            self.assertFalse(any(name.split('.')[0] in forbidden for name in names))
        self.assertFalse(forbidden.intersection(sys.modules))

    def test_exclusive_json_write(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            p=Path(tmp)/'evidence.json';pilot.write(p,{'original':True})
            old=p.read_bytes()
            with self.assertRaises(FileExistsError): pilot.write(p,{'overwrite':True})
            self.assertEqual(p.read_bytes(),old)

    def test_existing_output_refused_before_checks(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp, patch.object(pilot,'OUT',Path(tmp)), \
             patch.object(pilot,'git',side_effect=AssertionError('git called')):
            sentinel=Path(tmp)/'user-feedback.json';sentinel.write_text('keep',encoding='utf-8')
            with self.assertRaises(FileExistsError): pilot.prepare()
            self.assertEqual(sentinel.read_text(encoding='utf-8'),'keep')

    def test_unapproved_condition_before_file_read(self):
        with self.assertRaises(ValueError): pilot.make_input({},'D')

    def test_changed_image_refused(self):
        sample=fixtures()[0][0];sample['image_sha256']='expected'
        with patch.object(pilot,'sha',return_value='changed'):
            with self.assertRaisesRegex(ValueError,'Source changed'): pilot.make_input(sample,'A')


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.rt=pilot.runtime();self.views=fixtures()[1]
        self.records=[]
        for v in self.views:
            v.update(input_rgb_sha256='synthetic-rgb-'+v['id'],input_file_sha256=None)
            self.records.append(dict(v,actual_tensor_shape=[1,3,640,640],image_shape=[640,640],effective_params=dict(pilot.PARAMS),
                                     candidates=[self.rt.candidate(pilot,v,0,[.25,12.5,300.75,620.5,.5,3])]))
        self.pred=dict(experiment_id='synthetic-experiment',weight_sha256=pilot.WEIGHT_HASH,records=self.records)

    def item(self,condition='A',candidate=True,status='usable',group='bus1'):
        r=next(r for r in self.records if r['frame']==434 and r['condition']==condition)
        return dict(self.rt.feedback_identity(pilot,r,r['candidates'][0]['id'] if candidate else None),status=status,target_group=group,note='合成测试，不是人工意见')

    def feedback(self,items):
        return dict(schema='bus-crop-feedback-v1',experiment_id='synthetic-experiment',weight_sha256=pilot.WEIGHT_HASH,records=items)

    def test_all39_predictions_and_geometry(self):
        with patch.object(pilot,'read',return_value={'experiment_id':'synthetic-experiment'}):
            self.assertEqual(len(self.rt.validate_predictions(pilot,self.pred,self.views)),39)

    def test_candidate_preserves_float_and_edge_distances(self):
        c=self.records[1]['candidates'][0]
        self.assertEqual(c['xyxy_original'],[186.125,6.25,336.375,310.25])
        self.assertEqual(c['edge_distances_input'],[.25,12.5,339.25,19.5])
        self.assertEqual(c['edge_distances_original'],[.125,6.25,169.625,9.75])

    def test_bad_prediction_rows(self):
        for row in ([0,0,20,20,.5,4],[0,0,20,20,.5,1.5],[0,0,20,20,.01,3],[0,0,math.nan,20,.5,3],[0,0,641,20,.5,3],[0,0,20,20,.5]):
            with self.assertRaises(ValueError):self.rt.candidate(pilot,self.views[0],0,row)

    def test_tampered_mapping_rejected(self):
        self.pred['records'][0]['candidates'][0]['xyxy_original'][0]+=1
        with patch.object(pilot,'read',return_value={'experiment_id':'synthetic-experiment'}):
            with self.assertRaises(ValueError):self.rt.validate_predictions(pilot,self.pred,self.views)

    def test_missing_result_rejected(self):
        self.pred['records'].pop()
        with patch.object(pilot,'read',return_value={'experiment_id':'synthetic-experiment'}):
            with self.assertRaises(ValueError):self.rt.validate_predictions(pilot,self.pred,self.views)

    def test_all_statuses_with_no_fake_negative(self):
        self.assertEqual(self.rt.summarize(pilot,self.feedback([]),self.pred)['reviewed_entries'],0)
        for status in self.rt.STATUSES:
            item=self.item(candidate=status!='not_detected',status=status)
            self.assertEqual(len(self.rt.validate_feedback(pilot,self.feedback([item]),self.pred)),1)

    def test_duplicate_feedback_rejected(self):
        item=self.item()
        with self.assertRaises(ValueError):self.rt.validate_feedback(pilot,self.feedback([item,copy.deepcopy(item)]),self.pred)

    def test_candidate_duplicate_different_group_rejected(self):
        a=self.item();b=self.item(group='other')
        with self.assertRaises(ValueError):self.rt.validate_feedback(pilot,self.feedback([a,b]),self.pred)

    def test_batch_weight_input_and_candidate_mismatch(self):
        for field in ('sample_id','view_id','input_rgb_sha256','weight_sha256','candidate_id','condition'):
            item=self.item();item[field]='wrong'
            with self.subTest(field=field),self.assertRaises(ValueError):self.rt.validate_feedback(pilot,self.feedback([item]),self.pred)

    def test_null_requires_no_box_status(self):
        with self.assertRaises(ValueError):self.rt.validate_feedback(pilot,self.feedback([self.item(candidate=False)]),self.pred)
        with self.assertRaises(ValueError):self.rt.validate_feedback(pilot,self.feedback([self.item(status='not_detected')]),self.pred)

    def test_two_local_views_one_grouped_example(self):
        items=[self.item('A',False,'not_detected'),self.item('B'),self.item('C')]
        result=self.rt.summarize(pilot,self.feedback(items),self.pred)
        self.assertEqual(result['distinct_manually_grouped_targets'],1)
        self.assertEqual(len(result['grouped_local_gain_examples']),1)
        self.assertEqual(result['grouped_local_gain_examples'][0]['conditions'],['B','C'])
        self.assertEqual(result['by_condition']['suspected11']['A']['no_candidate']['not_detected'],1)

    def test_weak_local_is_not_high_threshold_gain(self):
        r=next(r for r in self.records if r['frame']==434 and r['condition']=='B')
        r['candidates'][0]['confidence']=.1;r['candidates'][0]['band']='weak'
        result=self.rt.summarize(pilot,self.feedback([self.item('A',False,'not_detected'),self.item('B')]),self.pred)
        self.assertEqual(result['grouped_local_gain_examples'],[])
        self.assertEqual(result['by_condition']['all13']['B']['weak']['usable'],1)

    def test_ungrouped_not_inferred_or_deduplicated(self):
        result=self.rt.summarize(pilot,self.feedback([self.item('A',False,'not_detected',''),self.item('B',group='')]),self.pred)
        self.assertEqual(result['grouped_local_gain_examples'],[])
        self.assertEqual(result['distinct_manually_grouped_targets'],0)

    def test_settings_no_telemetry_or_keys(self):
        cfg=self.rt.settings(SCRATCH/'synthetic-runtime')
        self.assertFalse(cfg['sync']);self.assertEqual(cfg['api_key'],'');self.assertFalse(cfg['wandb'])

    def test_environment_isolated_no_inherited_keys(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp,patch.dict(os.environ,{'FAKE_API_KEY':'do-not-pass'}):
            env=self.rt.environment(Path(tmp))
            self.assertNotIn('FAKE_API_KEY',env);self.assertEqual(env['YOLO_OFFLINE'],'true')
            self.assertEqual(env['YOLO_AUTOINSTALL'],'false')
            self.assertTrue(Path(tmp) in Path(env['HOME']).parents)

    def test_exclusive_run_directory(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            dest=Path(tmp)/'run';self.rt.exclusive_directory(dest)
            with self.assertRaises(FileExistsError):self.rt.exclusive_directory(dest)

    def test_model_imports_only_in_worker(self):
        tree=ast.parse((ROOT/'tools/bus_crop_runtime.py').read_text(encoding='utf-8'))
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name!='worker':
                for n in ast.walk(node):
                    if isinstance(n,ast.Import):self.assertFalse(any(a.name.split('.')[0] in {'torch','ultralytics','tlc'} for a in n.names))
                    if isinstance(n,ast.ImportFrom):self.assertNotIn((n.module or '').split('.')[0],{'torch','ultralytics','tlc'})


    def test_partial_corrupt_json_retained_and_reported(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            run=Path(tmp);(run/'partial').mkdir()
            pilot.write(run/'partial/00.json',self.records[0])
            damaged=run/'partial/01.json';damaged.write_text('{"unfinished":',encoding='utf-8')
            original_read=pilot.read
            def read(path):
                return {'experiment_id':'synthetic-experiment','source_dataset_id':'synthetic'} if Path(path).name=='plan_config.json' else original_read(path)
            errors=[]
            with patch.object(pilot,'read',side_effect=read):result=self.rt.build_payload(pilot,run,self.views,errors)
            self.assertEqual(len(result['records']),1);self.assertEqual(len(errors),1)
            self.assertIn('JSONDecodeError',errors[0]['error'])
            self.assertEqual(damaged.read_text(encoding='utf-8'),'{"unfinished":')

    def test_partial_gap_reported_without_inference(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            run=Path(tmp);(run/'partial').mkdir();pilot.write(run/'partial/01.json',self.records[1]);errors=[]
            with patch.object(pilot,'read',return_value={'experiment_id':'synthetic-experiment','source_dataset_id':'synthetic'}):
                result=self.rt.build_payload(pilot,run,self.views,errors)
            self.assertEqual(result['records'],[]);self.assertEqual(len(errors),1)

    def test_null_feedback_fields_rejected(self):
        for field in ('note','target_group'):
            item=self.item();item[field]=None
            with self.assertRaises(ValueError):self.rt.validate_feedback(pilot,self.feedback([item]),self.pred)


if __name__=='__main__': unittest.main()
