"""Synthetic geometry, scope and evidence safety; no model imports/inference."""
import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('expansion_under_test',ROOT/'tools/bus_crop_expansion.py')
e=importlib.util.module_from_spec(spec);sys.modules['expansion_under_test']=e;spec.loader.exec_module(e)


def fixtures():
    samples=[];views=[]
    for sid,size in zip(e.GROUPS,e.GROUP_SIZES):
        for frame in range(size):
            ident=sid+f'-synthetic-{frame}';base=f'{sid}_img{frame:05d}_jpg.rf.synthetic'
            sample=dict(id=ident,scene=sid,split='train',frame=frame,image_path=f'{e.P.KIT.relative_to(ROOT).as_posix()}/data/train/images/{base}.jpg',
                label_path=f'{e.P.KIT.relative_to(ROOT).as_posix()}/data/train/labels/{base}.txt',image_sha256='synthetic',label_sha256='synthetic',labels=[],bus_boxes=0)
            samples.append(sample)
            for i,roi in enumerate(e.EXPECTED_ROIS[sid]):
                tile=f'T{i+1}';views.append(dict(id=ident+':'+tile,sample_id=ident,scene=sid,frame=frame,tile=tile,roi=list(roi),scale=2,input_file=f'inputs/{ident}_{tile}.png'))
    return samples,views


def record(view,row=None):
    return dict(view,actual_tensor_shape=[1,3,640,640],image_shape=[640,640],effective_params=dict(e.PARAMS),
        candidates=[] if row is None else [e.RT.candidate(e.P,view,0,row)])


class CoverageTests(unittest.TestCase):
    def test_original_pilot_uses_same_geometry(self):
        self.assertEqual(e.coverage_rois([331,0,362,76]),[[186,0,506,320]])

    def test_full_image_four_tiles(self):
        self.assertEqual(e.coverage_rois([0,0,640,640]),e.EXPECTED_ROIS['MVI_40181'])
        self.assertTrue(e.union_mask(e.EXPECTED_ROIS['MVI_40181']).all())

    def test_edge_just_fits(self):
        self.assertEqual(e.coverage_rois([453,1,640,321]),[[320,1,640,321]])

    def test_321_requires_two_per_axis(self):
        self.assertEqual(len(e.coverage_rois([0,0,321,320])),2)
        self.assertEqual(len(e.coverage_rois([0,0,321,321])),4)

    def test_random_bounding_boxes_covered(self):
        rng=random.Random(17)
        for _ in range(100):
            x0=rng.randrange(640);y0=rng.randrange(640);x1=rng.randrange(x0+1,641);y1=rng.randrange(y0+1,641)
            rois=e.coverage_rois([x0,y0,x1,y1]);mask=e.union_mask(rois)
            self.assertTrue(mask[y0:y1,x0:x1].all());self.assertIn(len(rois),(1,2,4))

    def test_bad_bounds(self):
        for b in ([0,0,641,640],[0,0,0,5],[-1,0,5,5],[1,0,0,5],[0,0,320.5,320],[0,0,math.nan,320]):
            with self.assertRaises(ValueError):e.coverage_rois(b)

    def test_non320_roi_rejected(self):
        with self.assertRaises(ValueError):e.union_mask([[0,0,640,640]])

    def test_overlap_is_preserved(self):
        rois=e.EXPECTED_ROIS['MVI_39401'];self.assertEqual(len(rois),4)
        self.assertLess(e.union_mask(rois).sum(),4*320*320)

    def test_crop_exact_bilinear_rgb_and_no_source_change(self):
        y,x=np.mgrid[0:640,0:640];a=np.stack([x%256,y%256,(x+y)%256],axis=-1).astype(np.uint8);im=Image.fromarray(a)
        before=im.tobytes();roi=[152,97,472,417];out=e.crop_input(im,roi)
        expected=im.crop(tuple(roi)).resize((640,640),Image.Resampling.BILINEAR)
        self.assertEqual(out.size,(640,640));self.assertEqual(out.tobytes(),expected.tobytes());self.assertEqual(im.tobytes(),before)

    def test_mapping_fractional_independent_axes(self):
        for roi in sum(e.EXPECTED_ROIS.values(),[]):
            original=e.P.map_box([.25,13.5,600.75,639.5],roi)
            self.assertEqual(original,[roi[0]+.125,roi[1]+6.75,roi[0]+300.375,roi[1]+319.75])
            self.assertEqual(e.P.inverse_box(original,roi),[.25,13.5,600.75,639.5])

    def test_all_five_heat_sources_and_fixed_counts(self):
        inv,groups,samples,views=e.freeze_sources()
        self.assertEqual([g['id'] for g in groups],e.GROUPS)
        self.assertEqual([len(g['rois']) for g in groups],[4,2,4,2,1])
        self.assertEqual((len(samples),len(views)),(492,1239))
        self.assertTrue(all(g['heat_covered'] for g in groups))
        self.assertEqual(sum(g['train_without_bus_label'] for g in groups),181)


class ScopeTests(unittest.TestCase):
    def setUp(self):self.samples,self.views=fixtures()
    def reject(self):
        with self.assertRaises(ValueError):e.validate_fixed(self.samples,self.views)
    def test_exact_fixed_scope(self):e.validate_fixed(self.samples,self.views)
    def test_extra_image(self):self.samples.append(copy.deepcopy(self.samples[0]));self.reject()
    def test_duplicate_image(self):self.samples[1]=copy.deepcopy(self.samples[0]);self.reject()
    def test_wrong_group(self):self.samples[0]['scene']='MVI_20011';self.reject()
    def test_val_image_rejected(self):self.samples[0]['split']='val';self.reject()
    def test_external_path_rejected(self):self.samples[0]['image_path']='../other.jpg';self.reject()
    def test_wrong_label_path(self):self.samples[0]['label_path']='../other.txt';self.reject()
    def test_filename_frame_mismatch(self):self.samples[0]['frame']=10000;self.reject()
    def test_moved_roi(self):self.views[0]['roi'][0]+=1;self.reject()
    def test_extra_view(self):self.views.append(copy.deepcopy(self.views[0]));self.reject()
    def test_missing_view(self):self.views.pop();self.reject()
    def test_reordered_views(self):self.views.reverse();self.reject()
    def test_duplicate_view(self):self.views[1]=copy.deepcopy(self.views[0]);self.reject()
    def test_changed_scale(self):self.views[0]['scale']=4;self.reject()
    def test_changed_input_path(self):self.views[0]['input_file']='../extra.png';self.reject()


class ResultTests(unittest.TestCase):
    def setUp(self):self.view=fixtures()[1][0]
    def test_valid_result(self):e.validate_result(record(self.view,[1,2,639,640,.5,3]),self.view)
    def test_wrong_class_rejected(self):
        for cls in (-1,4,1.5):
            with self.assertRaises(ValueError):record(self.view,[0,0,30,30,.5,cls])
    def test_confidence_finite_and_band(self):
        for conf in (.0499,1.1,math.inf,math.nan):
            with self.assertRaises(ValueError):record(self.view,[0,0,30,30,conf,3])
        self.assertEqual(record(self.view,[0,0,30,30,.249,3])['candidates'][0]['band'],'weak')
        self.assertEqual(record(self.view,[0,0,30,30,.25,3])['candidates'][0]['band'],'high')
    def test_mapping_tamper_rejected(self):
        r=record(self.view,[0,0,30,30,.5,3]);r['candidates'][0]['xyxy_original'][0]+=1
        with self.assertRaises(ValueError):e.validate_result(r,self.view)
    def test_effective_param_tamper_rejected(self):
        r=record(self.view);r['effective_params']['conf']=.01
        with self.assertRaises(ValueError):e.validate_result(r,self.view)
    def test_input_shape_tamper_rejected(self):
        r=record(self.view);r['actual_tensor_shape']=[1,3,320,320]
        with self.assertRaises(ValueError):e.validate_result(r,self.view)
    def test_input_hash_mismatch_rejected(self):
        v=dict(self.view,input_rgb_sha256='old');r=record(v);r['input_rgb_sha256']='wrong'
        with self.assertRaises(ValueError):e.validate_result(r,v)
    def test_partial_json_corrupt_retained(self):
        with tempfile.TemporaryDirectory(dir=e.SCRATCH) as tmp,patch.object(e,'OUT',Path(tmp)):
            (Path(tmp)/'partial').mkdir();p=Path(tmp)/'partial/0000.json';p.write_text('{unfinished',encoding='utf-8')
            records,errors=e.collect([self.view]);self.assertEqual(records,[]);self.assertEqual(len(errors),1);self.assertEqual(p.read_text(),'{unfinished')
    def test_partial_gap_reported(self):
        with tempfile.TemporaryDirectory(dir=e.SCRATCH) as tmp,patch.object(e,'OUT',Path(tmp)):
            (Path(tmp)/'partial').mkdir();e.write(Path(tmp)/'partial/0001.json',record(self.view));records,errors=e.collect([self.view,self.view])
            self.assertEqual(records,[]);self.assertEqual(len(errors),1)
    def test_multi_view_counts_are_not_target_counts(self):
        samples,views=fixtures();r0=record(views[0],[0,0,30,30,.5,3]);r1=record(views[1],[0,0,30,30,.5,3]);out=e.machine_summary(samples,[r0,r1])[0]
        self.assertEqual(out['bus_high'],2);self.assertEqual(out['images_with_high_bus'],1);self.assertEqual(out['images_without_bus_labels_with_high_bus'],1)
    def test_weak_not_counted_as_high_image(self):
        samples,views=fixtures();out=e.machine_summary(samples,[record(views[0],[0,0,30,30,.1,3])])[0]
        self.assertEqual(out['bus_weak'],1);self.assertEqual(out['images_with_high_bus'],0)
    def test_iou_not_adjudication(self):
        self.assertEqual(e.box_iou([0,0,10,10],[0,0,10,10]),1)
        self.assertEqual(e.box_iou([0,0,10,10],[10,0,20,10]),0)
        self.assertAlmostEqual(e.box_iou([0,0,10,10],[5,0,15,10]),1/3)
        self.assertEqual(e.label_box([3,'.5','.5','.5','.25']),[160,240,480,400])


class SafetyTests(unittest.TestCase):
    def test_cli_dispatch_all_commands_without_executing(self):
        for command,target in [('prepare','prepare'),('run','run'),('verify','verify'),('_worker','worker')]:
            with self.subTest(command=command),patch.object(sys,'argv',['bus_crop_expansion.py',command]),patch.object(e,'prepare') as prep,patch.object(e,'run') as run,patch.object(e,'verify') as verify,patch.object(e,'worker') as worker:
                e.main()
                mocks={'prepare':prep,'run':run,'verify':verify,'worker':worker}
                for name,mock in mocks.items():
                    if name==target:mock.assert_called_once_with()
                    else:mock.assert_not_called()
    def test_prepare_preserves_existing(self):
        with tempfile.TemporaryDirectory(dir=e.SCRATCH) as tmp,patch.object(e,'OUT',Path(tmp)),patch.object(e.P,'git',side_effect=AssertionError('too late')):
            with self.assertRaises(FileExistsError):e.prepare()
    def test_run_refuses_second_attempt(self):
        with tempfile.TemporaryDirectory(dir=e.SCRATCH) as tmp,patch.object(e,'OUT',Path(tmp)),patch.object(e,'validate_prepared',side_effect=AssertionError('too late')):
            (Path(tmp)/'partial').mkdir()
            with self.assertRaises(FileExistsError):e.run()
    def test_private_worker_token_required(self):
        with patch.object(e,'read',return_value={'scratch_work':str(e.SCRATCH),'token':'required-private-token'}):
            with self.assertRaises(RuntimeError):e.worker()
    def test_model_imports_only_in_worker(self):
        tree=ast.parse((ROOT/'tools/bus_crop_expansion.py').read_text(encoding='utf-8'))
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name!='worker':
                for n in ast.walk(node):
                    if isinstance(n,ast.Import):self.assertFalse(any(a.name.split('.')[0] in ('torch','tlc','ultralytics','tlc_ultralytics') for a in n.names))
                    if isinstance(n,ast.ImportFrom):self.assertNotIn((n.module or '').split('.')[0],('torch','tlc','ultralytics','tlc_ultralytics'))
        self.assertFalse(any(name in sys.modules for name in ('torch','ultralytics','tlc','tlc_ultralytics')))
    def test_no_review_form_or_storage(self):
        text=(ROOT/'tools/bus_crop_expansion.template.html').read_text(encoding='utf-8')
        self.assertNotIn('localStorage',text);self.assertNotIn('type="file"',text)
        self.assertEqual(text.count('__EXPANSION_DATA__'),1)
    def test_draw_clears_stale_frame_before_loading(self):
        text=(ROOT/'tools/bus_crop_expansion.template.html').read_text(encoding='utf-8')
        draw=text.split('async function draw(){',1)[1].split('function fmt(',1)[0]
        loading=draw.index('await Promise.all')
        self.assertLess(draw.index('ca.clearRect(0,0,640,640)'),loading)
        self.assertLess(draw.index('cb.clearRect(0,0,640,640)'),loading)
        self.assertLess(draw.index('window.Expansion.lastDraw=null'),loading)
    def test_json_exclusive_write(self):
        with tempfile.TemporaryDirectory(dir=e.SCRATCH) as tmp:
            path=Path(tmp)/'opinion.json';e.write(path,{'original':True})
            with self.assertRaises(FileExistsError):e.write(path,{'overwrite':True})
            self.assertTrue(e.read(path)['original'])
    def test_render_preserves_existing(self):
        with tempfile.TemporaryDirectory(dir=e.SCRATCH) as tmp,patch.object(e,'OUT',Path(tmp)):
            path=Path(tmp)/'review.html';path.write_text('user original',encoding='utf-8')
            with self.assertRaises(FileExistsError):e.render([],[],{})
            self.assertEqual(path.read_text(),'user original')

if __name__=='__main__':unittest.main()
