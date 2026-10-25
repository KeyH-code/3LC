import argparse, collections, concurrent.futures, hashlib, io, json, math, os, platform, re, sys, time, warnings
from pathlib import Path
from decimal import Decimal
import numpy as np
import PIL
from PIL import Image

ROOT = Path(r'E:\kaggle\3LC_Multi_Vehicle_detection_Challenge\.worktrees\score-search-20261019-150516')
DATA = ROOT / 'multi-vehicle_kaggle_competition_starter_kit_files' / 'data'
OUT = Path(os.environ['PI_SCRATCH_DIR'])
PREFIX = 'supplemental_data_audit'

def save(name, obj):
    path = OUT / (PREFIX + name + '.json')
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return str(path)

def rel(path):
    return path.relative_to(DATA).as_posix()

def audit_one(item):
    split, p = item
    r = {'split': split, 'image': rel(p), 'label': rel(DATA / split / 'labels' / (p.stem + '.txt')), 'issues': [], 'duplicate_text': [], 'duplicate_numeric': [], 'overlaps': [], 'blank_lines': 0, 'lines': 0, 'valid_geometry_boxes': 0, 'class_counts': {}}
    lp = DATA / r['label']
    boxes = []
    textgroups = collections.defaultdict(list)
    numericgroups = collections.defaultdict(list)
    classes = collections.Counter()
    try:
        text = lp.read_text(encoding='utf-8-sig')
        r['empty_label'] = not text.strip()
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                r['blank_lines'] += 1
                continue
            r['lines'] += 1
            textgroups[line].append(lineno)
            toks = line.split()
            def issue(kind, **kw):
                r['issues'].append({'line': lineno, 'kind': kind, 'text': line, **kw})
            if len(toks) != 5:
                issue('column_count', count=len(toks))
                continue
            try:
                vals = [float(t) for t in toks]
            except ValueError:
                issue('non_numeric')
                continue
            if not all(math.isfinite(x) for x in vals):
                issue('non_finite')
                continue
            c, x, y, w, h = vals
            numericgroups[tuple(vals)].append(lineno)
            class_ok = c.is_integer() and 0 <= c <= 3
            if not class_ok:
                issue('class_not_integer_0_to_3')
            elif not re.fullmatch(r'[0-3]', toks[0]):
                issue('class_token_not_canonical_integer')
            if class_ok:
                classes[str(int(c))] += 1
            if not all(0 <= v <= 1 for v in (x,y,w,h)):
                issue('normalized_component_outside_0_1')
            if w <= 0 or h <= 0:
                issue('non_positive_width_or_height')
            edges = (x-w/2, y-h/2, x+w/2, y+h/2)
            excess = max(-edges[0], -edges[1], edges[2]-1, edges[3]-1, 0)
            if excess > 0:
                issue('box_edges_outside_image', xyxy=list(edges), max_excess=excess, beyond_1e_6=excess>1e-6)
            if class_ok and w > 0 and h > 0:
                boxes.append((lineno, int(c), *edges))
        r['class_counts'] = dict(classes)
        r['valid_geometry_boxes'] = len(boxes)
        r['duplicate_text'] = [{'text': k, 'lines': v} for k,v in textgroups.items() if len(v)>1]
        r['duplicate_numeric'] = [{'values': list(k), 'lines': v} for k,v in numericgroups.items() if len(v)>1]
        if len(boxes)>1:
            a = np.asarray(boxes, dtype=np.float64)
            xy = a[:,2:6]
            lo = np.maximum(xy[:,None,:2], xy[None,:,:2])
            hi = np.minimum(xy[:,None,2:], xy[None,:,2:])
            wh = np.maximum(hi-lo, 0)
            inter = wh[:,:,0]*wh[:,:,1]
            areas = (xy[:,2]-xy[:,0])*(xy[:,3]-xy[:,1])
            union = areas[:,None]+areas[None,:]-inter
            ious = np.divide(inter, union, out=np.zeros_like(inter), where=union>0)
            ii,jj = np.where(np.triu(ious>=0.9, 1))
            for i,j in zip(ii,jj):
                r['overlaps'].append({'line_a': int(a[i,0]), 'line_b': int(a[j,0]), 'class_a': int(a[i,1]), 'class_b': int(a[j,1]), 'iou': float(ious[i,j]), 'same_class': bool(a[i,1]==a[j,1])})
    except Exception as e:
        r['issues'].append({'kind': 'label_read_error', 'error': str(e)})
        r['empty_label'] = None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with Image.open(p) as im:
                im.load()  # full decode, no draft/reduced-resolution shortcut
                r['size'] = list(im.size)
                r['mode'] = im.mode
                gray = np.asarray(im.convert('L').resize((9,8), Image.Resampling.LANCZOS), dtype=np.uint8)
                bits = gray[:,1:] > gray[:,:-1]
                hashint = int.from_bytes(np.packbits(bits.reshape(-1), bitorder='big').tobytes(), 'big')
                r['dhash64'] = f'{hashint:016x}'
            r['decode_warnings'] = [str(w.message) for w in caught]
    except Exception as e:
        r['decode_error'] = repr(e)
    return r

def main():
    start = time.time()
    audit = json.loads((ROOT/'reports/data-audit/cleaning_audit.json').read_text(encoding='utf-8-sig'))
    manifest = json.loads((ROOT/'reports/data_manifest.json').read_text(encoding='utf-8-sig'))
    known = {(p['train']['rel'],p['val']['rel']) for p in audit['pairs']}
    items = []
    inventory = {}
    for split in ('train','val'):
        images = sorted(p for p in (DATA/split/'images').iterdir() if p.is_file())
        labels = sorted(p for p in (DATA/split/'labels').iterdir() if p.is_file())
        ist = {p.stem for p in images}
        lst = {p.stem for p in labels if p.suffix.lower()=='.txt'}
        inventory[split] = {'images': len(images), 'labels':len(labels), 'missing_labels': sorted(ist-lst), 'orphan_labels': sorted(lst-ist), 'unexpected_label_extensions': [p.name for p in labels if p.suffix.lower()!='.txt']}
        items += [(split,p) for p in images]
    print(json.dumps({'stage':'start','images':len(items),'workers':8}), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(audit_one, items))
    decode_secs = time.time()-start
    full_path = save('_per_image', results)
    trains = [r for r in results if r['split']=='train' and 'dhash64' in r]
    vals = [r for r in results if r['split']=='val' and 'dhash64' in r]
    th = np.asarray([int(r['dhash64'],16) for r in trains],dtype=np.uint64)
    lut = np.asarray([i.bit_count() for i in range(256)],dtype=np.uint8)
    candidates = []
    excluded=[]
    for v in vals:
        xor = np.bitwise_xor(th, np.uint64(int(v['dhash64'],16)))
        ds = lut[xor.view(np.uint8).reshape(-1,8)].sum(axis=1)
        for i in np.flatnonzero(ds<=4):
            t = trains[int(i)]
            pair = (t['image'],v['image'])
            entry = {'train':t['image'],'val':v['image'],'distance':int(ds[i]),'train_dhash64':t['dhash64'],'val_dhash64':v['dhash64'],'same_video_prefix':t['image'].split('/')[-1].split('_img')[0]==v['image'].split('/')[-1].split('_img')[0]}
            if pair in known:
                excluded.append(entry)
            else:
                candidates.append(entry)
    candidates.sort(key=lambda c:(c['distance'],c['train'],c['val']))
    candidates_path=save('_near_duplicate_candidates', candidates)
    top = candidates[:20]
    top_path=save('_top20_review',top)
    # All labels, including any orphan labels, must be covered. Abort rather than silently omit.
    if any(x['orphan_labels'] or x['unexpected_label_extensions'] for x in inventory.values()):
        raise RuntimeError('Unexpected/orphan labels require explicit extension of bounded scan; per-image results already saved')
    totals={}
    for split in ('train','val'):
        rr=[r for r in results if r['split']==split]
        issues=collections.Counter(i['kind'] for r in rr for i in r['issues'])
        cls=collections.Counter()
        for r in rr: cls.update(r['class_counts'])
        overlaps=[(r,o) for r in rr for o in r['overlaps']]
        totals[split]={'images':len(rr),'nonblank_label_lines':sum(r['lines'] for r in rr),'class_counts':dict(cls),'empty_labels':[r['label'] for r in rr if r['empty_label']], 'issue_counts_by_kind':dict(issues),'images_with_any_label_issue':sum(bool(r['issues']) for r in rr),'edge_overflow_lines_beyond_1e_6':sum(i['kind']=='box_edges_outside_image' and i['beyond_1e_6'] for r in rr for i in r['issues']),'exact_text_duplicate_groups':sum(len(r['duplicate_text']) for r in rr),'exact_text_duplicate_extra_rows':sum(len(d['lines'])-1 for r in rr for d in r['duplicate_text']),'numeric_duplicate_groups':sum(len(r['duplicate_numeric']) for r in rr),'images_with_exact_text_duplicates':sum(bool(r['duplicate_text']) for r in rr),'high_overlap_same_class_pairs':sum(o['same_class'] for _,o in overlaps),'high_overlap_different_class_pairs':sum(not o['same_class'] for _,o in overlaps),'images_with_high_overlap_same_class':len({r['image'] for r,o in overlaps if o['same_class']}),'images_with_high_overlap_different_class':len({r['image'] for r,o in overlaps if not o['same_class']}),'decoded':sum('dhash64' in r for r in rr),'decode_errors':[{'image':r['image'],'error':r['decode_error']} for r in rr if 'decode_error' in r],'decode_warning_images':[{'image':r['image'],'warnings':r['decode_warnings']} for r in rr if r.get('decode_warnings')],'dimensions':dict(collections.Counter('x'.join(map(str,r['size'])) for r in rr if 'size' in r))}
    # Decimal recomputation distinguishes binary floating-point epsilon from actual edge overflow.
    edge_detail=[]
    for r in results:
        for item in r['issues']:
            if item['kind'] != 'box_edges_outside_image':
                continue
            c,x,y,w,h = map(Decimal, item['text'].split())
            edges = (x-w/2, y-h/2, x+w/2, y+h/2)
            exc = max(-edges[0],-edges[1],edges[2]-1,edges[3]-1,Decimal(0))
            edge_detail.append({'image':r['image'],'label':r['label'],'line':item['line'],'text':item['text'],'decimal_max_excess':str(exc),'decimal_overflow':exc>0,'decimal_beyond_1e_6':exc>Decimal('0.000001'),'float_max_excess':item['max_excess']})
    edge_path=save('_edge_overflow_decimal',edge_detail)
    for split in ('train','val'):
        ee=[e for e in edge_detail if e['image'].startswith(split+'/')]
        totals[split]['edge_overflow_decimal_summary']={'float_flagged_lines':len(ee),'exact_positive_excess_lines':sum(e['decimal_overflow'] for e in ee),'exact_positive_excess_images':len({e['image'] for e in ee if e['decimal_overflow']}),'float_only_epsilon_lines':sum(not e['decimal_overflow'] for e in ee),'exact_beyond_1e_6_lines':sum(e['decimal_beyond_1e_6'] for e in ee),'max_exact_normalized_excess':str(max((Decimal(e['decimal_max_excess']) for e in ee),default=Decimal(0))),'exact_excess_histogram':dict(collections.Counter(e['decimal_max_excess'] for e in ee))}
    overlap_review=[]
    for r in results:
        for o in r['overlaps']:
            overlap_review.append({'image':r['image'],'label':r['label'],**o})
    overlap_review.sort(key=lambda o:(o['same_class'],-o['iou'],o['image']))
    overlap_path=save('_all_high_overlap_pairs',overlap_review)
    report={'scope':{'project_root':str(ROOT),'allowed_data_splits':['train','val'],'project_writes':False,'model_used':False,'new_dependencies':False,'visual_review_performed':False},'versions':{'python':sys.version,'executable':sys.executable,'pillow':PIL.__version__,'numpy':np.__version__},'parameters':{'workers':8,'iou_threshold':0.9,'dhash_bits':64,'dhash_resize':[9,8],'dhash_grayscale':'Pillow L','dhash_resampling':'LANCZOS','dhash_comparison':'right > left','dhash_exif_transpose':False,'hamming_threshold':4,'geometry_checks':'finite; 5 columns; integer class 0..3; all xywh in [0,1]; w,h>0; xyxy edges in [0,1] strict and severity >1e-6','duplicate_definition':'stripped text exactly equal; numeric equal also recorded separately','iou_geometry':'unclipped normalized xyxy; finite valid class and positive width/height; includes exact duplicates'},'inventory':inventory,'manifest_records':len(manifest),'known_pairs_input':len(known),'known_pairs_excluded_count':len(excluded),'known_pairs_excluded':excluded,'label_decode_summary':totals,'near_duplicates':{'compared_cross_pairs':len(trains)*len(vals),'candidate_pairs_after_known_exclusion':len(candidates),'distance_histogram':dict(sorted(collections.Counter(str(c['distance']) for c in candidates).items())),'unique_train_images':len({c['train'] for c in candidates}),'unique_val_images':len({c['val'] for c in candidates}),'same_video_prefix_pairs':sum(c['same_video_prefix'] for c in candidates),'top20_for_visual_review':top},'overlap_visual_review_first20':overlap_review[:20],'timing_seconds':{'decode_labels':decode_secs,'total':time.time()-start},'artifacts':{'script':str(Path(__file__).resolve()),'all_per_image_records':full_path,'all_near_duplicate_candidates':candidates_path,'top20_review':top_path,'all_high_overlap_pairs':overlap_path,'all_decimal_edge_overflow_details':edge_path},'limitations':['dHash proximity is not proof of identical images or leakage; fixed-camera scenes may generate false positives.','No visual inspection performed in this supplementary pass; no semantic wrong/missing-label conclusion.','Known 16 identical-image pairs and 2 empty validation labels not visually rechecked.','No models, upstream-source analysis, test split access, dataset/label/split/config edits, or 6luna.']}
    rp=save('_report',report)
    print(json.dumps({'report':rp,'label_decode_summary':totals,'candidates':report['near_duplicates']['candidate_pairs_after_known_exclusion'],'distances':report['near_duplicates']['distance_histogram'],'excluded_known':len(excluded),'seconds':report['timing_seconds']},ensure_ascii=False), flush=True)

if __name__=='__main__':
    main()