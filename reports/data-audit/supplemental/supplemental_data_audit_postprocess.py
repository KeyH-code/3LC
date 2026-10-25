import collections, json, os, sys
from pathlib import Path
from decimal import Decimal
base=Path(os.environ['PI_SCRATCH_DIR'])
prefix='supplemental_data_audit'
def load(name): return json.loads((base/(prefix+name+'.json')).read_text(encoding='utf-8'))
report=load('_report')
edges=sorted((e for e in load('_edge_overflow_decimal') if e['decimal_beyond_1e_6']),key=lambda e:-Decimal(e['decimal_max_excess']))
near=load('_top20_review')[:3]
files=[]
for p in near:
    for split in ('train','val'):
        files.append({'image':p[split],'reason':'cross_split_dhash_distance_0_not_visually_confirmed','pair':p})
seen={r['image'] for r in files}
for e in edges:
    if e['image'] not in seen and len(files)<20:
        files.append({'image':e['image'],'reason':'largest_decimal_bbox_edge_overflow_not_semantic_mislabel_conclusion','overflow':e})
        seen.add(e['image'])
queue_path=base/(prefix+'_review_files_max20.json')
queue_path.write_text(json.dumps(files,ensure_ascii=False,indent=2),encoding='utf-8')
report['artifacts']['prioritized_review_files_max20']=str(queue_path)
report['artifacts']['scratch_postprocess_script']=str(Path(__file__).resolve())
report['label_decode_summary']['val']['edge_overflow_decimal_summary']['exact_beyond_1e_6_images']=len({e['image'] for e in edges})
report['label_decode_summary']['val']['edge_overflow_decimal_summary']['largest_overflow_pixels_at_640']=float(max(Decimal(e['decimal_max_excess']) for e in edges)*640)
report['execution']={'audit_command':f'& "{sys.executable}" -B "{base / (prefix+".py")}"','postprocess_command':f'& "{sys.executable}" -B "{Path(__file__).resolve()}"','audit_exit_code':int((base/(prefix+'_exitcode.txt')).read_text(encoding='utf-8')),'audit_execution_log':str(base/(prefix+'_execution.txt')),'audit_exit_code_file':str(base/(prefix+'_exitcode.txt'))}
(base/(prefix+'_report.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({'review_queue':str(queue_path),'review_file_count':len(files),'large_edge_overflow_box_count':len(edges),'large_edge_overflow_image_count':len({e['image'] for e in edges}),'largest5':edges[:5]},ensure_ascii=False))