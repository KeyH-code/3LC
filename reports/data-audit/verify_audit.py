"""Rerun read-only train/val audits; write derived evidence only beside this file."""
from pathlib import Path
from datetime import datetime
from decimal import Decimal
from collections import Counter
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUTPUT = HERE / 'verification.json'
SCRIPTS = ('cleaning_audit', 'cleaning_audit_labels', 'cleaning_audit_report')
EXPECTED_PYTHON = Path(r'E:\anaconda\envs\dl-3lc\python.exe')


def now():
    return datetime.now().astimezone().isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(), sys.executable
    result = {'started': now(), 'status': 'running', 'python': sys.executable,
              'scope': str(ROOT), 'runs': [], 'checks': {}, 'artifacts': []}
    try:
        env = os.environ.copy()
        env['PYTHONUTF8'] = '1'
        for name in SCRIPTS:
            cmd = [sys.executable, '-B', '-W', 'error', str(HERE / (name + '.py'))]
            started = now()
            proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                                  encoding='utf-8', errors='strict', timeout=180)
            (HERE / ('run_' + name + '.log')).write_text(proc.stdout, encoding='utf-8')
            (HERE / ('run_' + name + '.stderr.log')).write_text(proc.stderr, encoding='utf-8')
            result['runs'].append({'command': cmd, 'started': started, 'ended': now(),
                                   'exit_code': proc.returncode,
                                   'stderr_bytes_utf8': len(proc.stderr.encode('utf-8'))})
            assert proc.returncode == 0, (name, proc.returncode, proc.stderr)
            assert not proc.stderr, (name, proc.stderr)

        audit = json.loads((HERE / 'cleaning_audit.json').read_text(encoding='utf-8'))
        assert audit['verified_files'] == audit['manifest_files'] == audit['actual_files'] == 17668
        assert audit['errors'] == []
        assert audit['images'] == {'train': 7852, 'val': 982}
        assert audit['boxes'] == {'train': 72273, 'val': 11615}
        pairs = audit['pairs']
        assert len(pairs) == audit['cross_hash_groups'] == 16
        assert audit['independent_manifest_groups_confirmed'] == 16
        assert audit['direct_byte_equality_all_16_pairs']
        for pair in pairs:
            assert pair['train']['sha256'] == pair['val']['sha256']
            assert pair['pixels_equal'] and pair['rgb_pixels_equal']
            for side in ('train_decode', 'val_decode'):
                assert pair[side]['size'] == [640, 640]
                assert pair[side]['mode'] == 'RGB' and pair[side]['format'] == 'JPEG'
            labels = pair['labels']
            assert not any(labels[k] for k in ('exact_bytes_equal', 'exact_text_equal',
                                               'line_numeric_equal', 'unordered_numeric_equal'))
            assert labels['val_all_tokens_six_decimals']
            if labels['class_counts_equal']:
                cp = labels['class_preserving_rectangular']
                assert not cp['unmatched_train'] and not cp['unmatched_val']
                assert Decimal(cp['max_abs_delta']) <= Decimal('0.0000005')
        assert sum(p['labels']['class_counts_equal'] for p in pairs) == 4
        assert Counter(p['train']['video'] for p in pairs) == {'MVI_40751': 5, 'MVI_40752': 11}
        videos = audit['video_summary']
        assert len(videos) == 100 and sum(bool(x['train'] and x['val']) for x in videos) == 99
        assert len(audit['cross_original_id_groups']) == 16
        assert sum(x['different_hash_pairs'] for x in audit['cross_original_id_groups']) == 0
        empty = audit['empty_labels']
        assert {e['image']['original_id'] for e in empty} == {
            'MVI_40172_img00696', 'MVI_40172_img02570'}
        assert all(e['image']['split'] == 'val' and e['image']['label_bytes'] == 0
                   and not e['train_same_hash'] and not e['train_same_original_id'] for e in empty)
        result['checks']['manifest_and_report_assertions'] = 'passed'
        result['checks']['verified_files'] = audit['verified_files']

        for path in sorted(HERE.glob('*.py')):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        result['checks']['python_ast'] = 'passed'
        text_paths = [ROOT / 'PROJECT_LOG.md']
        text_paths += [p for p in HERE.iterdir() if p.suffix in ('.py', '.md', '.html', '.json', '.log')
                       and p != OUTPUT]
        for path in text_paths:
            text = path.read_text(encoding='utf-8')
            assert '\ufffd' not in text and '\x00' not in text, path
            assert not re.search(r'^(?:<<<<<<< |=======$|>>>>>>> )', text, re.M), path
            assert all(line.rstrip() == line for line in text.splitlines()), path
            if path.suffix == '.md' and path.parent == HERE:
                for target in re.findall(r'\]\(([^)]+)\)', text):
                    if not target.startswith(('https://', 'http://', '#')):
                        resolved = (HERE / target.split('#')[0]).resolve()
                        assert resolved == OUTPUT or resolved.is_file(), (path, target)
        result['checks']['utf8_conflict_whitespace_local_links'] = 'passed'

        evidence = sorted((HERE / 'evidence').glob('*.jpg'))
        assert len(evidence) == 7
        for path in evidence:
            with Image.open(path) as image:
                image.load()
                assert image.width > 0 and image.height > 0
                result['artifacts'].append({'path': str(path.relative_to(ROOT)),
                                            'bytes': path.stat().st_size, 'sha256': sha(path),
                                            'size': list(image.size)})
        result['checks']['seven_archived_evidence_images_decode'] = 'passed'
        for name in [s + '.py' for s in SCRIPTS] + ['verify_audit.py', 'SUMMARY.md',
                     'CLEANING_AUDIT.md', 'index.html', 'cleaning_audit.json']:
            path = HERE / name
            result['artifacts'].append({'path': str(path.relative_to(ROOT)),
                                        'bytes': path.stat().st_size, 'sha256': sha(path)})
        result['status'] = 'passed'
    except Exception as exc:
        result['status'] = 'failed'
        result['error'] = repr(exc)
        raise
    finally:
        result['ended'] = now()
        OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'checks': result['checks'],
                      'output': str(OUTPUT)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
