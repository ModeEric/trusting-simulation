import hashlib, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw'; PROCESSED=ROOT/'data/processed'; RESULTS=ROOT/'results'
PLAN=json.loads((ROOT/'analysis_plan.json').read_text())
SEED=PLAN['seed']

def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def validate_scores(frame, keys):
    if frame.duplicated(keys).any(): raise ValueError('Duplicate observation identity')
    if frame['score'].isna().any() or not frame['score'].between(0,1).all():
        raise ValueError('Scores must be finite and in [0,1]')

def verify_raw(names=None):
    records=json.loads((ROOT/'data/download_audit.json').read_text())
    latest={r['file']:r for r in records if r['status']=='downloaded'}
    release=ROOT/'data/release_inputs.json'
    required={r['path']:r for r in json.loads(release.read_text())['files']} if release.exists() else latest
    if names is not None:
        required={name:required[name] for name in names}
    for name,record in required.items():
        if not (ROOT/name).is_file():
            raise ValueError(f'Missing input: {name}. Run python scripts/acquire_inputs.py first.')
        if sha256(ROOT/name)!=record['sha256']:
            raise ValueError(f'Raw checksum mismatch: {name}')
    for name,digest in json.loads((ROOT/'data/plan_freeze.json').read_text()).items():
        if sha256(ROOT/name)!=digest: raise ValueError(f'Frozen plan changed: {name}')
    return {name:record for name,record in latest.items() if name in required}
