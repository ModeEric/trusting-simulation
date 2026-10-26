"""Download pinned public outcome files; verify bytes before any analysis."""
import argparse
import concurrent.futures
import gzip
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/libero_plus'
REVISION = '554a0d06924f64f15551119e393373a9bc008771'
BASE = f'https://raw.githubusercontent.com/Alakazam-studios/targeted-data-libero-plus/{REVISION}/'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def download(path):
    dest = RAW / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with urllib.request.urlopen(BASE + path, timeout=60) as r:
            data = r.read()
        dest.write_bytes(data)
    return dest

def verify():
    provenance=json.loads((RAW/'provenance.json').read_text())
    assert provenance['source_revision']==REVISION
    for record in provenance['files']:
        assert sha((RAW/record['path']).read_bytes())==record['sha256'],record['path']
    manifest = json.loads((RAW / 'results/MANIFEST.json').read_text())
    records = []
    reference = None
    for f in manifest['files']:
        path = RAW / 'results' / f['file']
        data = path.read_bytes()
        assert sha(data) == f['file_sha256'], path
        rows = gzip.decompress(data)
        assert sha(rows) == f['rows_sha256_uncompressed'], path
        parsed = [json.loads(line) for line in rows.splitlines()]
        ids = [r['case_id'] for r in parsed]
        assert len(ids) == f['cases'] == len(set(ids)) == 10030
        assert all(type(r['success']) is bool for r in parsed)
        metadata = [(r['case_id'], r['suite'], r['axis'], r['level']) for r in parsed]
        if reference is None:
            reference = metadata
        assert metadata == reference
        axis = sorted(set(r['axis'] for r in parsed))
        rates = {a: sum(r['success'] for r in parsed if r['axis'] == a) / sum(r['axis'] == a for r in parsed) for a in axis}
        mean = sum(rates.values()) / len(rates) * 100
        assert abs(mean - f['seven_axis_mean_pct']) <= .00051, (f['file'],mean)
        records.append(dict(file=f['file'], sha256=sha(data), n_rows=len(ids), recomputed_score=mean))
    return manifest, records

def main():
    p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');args=p.parse_args()
    if args.download:
        for path in ['README.md','NOTICE','LICENSE','CLAIMS.md','results/README.md','results/MANIFEST.json','results/recompute.py']:
            download(path)
        manifest=json.loads((RAW/'results/MANIFEST.json').read_text())
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(download,['results/'+f['file'] for f in manifest['files']]))
        if not (RAW/'provenance.json').exists():
            provenance=dict(source_url=BASE,source_revision=REVISION,accessed_utc=datetime.now(timezone.utc).isoformat(),
                files=[dict(path=str(p.relative_to(RAW)),sha256=sha(p.read_bytes()),size=p.stat().st_size) for p in sorted(RAW.rglob('*')) if p.is_file()])
            (RAW/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    manifest,records=verify()
    (ROOT/'results/libero_plus_ingestion.json').write_text(json.dumps(dict(status='passed',source_revision=REVISION,
        passes=len(records),outcome_records=sum(r['n_rows'] for r in records),records=records),indent=2)+'\n')
    print(f'Verified {len(records)} passes, {sum(r["n_rows"] for r in records):,} outcomes; all published means reproduced.')

if __name__=='__main__':
    main()
