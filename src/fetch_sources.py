"""Explicit, size-limited public download helper. Analysis itself is offline.

Only acquire URLs intentionally listed by the researcher. Never run upstream
code or fetch trajectory archives. Append provenance for every attempt.
"""
import argparse, datetime, hashlib, json, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def fetch(url, relative_path, revision=None, limit=32_000_000):
    path = ROOT / relative_path
    record = dict(file=relative_path, url=url, source_revision=revision,
                  accessed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    try:
        request = urllib.request.Request(url, headers={'User-Agent':'TrustingSimulationResearch/0.1'})
        with urllib.request.urlopen(request, timeout=25) as response:
            data=response.read(limit+1)
            record['http_status']=response.status
        if len(data)>limit: raise ValueError('Small-file size cap exceeded')
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
        record.update(status='downloaded',bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    except Exception as error:
        record.update(status='unavailable',error=str(error))
    return record

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',required=True);parser.add_argument('--path',required=True)
    parser.add_argument('--revision');args=parser.parse_args()
    record=fetch(args.url,args.path,args.revision)
    audit=ROOT/'data/download_audit.json'
    records=json.loads(audit.read_text()) if audit.exists() else []
    records.append(record);audit.write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(record,indent=2))
    if record['status']!='downloaded': raise SystemExit(1)
if __name__=='__main__': main()
