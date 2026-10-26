#!/usr/bin/env python3
"""Acquire only missing research inputs and require the archived SHA-256 bytes.

This is the explicit network step. src.reproduce itself remains offline.
An original private archive can supply the same bytes without network access.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 32_000_000


def digest(data):
    return hashlib.sha256(data).hexdigest()


def destination(root, name):
    root = root.resolve()
    path = (root / name).resolve()
    if not name.startswith('data/raw/') or not path.is_relative_to(root / 'data/raw'):
        raise ValueError(f'Unsafe input path: {name}')
    return path


def acquire(record, root=ROOT, archive=None, verify_only=False):
    path = destination(root, record['path'])
    if path.exists():
        if digest(path.read_bytes()) != record['sha256']:
            raise ValueError(f'Existing input has changed; preserved for inspection: {record["path"]}')
        return 'verified'
    if verify_only:
        raise ValueError(f'Missing input: {record["path"]}')
    if record['bundled']:
        raise ValueError(f'Missing bundled input; restore the release checkout: {record["path"]}')
    if archive:
        with ZipFile(archive) as recovered:
            member = 'trusting-simulation/' + record['path']
            if recovered.getinfo(member).file_size > MAX_BYTES:
                raise ValueError(f'Archive input exceeds size cap: {member}')
            data = recovered.read(member)
    else:
        request = urllib.request.Request(record['url'], headers={'User-Agent': 'TrustingSimulationAudit/0.1'})
        with urllib.request.urlopen(request, timeout=45) as response:
            data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES or len(data) != record['bytes'] or digest(data) != record['sha256']:
        raise ValueError(f'Input differs from the October 2026 snapshot: {record["path"]}. No replacement was written.')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.download')
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return 'acquired'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Original Trusting_Simulation_Analysis.zip; no network access')
    parser.add_argument('--verify-only', action='store_true', help='Check local files without downloading')
    args = parser.parse_args()
    records = json.loads((ROOT / 'data/release_inputs.json').read_text())['files']
    failures = []
    def run(record):
        try:
            return record['path'], acquire(record, archive=args.archive, verify_only=args.verify_only), None
        except Exception as exc:
            return record['path'], 'FAILED', str(exc)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, status, error in pool.map(run, records):
            if status != 'verified':
                print(f'{status}: {name}' + (f': {error}' if error else ''), flush=True)
            if error:
                failures.append(name)
    if failures:
        raise SystemExit(f'{len(failures)} inputs unavailable. Hashes were not relaxed. See docs/reproduction.md.')
    print(f'All {len(records)} inputs match their recorded SHA-256 hashes.')


if __name__ == '__main__':
    main()
