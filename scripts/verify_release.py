#!/usr/bin/env python3
"""Check that released code, plans and bundled inputs match the tested snapshot."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    record = json.loads((ROOT / 'reports/verification.json').read_text())
    failures = []
    for name, expected in record['tested_files_sha256'].items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(name)
    if failures:
        raise SystemExit('Released files differ from the tested snapshot: ' + ', '.join(failures))
    print(f'All {len(record["tested_files_sha256"])} tested source/input files match the verification record.')


if __name__ == '__main__':
    main()
