"""Offline SIMPLER and LIBERO-Plus reproduction using only licensed bundled inputs."""
import json
import os
import subprocess
import sys
import time
from .common import ROOT, RESULTS, verify_raw, write_json


def main():
    start = time.perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / 'figures').mkdir(parents=True, exist_ok=True)
    (ROOT / 'data/processed').mkdir(parents=True, exist_ok=True)
    status = RESULTS / 'public_validation.json'
    stages = []
    write_json(status, dict(status='running', stages=stages))
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    try:
        records = json.loads((ROOT / 'data/release_inputs.json').read_text())['files']
        verify_raw(names=[r['path'] for r in records if r['bundled']])
        commands = [
            ['src.ingest_simpler'], ['src.evaluate', '--dataset', 'simpler', '--cv', 'leave-one-task-out'],
            ['src.ingest_libero_plus'], ['src.libero_plus_experiment'], ['src.libero_plus_report'],
            ['unittest', '-v', 'tests.test_metrics', 'tests.test_no_leakage', 'tests.test_libero_plus',
             'tests.test_release_inputs', 'tests.test_reproduction_status'],
        ]
        for command in commands:
            print('Running ' + ' '.join(command), flush=True)
            stage_start = time.perf_counter()
            result = subprocess.run([sys.executable, '-m', *command], cwd=ROOT, env=environment,
                                    capture_output=True, text=True)
            (RESULTS / (command[0].split('.')[-1] + '_public.log')).write_text(result.stdout + result.stderr)
            stages.append(dict(command='python -m ' + ' '.join(command), returncode=result.returncode,
                               seconds=round(time.perf_counter() - stage_start, 3)))
            if result.returncode:
                print(result.stdout + result.stderr, flush=True)
                raise RuntimeError('Stage failed: ' + command[0])
    except Exception as exc:
        write_json(status, dict(status='failed', error=str(exc), stages=stages))
        raise
    write_json(status, dict(status='passed', scope=['SIMPLER', 'LIBERO-Plus'], stages=stages,
                            elapsed_seconds=round(time.perf_counter() - start, 3),
                            network_downloads_in_reproduction=0, new_robot_trials=0,
                            new_simulation_rollouts=0, policy_inference_calls=0, paid_api_calls=0))
    print('Public-data reproduction passed. See results/public_validation.json.', flush=True)


if __name__ == '__main__':
    main()
