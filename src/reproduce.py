"""One-command offline reproduction of all analysis, figures, note and tests."""
import json,subprocess,sys,time,os
from pathlib import Path
from .common import ROOT,RESULTS,verify_raw,write_json

def main():
    start=time.perf_counter();stages=[]
    for directory in [RESULTS, RESULTS/'figures', ROOT/'data/processed']:
        directory.mkdir(parents=True, exist_ok=True)
    # Never leave a previous successful status attached to a failed new run.
    write_json(RESULTS/'validation.json',dict(status='running',stages=stages))
    try:verify_raw()
    except Exception as exc:
        write_json(RESULTS/'validation.json',dict(status='failed',failure_stage='input_verification',error=str(exc),stages=stages))
        raise
    commands=[['src.ingest_simpler'],['src.evaluate','--dataset','simpler','--cv','leave-one-task-out'],
      ['src.blend_experiment'],['src.blend_report'],
      ['src.ingest_robodojo','--audit-only'],['src.sim_prediction'],['src.sim_prediction_report'],
      ['src.robotwin_prediction'],['src.robotwin_report'],
      ['src.robotwin_learning_curve'],['src.robotwin_learning_curve_report'],
      ['src.robotwin_model_sweep','--stage','1'],['src.robotwin_model_sweep','--stage','2'],
      ['src.robotwin_model_sweep','--stage','3'],['src.robotwin_model_sweep_report'],
      ['src.ingest_libero_plus'],['src.libero_plus_experiment'],['src.libero_plus_report'],
      ['src.ingest_robocurve'],['src.source_audit'],
      ['src.manifest'],['src.plots'],['src.render_note']]
    for cmd in commands:
        print('Running '+cmd[0],flush=True);t=time.perf_counter()
        environment=os.environ.copy()
        if 'libero_plus' in cmd[0]:environment.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        result=subprocess.run([sys.executable,'-m',*cmd],cwd=ROOT,capture_output=True,text=True,env=environment)
        stages.append(dict(command='python -m '+' '.join(cmd),returncode=result.returncode,seconds=round(time.perf_counter()-t,3)))
        (RESULTS/(cmd[0].split('.')[-1]+'_run.log')).write_text(result.stdout+result.stderr)
        if result.returncode:
            write_json(RESULTS/'validation.json',dict(status='failed',failure_stage=cmd[0],stages=stages))
            print(result.stdout+result.stderr);raise SystemExit(result.returncode)
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,capture_output=True,text=True)
    (RESULTS/'tests.log').write_text(test.stdout+test.stderr)
    print(test.stdout+test.stderr,flush=True)
    if test.returncode:
        write_json(RESULTS/'validation.json',dict(status='failed',failure_stage='tests',stages=stages))
        raise SystemExit(test.returncode)
    write_json(RESULTS/'validation.json',dict(status='passed',stages=stages,
        elapsed_seconds=round(time.perf_counter()-start,3),
        test_command='python -m unittest discover -s tests -v',test_runner='unittest',
        network_downloads_in_reproduction=0,new_robot_trials=0,new_simulation_rollouts=0,policy_inference_calls=0,paid_api_calls=0))
    print('Reproduction completed. See results/validation.json and results/tests.log.',flush=True)
if __name__=='__main__':main()
