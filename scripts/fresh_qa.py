"""Validate the committed project in a fresh checkout and project-dependency environment.

The supplied base PyTorch/CUDA is deliberately reused; project dependencies are
installed into a new venv. No existing directory is overwritten.
"""
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    root=Path.cwd();scratch=root/'.local'/'fresh_qa'
    scratch.mkdir(parents=True,exist_ok=False)
    checkout=scratch/'repo';venv=scratch/'venv'
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    subprocess.run(['git','clone','--no-hardlinks',str(root),str(checkout)],check=True)
    subprocess.run([sys._base_executable,'-m','venv','--system-site-packages',str(venv)],check=True)
    python=str(venv/'bin'/'python');start=time.perf_counter()
    env=dict(os.environ,MUJOCO_GL='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    commands=[
        [python,'-m','pip','install','-i','https://pypi.org/simple','-r','requirements.lock.txt'],
        [python,'-m','pip','install','--no-deps','-e','.'],
        [python,'-m','pip','check'],
        [python,'-m','pytest','-q'],
        [python,'scripts/diagnostics.py'],
        [python,'scripts/smoke.py','--video'],
        [python,'-m','halfcheetah.train','--config','configs/qa.yaml','--name','qa_fresh'],
        [python,'scripts/evaluate.py','results/qa_fresh/checkpoint.pt','--episodes','2','--output','results/qa_fresh/independent_eval.csv'],
        [python,'scripts/report.py'],
    ]
    out=root/'results'/'qa';out.mkdir(parents=True,exist_ok=True)
    with (out/'fresh_environment.log').open('w') as log:
        for command in commands:
            print('FRESH_QA',command,flush=True)
            log.write('\nCOMMAND '+repr(command)+'\n');log.flush()
            subprocess.run(command,cwd=checkout,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    versions=json.loads(subprocess.check_output([python,'-c','import json, importlib.metadata as m; print(json.dumps({p:m.version(p) for p in ["torch","gymnasium","mujoco","stable-baselines3","numpy"]}))'],env=env,text=True))
    report=dict(git_commit=revision,fresh_checkout=True,new_project_dependency_venv=True,reused_base_torch_cuda=True,commands_passed=len(commands),versions=versions,wall_clock=time.perf_counter()-start)
    (out/'fresh_environment.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':
    main()
