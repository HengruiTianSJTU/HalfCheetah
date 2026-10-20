"""Run under tmux; preserves successful cases and propagates failures."""
import json
import subprocess
import sys
from pathlib import Path

cases=[(d,n,'dummy') for d in ['cpu','cuda'] for n in [1,4,8]]+ [('cpu',n,'subproc') for n in [4,8]]
for device,n,vec in cases:
    name=f'{device}_n{n}_{vec}'
    out=Path('results/benchmark')/name
    if (out/'summary.json').exists():
        continue
    Path('raw_logs').mkdir(exist_ok=True)
    print('Starting',name,flush=True)
    with (Path('raw_logs')/f'benchmark_{name}.log').open('w') as log:
        subprocess.run([sys.executable,'scripts/benchmark.py','--device',device,'--n-envs',str(n),'--vec',vec],stdout=log,stderr=subprocess.STDOUT,check=True)
    print((out/'summary.json').read_text(),flush=True)
print('BENCHMARKS_COMPLETE',flush=True)
