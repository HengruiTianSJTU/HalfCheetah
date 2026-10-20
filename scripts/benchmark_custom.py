"""Check custom PPO learning and compare transport using total CPU-time efficiency."""
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import yaml

results=[]
for backend in ['sync','async']:
    name='pilot_'+backend+'_100k'
    path=Path('results')/name
    if path.exists():
        raise RuntimeError(f'Preserve existing pilot {path}; choose a new name before repeating')
    Path('raw_logs').mkdir(exist_ok=True)
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    start=time.perf_counter()
    with (Path('raw_logs')/(name+'.log')).open('w') as log:
        subprocess.run([sys.executable,'-m','halfcheetah.train','--config','configs/baseline.yaml','--name',name,'--total-timesteps','100000','--vec-backend',backend],stdout=log,stderr=subprocess.STDOUT,check=True)
    elapsed=time.perf_counter()-start;after=resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
    metadata=json.loads((path/'metadata.json').read_text())
    row=dict(vec_backend=backend,wall_clock=elapsed,total_cpu_seconds=cpu,cpu_equivalent=cpu/elapsed,training_fps=metadata['actual_timesteps']/metadata['wall_clock_time'],end_to_end_fps=metadata['actual_timesteps']/elapsed,steps_per_cpu_second=metadata['actual_timesteps']/cpu)
    results.append(row);print(json.dumps(row),flush=True)
Path('results/custom_benchmark.json').write_text(json.dumps(results,indent=2))
