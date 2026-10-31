"""Run the entropy ablation in a fresh workspace, never in reference-result directories."""
import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import yaml

ROOT = Path(__file__).resolve().parents[1]
COEFFICIENTS = (0., .001, .005, .01)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_cases(base, seeds, device, timesteps):
    return [dict(base, entropy_coef=e, seed=s, device=device, total_timesteps=timesteps)
            for e in COEFFICIENTS for s in seeds]


def name_of(config):
    return f"hc_entropy{config['entropy_coef']:g}_seed{config['seed']}"


def validate_reuse(path, expected, source_hashes, versions):
    """Require real, matching local weights; reference CSVs alone cannot skip training."""
    config = yaml.safe_load((path/'config.yaml').read_text())
    metadata = json.loads((path/'metadata.json').read_text())
    batch = expected['n_envs'] * expected['n_steps']
    actual = ((expected['total_timesteps'] + batch - 1)//batch)*batch
    if config != expected or metadata.get('status') != 'completed' or metadata.get('actual_timesteps') != actual:
        raise ValueError(f'Baseline configuration/budget/status mismatch: {path}')
    if metadata.get('source_sha256') != source_hashes or metadata.get('versions') != versions:
        raise ValueError(f'Baseline source/dependency versions differ: {path}')
    for filename in ['checkpoint.pt', 'metrics.csv', 'eval.csv', 'episodes.csv']:
        if not (path/filename).is_file():
            raise ValueError(f'Required baseline artifact missing: {path/filename}')
    import torch
    state = torch.load(path/'checkpoint.pt', map_location='cpu', weights_only=False)
    if state.get('config') != expected or state.get('timesteps') != actual:
        raise ValueError(f'Baseline checkpoint configuration/budget mismatch: {path}')
    if not state.get('model') or not all(torch.isfinite(v).all() for v in state['model'].values()):
        raise ValueError(f'Invalid baseline model tensors: {path}')
    for key in ['obs_rms', 'ret_rms']:
        if key not in state:
            raise ValueError(f'Missing baseline normalization: {path}')
    return sha(path/'checkpoint.pt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('.local/entropy_reproduction'),
                        help='New workspace directory; existing directories are refused')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--gpus', help='Visible CUDA device IDs, comma separated; omitted: inherit CUDA_VISIBLE_DEVICES')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument('--timesteps', type=int, default=2000000)
    parser.add_argument('--baseline-results', type=Path,
                        help='Explicit local results directory containing compatible entropy=0 weights')
    parser.add_argument('--dry-run', action='store_true', help='Print the plan without creating files or training')
    args = parser.parse_args()
    if args.workers < 1 or args.timesteps < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('workers/timesteps must be positive and seeds must be unique')
    gpu_ids = args.gpus.split(',') if args.gpus else []
    if gpu_ids and (args.device != 'cuda' or any(not g.isdigit() for g in gpu_ids)):
        parser.error('--gpus requires --device cuda and comma-separated integer IDs')
    base = yaml.safe_load((ROOT/'configs/entropy/entropy0.yaml').read_text())
    cases = make_cases(base, args.seeds, args.device, args.timesteps)
    work = args.output.resolve()
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT/'src').rglob('*.py'))}
    versions = {p: importlib.metadata.version(p) for p in ['torch', 'gymnasium', 'mujoco', 'numpy', 'stable-baselines3']}
    reuse = {}
    if args.baseline_results:
        for config in [c for c in cases if c['entropy_coef'] == 0]:
            parent = args.baseline_results.resolve()
            candidates = [parent/name_of(config), parent/f"hc_lr0.0003_clip0.2_seed{config['seed']}"]
            source = next((p for p in candidates if p.is_dir()), candidates[0])
            digest = validate_reuse(source, config, source_hashes, versions)
            reuse[name_of(config)] = dict(source=str(source), checkpoint_sha256=digest)
    plan = dict(entropy_coefficients=list(COEFFICIENTS), training_seeds=args.seeds,
                fixed_config=dict(base, device=args.device, total_timesteps=args.timesteps),
                source_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
                source_sha256=source_hashes, versions=versions, output=str(work), workers=args.workers,
                gpus=gpu_ids, baseline_reuse=reuse, validation_seeds=list(range(10000,10005)),
                held_out_test_used=False, matrix=cases)
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return
    if work.exists():
        parser.error(f'Existing directory preserved: {work}; choose a new --output')
    if args.device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            parser.error('CUDA is unavailable; select --device cpu or install an appropriate PyTorch build')
    work.mkdir(parents=True)
    shutil.copytree(ROOT/'src', work/'src', ignore=shutil.ignore_patterns('__pycache__'))
    for directory in ['results', 'configs', 'raw_logs']:
        (work/directory).mkdir()
    (work/'results/protocol.json').write_text(json.dumps(plan, indent=2))
    for config in cases:
        (work/'configs'/f'{name_of(config)}.yaml').write_text(yaml.safe_dump(config))
    hashes = {}
    for name, record in reuse.items():
        shutil.copytree(record['source'], work/'results'/name)
        hashes[name] = record['checkpoint_sha256']
    (work/'results/checkpoint_hashes.json').write_text(json.dumps(hashes, indent=2))
    pending = [c for c in cases if name_of(c) not in reuse]
    status = dict(status='training', started_at=datetime.now(timezone.utc).isoformat(),
                  new_runs=len(pending), reused_runs=len(reuse))
    status_path = work/'status.json'
    status_path.write_text(json.dumps(status, indent=2))

    def train(item):
        index, config = item
        name = name_of(config)
        env = os.environ.copy()
        env.update(PYTHONPATH=str(work/'src'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
        if gpu_ids:
            env['CUDA_VISIBLE_DEVICES'] = gpu_ids[index % len(gpu_ids)]
        print('START', name, flush=True)
        with (work/'raw_logs'/f'{name}.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'halfcheetah.train', '--config', f'configs/{name}.yaml',
                            '--name', name], cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print('COMPLETE', name, flush=True)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in concurrent.futures.as_completed([pool.submit(train, item) for item in enumerate(pending)]):
                future.result()
        subprocess.run([sys.executable, str(ROOT/'scripts/report_entropy.py'), '--results', str(work/'results'),
                        '--figures', str(work/'figures'), '--html', str(work/'tutorial/entropy.html')], check=True)
        status.update(status='completed', finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        status.update(status='failed', error=repr(exc)); raise
    finally:
        status_path.write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
