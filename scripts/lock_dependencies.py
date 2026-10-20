"""Capture only the project's installed dependency closure, including chosen extras."""
from importlib import metadata
from pathlib import Path
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

roots=['gymnasium[mujoco]','mujoco','stable-baselines3','numpy','matplotlib','imageio[ffmpeg]','PyYAML','psutil','pytest']
queue=[Requirement(r) for r in roots]
seen=set();packages={}
while queue:
    requirement=queue.pop()
    key=(canonicalize_name(requirement.name),tuple(sorted(requirement.extras)))
    if key in seen:
        continue
    seen.add(key)
    dist=metadata.distribution(requirement.name)
    packages[canonicalize_name(dist.metadata['Name'])]=dist.version
    for raw in dist.requires or []:
        dependency=Requirement(raw)
        if dependency.marker is None or any(dependency.marker.evaluate({'extra':extra}) for extra in ['',*requirement.extras]):
            queue.append(dependency)
lines=['# Resolved on AutoDL Python 3.12; torch CUDA wheels use the additional index.', '--extra-index-url https://download.pytorch.org/whl/cu124']
lines += [f'{name}=={version}' for name,version in sorted(packages.items())]
Path('requirements.lock.txt').write_text('\n'.join(lines)+'\n')
print(f'Locked {len(packages)} packages')
