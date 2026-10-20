"""Create an isolated code-only workspace so committed reference metrics cannot skip new runs."""
import argparse
import re
import shutil
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',default='reproduction')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.name):
        parser.error('name must contain only letters, digits, underscore or hyphen')
    root=Path.cwd();target=root/'.local'/args.name
    target.mkdir(parents=True,exist_ok=False)
    for name in ['src','scripts','configs','tests','docs']:
        shutil.copytree(root/name,target/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info','.pytest_cache'))
    for name in ['README.md','pyproject.toml','requirements.lock.txt','.gitignore']:
        shutil.copy2(root/name,target/name)
    (target/'tutorial').mkdir()
    shutil.copy2(root/'tutorial/template.html',target/'tutorial/template.html')
    print(f'Created code-only reproduction workspace: {target}')
    print('Reference results were not copied; existing experiments were preserved.')


if __name__=='__main__':
    main()
