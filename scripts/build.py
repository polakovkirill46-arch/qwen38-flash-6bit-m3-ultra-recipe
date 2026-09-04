#!/usr/bin/env python3
"""Fetch pinned oMLX source, apply checked patch, and rebuild the MLX native ABI."""
import argparse, hashlib, json, os, platform, shutil, subprocess, sys, tarfile, urllib.request
from pathlib import Path
from common import REPO, outside, sha, write_json, runtime_snapshot
URL='https://github.com/jundot/omlx/archive/refs/tags/v0.6.4.tar.gz'
SHA='5d8781c0c6a782e9b90f071d36b0d8dbe6161fafb5fef5e3f0d8ac71da214b70'
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--app',required=True,type=Path,help='Official oMLX 0.6.4.app')
    p.add_argument('--python',required=True,type=Path,help='Native arm64 Python 3.11 with development headers')
    p.add_argument('--state',required=True,type=Path,help='New empty installation directory outside model/repository')
    a=p.parse_args(); a.app=a.app.expanduser().resolve(); a.python=a.python.expanduser().resolve(); a.state=a.state.expanduser().resolve()
    if platform.system()!='Darwin' or platform.machine()!='arm64':p.error('macOS arm64 required')
    outside(a.state,REPO); outside(a.state,a.app)
    res=a.app/'Contents/Resources'; runtime=res/'Python/cpython-3.11/bin/python3'
    if not runtime.is_file() or not (res/'omlx/_version.py').is_file():p.error('Official bundle dependencies missing')
    version=(res/'omlx/_version.py').read_text()
    if '0.6.4' not in version:p.error('Only oMLX 0.6.4 is supported')
    probe=json.loads(subprocess.check_output([str(a.python),'-c','import json,sys,sysconfig,platform;print(json.dumps([list(sys.version_info[:2]),sysconfig.get_path("include"),platform.machine()]))']))
    if probe[0]!=[3,11] or probe[2]!='arm64' or not (Path(probe[1])/'Python.h').is_file():p.error('Python 3.11 arm64 development headers required')
    for tool in ['uv','cmake','xcrun','patch']:
        if not shutil.which(tool):p.error(f'{tool} is required')
    subprocess.run(['xcrun','--find','metal'],check=True,stdout=subprocess.DEVNULL)
    a.state.mkdir(parents=True,exist_ok=False)
    archive=a.state/'upstream.tar.gz';urllib.request.urlretrieve(URL,archive)
    if sha(archive)!=SHA:raise RuntimeError('Upstream archive SHA256 mismatch')
    with tarfile.open(archive) as tar:
        for m in tar.getmembers():
            dest=(a.state/m.name).resolve()
            if a.state not in dest.parents or m.issym() or m.islnk():raise ValueError('Unsafe upstream archive member')
        tar.extractall(a.state)
    source=a.state/'source';(a.state/'omlx-0.6.4').rename(source)
    hashes=json.loads((REPO/'patches/hashes.json').read_text())
    for rel,h in hashes.items():
        if sha(source/rel)!=h['before']:raise RuntimeError(f'Unrecognized upstream file: {rel}')
    subprocess.run(['patch','--batch','--forward','-p1','-i',str(REPO/'patches/qualified.patch')],cwd=source,check=True)
    for rel,h in hashes.items():
        if sha(source/rel)!=h['after']:raise RuntimeError(f'Patched hash mismatch: {rel}')
    subprocess.run(['uv','pip','install','--python',str(a.python),'--target',str(a.state/'mlx'),'mlx==0.32.2','mlx-metal==0.32.2'],check=True)
    subprocess.run(['uv','pip','install','--python',str(a.python),'--target',str(a.state/'builddeps'),'setuptools==84.0.0','wheel==0.48.0','setuptools-scm==10.2.0','nanobind==2.15.0','ninja==1.13.2'],check=True)
    builder=source/'recipe_build_native.py'
    builder.write_text('import setup as config\nfrom setuptools import setup\nkw=config._custom_kernel_build_kwargs()\nkw["ext_modules"]=[m for m in kw["ext_modules"] if any(n in m.name for n in ["decode_fast","glm_moe_dsa","qwen35_prefill"])]\nsetup(**kw)\n')
    env=os.environ.copy()
    for k in ['PYTHONHOME','PYTHONPATH','CMAKE_ARGS']:env.pop(k,None)
    env.update(PYTHONPATH=os.pathsep.join(map(str,[a.state/'mlx',a.state/'builddeps',res/'Python/framework-mlx-base/lib/python3.11/site-packages'])),OMLX_WITH_CUSTOM_KERNEL='1',CMAKE_BUILD_PARALLEL_LEVEL='2',PATH=str(a.state/'builddeps/bin')+os.pathsep+env['PATH'])
    subprocess.run([str(a.python),str(builder),'build_ext','--inplace'],cwd=source,env=env,check=True)
    for name in ['decode_fast','glm_moe_dsa','qwen35_prefill']:
        d=source/'omlx/custom_kernels'/name
        for pattern in ['_ext*.so','*.dylib','*.metallib']:
            if not list(d.glob(pattern)):raise RuntimeError(f'Missing rebuilt {name}/{pattern}')
    receipt={'state':str(a.state),'app':str(a.app),'python':str(runtime),'source_archive_sha256':SHA,'omlx':'0.6.4','mlx':'0.32.2','patched_hashes':hashes}
    receipt['runtime_snapshot']=runtime_snapshot(a.state,receipt)
    write_json(a.state/'installation.json',receipt)
    print('Built. Next: scripts/prepare_model.py --state ... --target ... --draft ...')
if __name__=='__main__':main()
