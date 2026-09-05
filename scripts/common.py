"""Portable recipe paths and fail-closed filesystem guards (standard library only)."""
import hashlib, json, os
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()
def write_json(path, value):
    with Path(path).open('x') as f: json.dump(value, f, indent=2); f.write('\n')
def outside(destination, source):
    d, s = Path(destination).resolve(), Path(source).resolve()
    if d == s or s in d.parents or d in s.parents:
        raise ValueError('Runtime and source must be disjoint directories')
def state(path):
    p = Path(path).expanduser().resolve()
    data = json.loads((p/'installation.json').read_text())
    if Path(data['state']).resolve() != p: raise ValueError('Installation was moved; rebuild at final path')
    return p, data
def environment(path, data):
    res = Path(data['app'])/'Contents/Resources'
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('FLASH_', 'OMLX_', 'MLX_MAX_', 'R2_', 'R3_')): env.pop(key)
    env.update(json.loads((REPO/'runtime/qualified-env.json').read_text()))
    env.update(FLASH_RECIPE_ROOT=str(REPO), FLASH_OMLX_SOURCE=str(path/'source'),
               FLASH_MLX_PATH=str(path/'mlx'), PYTHONHOME=str(res/'Python/cpython-3.11'),
               PYTHONPATH=os.pathsep.join(map(str,[path/'native',REPO/'runtime',path/'source',path/'mlx',res,res/'Python/framework-mlx-base/lib/python3.11/site-packages'])),
               PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
    return env

# Receipts detect accidental drift. They are local evidence, not signed attestations.
RUNTIME_SUFFIXES={'.py','.so','.dylib','.metallib','.json'}
CORE_SCRIPTS=('common.py','build.py','prepare_model.py','serve.py','numerical_checks.py')
def inventory(directory, suffixes=None):
    directory=Path(directory)
    return {str(p.relative_to(directory)):sha(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and (suffixes is None or p.suffix in suffixes)}
def runtime_snapshot(root, data):
    return {'recipe_runtime':inventory(REPO/'runtime'),
            'recipe_native_source':inventory(REPO/'native'),
            'deferred_ple_native':inventory(root/'native'),
            'recipe_scripts':{name:sha(REPO/'scripts'/name) for name in CORE_SCRIPTS},
            'omlx':inventory(root/'source/omlx',RUNTIME_SUFFIXES),
            'mlx':inventory(root/'mlx',RUNTIME_SUFFIXES),
            'python_sha256':sha(Path(data['python']))}
def verify_runtime(root,data):
    expected=data.get('runtime_snapshot')
    if not expected:raise RuntimeError('Installation receipt predates integrity binding; rebuild in a fresh state directory')
    actual=runtime_snapshot(root,data)
    if actual!=expected:raise RuntimeError('Runtime, native library, recipe helper, table, or launcher drift; rebuild and recheck')
    return actual

def model_snapshot(root,receipt):
    target=Path(receipt['target']);view=root/'models'/receipt['model']
    expected_names=set(receipt['source_file_metadata'])|{'draft.safetensors'}
    if {p.name for p in view.iterdir()}!=expected_names:raise RuntimeError('Model view inventory changed')
    for name,st in receipt['source_file_metadata'].items():
        original=target/name;now=original.stat()
        if (now.st_size,now.st_mtime_ns,now.st_ino)!=(st['size'],st['mtime_ns'],st['inode']):raise RuntimeError('Original model metadata changed')
        if str(original.resolve())!=receipt['source_paths'][name]:raise RuntimeError('Original source file link changed')
        if name not in ('config.json','model.safetensors.index.json'):
            link=view/name
            if not link.is_symlink() or str(link.resolve())!=receipt['source_paths'][name]:raise RuntimeError('Model view shard link changed')
    draft=view/'draft.safetensors'
    if not draft.is_symlink() or draft.resolve()!=(root/'draft.safetensors').resolve():raise RuntimeError('Prepared draft link changed')
    for name in ['config.json','model.safetensors.index.json']:
        if (view/name).is_symlink():raise RuntimeError('Prepared metadata must be independent files')
    return {'draft_sha256':sha(root/'draft.safetensors'),
            'view_metadata':{name:sha(view/name) for name in ['config.json','model.safetensors.index.json']},
            'source_metadata':{name:sha(target/name) for name in ['config.json','model.safetensors.index.json']},
            'model_settings_sha256':sha(root/'base/model_settings.json'),
            'source_paths':receipt['source_paths'],'source_file_metadata':receipt['source_file_metadata']}
def artifact_fingerprint(root,data):
    runtime=verify_runtime(root,data)
    receipt=json.loads((root/'model-receipt.json').read_text())
    if 'prepared_snapshot' not in receipt:raise RuntimeError('Prepared model lacks integrity binding; prepare in a fresh state directory')
    model=model_snapshot(root,receipt)
    if model!=receipt['prepared_snapshot']:raise RuntimeError('Prepared draft, model metadata, or settings changed; reprepare and recheck')
    return hashlib.sha256(json.dumps({'runtime':runtime,'model':model},sort_keys=True,separators=(',',':')).encode()).hexdigest()
def validate_numerical_receipt(receipt,fingerprint):
    import math
    if receipt.get('success') is not True or receipt.get('artifact_fingerprint')!=fingerprint:raise RuntimeError('Numerical receipt is unsuccessful or stale')
    if receipt.get('deferred_ple_bit_exact') is not True:raise RuntimeError('Deferred PLE check missing')
    if receipt.get('mlx')!='0.32.2' or receipt.get('weighted10_bit_exact') is not True:raise RuntimeError('Numerical receipt has invalid result fields')
    if receipt.get('native_imports')!=['decode_fast','glm_moe_dsa','qwen35_prefill']:raise RuntimeError('Native checks incomplete')
    cells=receipt.get('cells',[])
    if [r.get('tokens') for r in cells]!=[3256,8192]:raise RuntimeError('Numerical cells incomplete')
    for r in cells:
        err=r.get('normalization_relative_rms',float('nan'))
        if r.get('mix_bit_exact') is not True or not isinstance(err,(int,float)) or not math.isfinite(err) or not 0<=err<1e-4:raise RuntimeError('Numerical tolerance check failed')
