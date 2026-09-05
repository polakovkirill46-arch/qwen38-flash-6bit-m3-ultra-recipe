#!/usr/bin/env python3
"""Serve the qualified recipe in its isolated base directory; never edit launchd."""
import argparse,json,os,socket,tempfile
from pathlib import Path
from common import environment,state,artifact_fingerprint,validate_numerical_receipt

def check_port(host,port):
    # Match asyncio/uvicorn's address reuse so a stopped server's TIME_WAIT
    # connections do not prevent restart. SO_REUSEPORT is intentionally absent:
    # an existing listening server must still make this check fail.
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        sock.bind((host,port))
        sock.listen(1)

def cache_arguments(root, enabled):
    """Explicit CLI overrides prevent cache state sticking across restarts."""
    if enabled:
        return ['--paged-ssd-cache-dir', str(root/'prefix-cache'), '--paged-ssd-cache-max-size', '8GB']
    return ['--no-cache']

def configure_cache(root, enabled):
    """Pin measured cache policy while preserving unrelated saved settings."""
    cache_dir=root/'prefix-cache'
    if cache_dir.is_symlink() or cache_dir.resolve()!=root.resolve()/'prefix-cache':
        raise ValueError('Cache directory must remain inside this state directory')
    path=root/'base/settings.json'
    if path.is_symlink():raise ValueError('Settings must not be a symlink')
    saved=json.loads(path.read_text()) if path.exists() else {}
    policy=dict(saved.get('cache',{}))
    policy.update(enabled=enabled,hot_cache_only=False,ssd_cache_dir=str(cache_dir),
                  ssd_cache_max_size='8GB',hot_cache_max_size='0',hot_cache_write_through=False,
                  gdn_snapshot_storage='auto',gdn_ssd_split_enabled=False,
                  gdn_sidecar_precision='fp32',gdn_ssd_pending_max_size='512MB',
                  initial_cache_blocks=256,ane_compile_cache=False)
    saved['cache']=policy
    # Auto resolves to SSD sidecars when cache is enabled. The legacy false
    # field is ignored under auto by this pinned upstream version.
    fd,name=tempfile.mkstemp(prefix='.cache-settings-',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:json.dump(saved,f,indent=2);f.write('\n')
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',required=True,type=Path);p.add_argument('--port',type=int,default=8024);p.add_argument('--host',default='127.0.0.1');group=p.add_mutually_exclusive_group();group.add_argument('--prefix-cache',action='store_true',help='Enable qualified 8GB SSD prefix cache');group.add_argument('--no-cache',action='store_true',help='Disable prefix cache (default and rollback)');a=p.parse_args()
    root,data=state(a.state)
    fingerprint=artifact_fingerprint(root,data)
    if not (root/'numerical-checks.json').is_file():p.error('Run numerical_checks.py first on an otherwise idle machine')
    validate_numerical_receipt(json.loads((root/'numerical-checks.json').read_text()),fingerprint)
    if not 1024<=a.port<=65535:p.error('Choose an unprivileged TCP port')
    check_port(a.host,a.port)
    configure_cache(root,a.prefix_cache)
    env=environment(root,data)
    args=[data['python'],'-m','omlx.cli','serve','--base-path',str(root/'base'),'--model-dir',str(root/'models'),'--host',a.host,'--port',str(a.port),'--no-hf-cache','--max-concurrent-requests','1','--memory-guard-gb','160','--hot-cache-max-size','0'] + cache_arguments(root,a.prefix_cache)
    os.chdir(root);os.execve(data['python'],args,env)
if __name__=='__main__':main()
