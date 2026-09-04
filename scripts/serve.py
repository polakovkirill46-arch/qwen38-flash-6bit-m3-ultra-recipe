#!/usr/bin/env python3
"""Serve the qualified recipe in its isolated base directory; never edit launchd."""
import argparse,json,os,socket
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

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',required=True,type=Path);p.add_argument('--port',type=int,default=8024);p.add_argument('--host',default='127.0.0.1');a=p.parse_args()
    root,data=state(a.state)
    fingerprint=artifact_fingerprint(root,data)
    if not (root/'numerical-checks.json').is_file():p.error('Run numerical_checks.py first on an otherwise idle machine')
    validate_numerical_receipt(json.loads((root/'numerical-checks.json').read_text()),fingerprint)
    if not 1024<=a.port<=65535:p.error('Choose an unprivileged TCP port')
    check_port(a.host,a.port)
    env=environment(root,data)
    args=[data['python'],'-m','omlx.cli','serve','--base-path',str(root/'base'),'--model-dir',str(root/'models'),'--host',a.host,'--port',str(a.port),'--no-hf-cache','--max-concurrent-requests','1','--memory-guard-gb','160','--no-cache','--hot-cache-max-size','0']
    os.chdir(root);os.execve(data['python'],args,env)
if __name__=='__main__':main()
