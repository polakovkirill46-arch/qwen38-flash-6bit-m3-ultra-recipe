#!/usr/bin/env python3
"""Create a new model view and adapt a separate donor. Never write source tensors."""
import argparse, copy, json, math, shutil, struct
from pathlib import Path
from common import REPO, outside, sha, state, write_json, verify_runtime, model_snapshot
DONOR_SHA256='3498cbee477938de1dd3a42242387bb8fa6ad921053af010543473ab020f717c'
NORM_SUFFIXES=('hc_norm.weight','q_norm.weight','k_norm.weight','q_layernorm.weight','k_layernorm.weight')
DTYPES={'U32':4,'BF16':2,'F16':2,'F32':4,'I32':4,'I64':8,'U8':1}
def header(path):
    with path.open('rb') as f:
        raw=f.read(8)
        if len(raw)!=8:raise ValueError('Truncated safetensors prefix')
        n=struct.unpack('<Q',raw)[0]
        if not 2<=n<=64*1024*1024:raise ValueError('Unsafe safetensors header size')
        h=json.loads(f.read(n))
    end=0
    tensors=[(name,s) for name,s in h.items() if name!='__metadata__']
    for name,s in sorted(tensors,key=lambda item:item[1]['data_offsets'][0]):
        lo,hi=s['data_offsets']
        if s['dtype'] not in DTYPES or lo!=end or hi-lo!=math.prod(s['shape'])*DTYPES[s['dtype']]:raise ValueError('Unexpected or noncontiguous tensor layout')
        end=hi
    if 8+n+end!=path.stat().st_size:raise ValueError('Tensor file length mismatch')
    return n,h

def validate_config(config):
    t=config.get('text_config',{});q=config.get('quantization',{})
    required={'hidden_size':2560,'num_hidden_layers':48,'hc_count':4,'hc_lowrank':320,'num_experts':512,'num_experts_per_tok':10,'moe_intermediate_size':640,'rms_norm_eps':1e-6}
    if config.get('model_type')!='qwen4_exp' or any(t.get(k)!=v for k,v in required.items()):raise ValueError('Unsupported target architecture')
    if q.get('bits')!=6 or q.get('group_size')!=64 or q.get('mode')!='affine':raise ValueError('Original affine 6-bit/group64 target required')

def adapt(src,dest):
    if sha(src)!=DONOR_SHA256:raise ValueError('Donor SHA256 differs from qualified input; do not silently substitute a newer revision')
    n,h=header(src)
    expected=json.loads((REPO/'runtime/donor-schema.json').read_text())
    actual={k:{'dtype':v['dtype'],'shape':v['shape']} for k,v in h.items() if k!='__metadata__'}
    if actual!=expected:raise ValueError('Donor tensor names, shapes, or dtypes differ from qualified donor')
    out_header={};replacements={};quant={};offset=0
    with src.open('rb') as f:
        for name,old in h.items():
            if name=='__metadata__':continue
            s=copy.deepcopy(old);size=s['data_offsets'][1]-s['data_offsets'][0]
            if name.endswith(NORM_SUFFIXES):
                if s['dtype']!='BF16':raise ValueError('Expected BF16 direct gamma')
                f.seek(8+n+s['data_offsets'][0]);raw=f.read(size)
                values=[struct.unpack('<f',b'\0\0'+raw[i:i+2])[0]-1 for i in range(0,size,2)]
                replacements[name]=struct.pack('<'+'f'*len(values),*values)
                s['dtype']='F32';size=len(replacements[name])
            if name.endswith('.weight'):
                module=name[:-7]
                if module+'.scales' in h:
                    ratio=s['shape'][-1]/h[module+'.scales']['shape'][-1]
                    if ratio not in (4,16):raise ValueError('Unsupported draft quantization layout')
                    bits,group=(4,32) if ratio==4 else (8,64)
                    quant[module]={'bits':bits,'group_size':group,'mode':'affine'}
                else:quant[module]=False
            s['data_offsets']=[offset,offset+size];offset+=size;out_header[name]=s
    if len(replacements)!=7:raise ValueError('Expected exactly seven recentered norms')
    encoded=json.dumps(out_header).encode();encoded+=b' '*(-len(encoded)%8)
    with dest.open('xb') as out,src.open('rb') as f:
        out.write(struct.pack('<Q',len(encoded)));out.write(encoded)
        for name,s in out_header.items():
            if name in replacements:out.write(replacements[name]);continue
            lo,hi=h[name]['data_offsets'];f.seek(8+n+lo);remaining=hi-lo
            while remaining:
                data=f.read(min(8<<20,remaining))
                if not data:raise ValueError('Truncated donor data')
                out.write(data);remaining-=len(data)
    return quant,list(replacements),out_header

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',required=True,type=Path);p.add_argument('--target',required=True,type=Path);p.add_argument('--draft',required=True,type=Path)
    a=p.parse_args();root,data=state(a.state);target=a.target.expanduser().resolve();donor=a.draft.expanduser().resolve()
    verify_runtime(root,data)
    outside(root,target);outside(root,donor.parent)
    if not target.is_dir() or not donor.is_file():p.error('Target directory and donor file must exist')
    pins=json.loads((REPO/'runtime/model-metadata-hashes.json').read_text())
    for name,digest in pins.items():
        if sha(target/name)!=digest:raise ValueError(f'Target metadata differs from qualified checkpoint: {name}')
    config=json.loads((target/'config.json').read_text());validate_config(config)
    index=json.loads((target/'model.safetensors.index.json').read_text())
    for name in index['weight_map'].values():
        if Path(name).name!=name or not (target/name).is_file():p.error('Unsafe or missing target shard')
    if any('mtp' in k.lower() for k in index['weight_map']):p.error('This recipe expects a target without an embedded draft')
    if (root/'models').exists() or (root/'base').exists() or (root/'draft.safetensors').exists():p.error('Model view already exists; use a fresh state directory')
    quant,norms,h=adapt(donor,root/'draft.safetensors')
    view=root/'models'/target.name;view.mkdir(parents=True)
    before={}
    for f in target.iterdir():
        if f.is_file():
            st=f.stat();before[f.name]={'size':st.st_size,'mtime_ns':st.st_mtime_ns,'inode':st.st_ino}
            if f.name not in ('config.json','model.safetensors.index.json'):(view/f.name).symlink_to(f.resolve())
    if 'draft.safetensors' in before:raise ValueError('Target draft filename collision')
    collision=set(h)&set(index['weight_map'])
    if collision:raise ValueError('Donor would replace existing target tensors')
    (view/'draft.safetensors').symlink_to(root/'draft.safetensors')
    index['weight_map'].update({k:'draft.safetensors' for k in h})
    for key in ['quantization','quantization_config']:
        if key in config:config[key].update(quant)
    write_json(view/'config.json',config);write_json(view/'model.safetensors.index.json',index)
    base=root/'base';base.mkdir()
    settings={'enable_thinking':False,'preserve_thinking':False,'mtp_enabled':True,'mtp_num_draft_tokens':3,'dflash_enabled':False,'vlm_mtp_enabled':False,'qwen4_ple_ssd_offload':True,'max_context_window':393216,'max_tokens':16384,'temperature':0.7,'top_p':0.8,'top_k':20,'presence_penalty':1.5,'is_default':True,'is_pinned':True}
    write_json(base/'model_settings.json',{'version':1,'models':{target.name:settings}})
    for name,st in before.items():
        now=(target/name).stat()
        if [now.st_size,now.st_mtime_ns,now.st_ino]!=list(st.values()):raise RuntimeError('Target changed during setup')
    receipt={'target':str(target),'model':target.name,'source_file_metadata':before,'source_paths':{name:str((target/name).resolve()) for name in before},'donor_sha256':sha(donor),'adapted_draft_sha256':sha(root/'draft.safetensors'),'recentered_norms':norms}
    receipt['prepared_snapshot']=model_snapshot(root,receipt)
    write_json(root/'model-receipt.json',receipt)
    print('Prepared immutable target view and separate draft. Run numerical checks before serving.')
if __name__=='__main__':main()
