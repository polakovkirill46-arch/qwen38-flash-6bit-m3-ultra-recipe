#!/usr/bin/env python3
"""GPU checks. Run ONLY with other inference stopped; does not load the model."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from common import environment,state,write_json,artifact_fingerprint

def checks():
    import mlx.core as mx
    from omlx.custom_kernels.decode_fast import _ext as decode
    from omlx.custom_kernels.glm_moe_dsa import _ext as glm
    from omlx.custom_kernels.qwen35_prefill import _ext as prefill
    from flash_hcmix import mix_mean
    from flash_hcnorm import norm
    from flash_weighted10_exactfast import weighted_sum
    if mx.__version__!='0.32.2':raise RuntimeError('Wrong MLX runtime')
    mx.random.seed(42);rows=[]
    for t in [3256,8192]:
        x=mx.random.normal((1,t,10240)).astype(mx.bfloat16);w=(mx.random.normal((10240,))*.1).astype(mx.bfloat16)
        y=x.astype(mx.float32).reshape(1,t,4,2560);y=y*mx.rsqrt(mx.mean(mx.square(y),axis=-1,keepdims=True)+1e-6)
        ref=(y*(1+w.astype(mx.float32).reshape(4,2560))).reshape(x.shape).astype(x.dtype);out=norm(x,w,1e-6,256)
        error=mx.sqrt(mx.mean((ref.astype(mx.float32)-out.astype(mx.float32))**2)/mx.mean(ref.astype(mx.float32)**2)).item()
        if error>=1e-4:raise RuntimeError(f'Normalization error: {error}')
        up=mx.random.normal(x.shape).astype(mx.bfloat16);ref=(mx.sigmoid(up).reshape(1,t,4,2560)*x.reshape(1,t,4,2560)).mean(-2);out=mix_mean(up,x)
        if not mx.array_equal(ref,out).item():raise RuntimeError('HC mix parity failed')
        rows.append({'tokens':t,'normalization_relative_rms':error,'mix_bit_exact':True})
        del x,w,y,ref,out,up;mx.synchronize();mx.clear_cache()
    # The reducer maps sorted expert outputs back into token/top-k order.
    t=3256;d=2560;order=mx.argsort(mx.random.uniform(shape=(t*10,))).astype(mx.uint32);scores=mx.softmax(mx.random.normal((1,t,10)).astype(mx.bfloat16),axis=-1)
    x=mx.random.normal((t*10,1,d)).astype(mx.bfloat16)
    ref=(x[order].reshape(1,t,10,d)*scores[...,None]).sum(-2);out=weighted_sum(x,order,scores.astype(mx.float32))
    if not mx.array_equal(ref,out).item():raise RuntimeError('Expert reduction parity failed')
    return {'mlx':mx.__version__,'native_imports':['decode_fast','glm_moe_dsa','qwen35_prefill'],'cells':rows,'weighted10_bit_exact':True}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',required=True,type=Path);p.add_argument('--worker',action='store_true');a=p.parse_args();root,data=state(a.state)
    if a.worker:
        before=artifact_fingerprint(root,data)
        result=checks()
        if artifact_fingerprint(root,data)!=before:raise RuntimeError('Artifacts changed during numerical checks')
        result.update(success=True,artifact_fingerprint=before)
        write_json(root/'numerical-checks.json',result);print(json.dumps(result,indent=2))
    else:
        if (root/'numerical-checks.json').exists():p.error('Check receipt already exists; retain it or move it aside explicitly')
        subprocess.run([data['python'],str(Path(__file__).resolve()),'--state',str(root),'--worker'],env=environment(root,data),cwd=root,check=True)
if __name__=='__main__':main()
