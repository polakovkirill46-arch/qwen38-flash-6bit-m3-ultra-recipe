# Adapted from oMLX PR3298 by onthehub97; backend from PR2853.
# See THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.
"""Staged Qwen4 GDN-only ANE adapter, based on upstream PR3298.
Original target tensors remain unchanged; private ANE uses approximate working copies.
"""
import os,logging,importlib,copy
import mlx.core as mx
from types import SimpleNamespace
from omlx.patches import qwen35_ane_prefill as p
log=logging.getLogger(__name__)
def enable(model):
    if getattr(model,'_flash_ane_enabled',False):return
    language=getattr(model,'language_model',model)
    body=getattr(language,'model',language)
    layers=getattr(body,'layers',[])
    gdns=[x.linear_attn for x in layers if getattr(x,'linear_attn',None) is not None]
    assert len(gdns)==36, len(gdns)
    original_gdns=gdns
    cpu_fraction=float(os.environ.get('FLASH_ANE_CPU','0'))
    threads=int(os.environ.get('FLASH_ANE_THREADS','4'))
    if os.environ.get('FLASH_ANE_TUNED','0')=='1':
        import json,pathlib
        best=json.loads((pathlib.Path(__file__).with_name('ane-cpu-fine-best.json')).read_text())
        cpu_fraction=best['fraction'];threads=best['threads']
    if cpu_fraction:
        class GdnProxy:pass
        gdns=[]
        for original in original_gdns:
            clone=GdnProxy()
            for name in ['in_proj_qkv','in_proj_z','in_proj_b','in_proj_a']:
                original_linear=getattr(original,name)
                linear=copy.copy(original_linear)
                linear.scales=original_linear.scales.astype(mx.float16)
                linear.biases=original_linear.biases.astype(mx.float16)
                mx.eval(linear.scales,linear.biases)
                assert original_linear.scales.dtype==mx.bfloat16
                setattr(clone,name,linear)
            gdns.append(clone)
        model._flash_ane_proxies=gdns
    seq=int(os.environ.get('FLASH_ANE_SEQ','3328'))
    config=p._AnePrefillConfig(sequence_length=seq, fraction=.4,variant=8,dual_ane=True,cpu_threads=threads,cpu_shared_resource=False,tail_padding_min_tokens=2048)
    # Restrict preparation to serving decoder GDNs; never include the MTP head.
    proxy=SimpleNamespace(modules=lambda:gdns)
    result=p._enable_fused_gdn_banks(proxy,config,fraction=.4,max_layers=36,cpu_fraction=cpu_fraction)
    assert result is not None and result[0]==36, result
    # Materialize load-thread working copies before the engine thread consumes them.
    mx.eval(*[v for g in gdns for v in vars(g._omlx_ane_gdn_state).values() if isinstance(v,mx.array)])
    mapping={id(original.in_proj_qkv):g for original,g in zip(original_gdns,gdns)}
    p._GDN_MODULES.update(mapping)
    def wrap(original):
        def dispatch(linears,x,target_verify=False):
            g=mapping.get(id(linears[0])) if linears else None
            if g is not None:
                y=p._gdn_backend(g,x.astype(g._omlx_ane_gdn_state.scales.dtype),target_verify)
                if y is not None:return tuple(v.astype(x.dtype) for v in y)
            return original(linears,x,target_verify)
        return dispatch
    p._install_dispatch()
    for name in ['mlx_vlm.models.qwen3_5.language','mlx_vlm.models.qwen4_exp.language']:
        module=importlib.import_module(name)
        module._target_verify_linears=wrap(module._target_verify_linears)
    model._flash_ane_enabled=True
    log.info('FLASH ANE: %s serving GDN layers, fixed sequence %s, tail padding >=2048, MTP excluded',result,seq)
