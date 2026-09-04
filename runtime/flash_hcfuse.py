"""Prefill-only concatenation of equal-quantization HC projections; originals retained."""
import copy
import mlx.core as mx
def fused(module):
 cached=getattr(module,'_flash_prefill_hc_weight',None)
 if cached is not None:return cached
 down=module.input_mix_weight_down;injection=module.block_inject_weight
 if not all(getattr(down,k,None)==getattr(injection,k,None) for k in ['bits','group_size','mode']):return None
 if getattr(down,'bits',None)!=6:return None
 clone=copy.copy(down)
 rows=down.weight.shape[0]+injection.weight.shape[0];padding=(-rows)%64
 arrays=[]
 for key in ['weight','scales','biases']:
  a=getattr(down,key);b=getattr(injection,key)
  value=mx.concatenate([a,b,mx.zeros((padding,*a.shape[1:]),dtype=a.dtype)],axis=0)
  setattr(clone,key,value);arrays.append(value)
 mx.eval(*arrays)
 module._flash_prefill_hc_weight=clone
 return clone
