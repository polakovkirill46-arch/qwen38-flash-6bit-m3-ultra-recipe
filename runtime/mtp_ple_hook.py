"""Deferred gather adapter; metadata only at graph-build time. No tensor reads."""
import importlib.metadata
import _mtp_ple_native as native

# The serving interpreter, extension and MLX build are pinned for its lifetime.
# Check once before exposing gather; repeated distribution metadata discovery
# can touch the filesystem on every speculative cycle. A failed import/check
# leaves this module unavailable, so no unchecked native operation can run.
_runtime_mlx_version = importlib.metadata.version('mlx')
if _runtime_mlx_version != native.BUILT_AGAINST_MLX:
 raise RuntimeError('Deferred PLE MLX ABI mismatch')

def gather(owner,indices):
 import os
 if os.environ.get("FLASH_PLE_AUDIT", "0") == "1":
  from mtp_ple_audit import ple_call
  ple_call()
 bank=getattr(owner,'_flash_deferred_ple_bank',None)
 if bank is None:
  paths=[];offsets=[];counts=[]
  for i in range(len(owner.shard_sizes)):
   key,sc,bi,bits,group=owner._shard_specs[i]
   reader=owner._tensor_readers[key];entry=reader._header[key]
   assert sc is None and bi is None and entry['dtype']=='BF16'
   assert entry['shape']==[owner.shard_sizes[i],owner.dims]
   paths.append(str(reader.path));offsets.append(reader._data_start+entry['data_offsets'][0]);counts.append(owner.shard_sizes[i])
  bank=native.Bank(paths,offsets,counts,owner.dims)
  owner._flash_deferred_ple_bank=bank
 owner.rows_read=indices.size
 # last_touched_shards is diagnostic only; do not sync IDs just to populate it.
 owner.last_touched_shards=()
 return native.gather(bank,indices)*owner.weight_scale
