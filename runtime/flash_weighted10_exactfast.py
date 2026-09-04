# Local top-10 specialization following MLX BF16 reduction order (Apple Inc., MIT).
# See licenses/MLX-MIT.txt and THIRD_PARTY_NOTICES.md.
"""Fuse scatter/product/reduction while retaining MLX's 8-lane BF16 accumulation."""
import mlx.core as mx
_kernel=mx.fast.metal_kernel(name='flash_weighted10_exact_parallel',input_names=['x','inv','scores'],output_names=['out'],source=r'''
uint row=threadgroup_position_in_grid.y;
uint lx=thread_position_in_threadgroup.x,ly=thread_position_in_threadgroup.y;
uint col=threadgroup_position_in_grid.x*128+lx*4;
threadgroup T partial[1024];
T sums[4]={T(0),T(0),T(0),T(0)};
for(uint j=ly;j<10;j+=8){
 uint k=row*10+j;
 uint offset=inv[k]*DIM+col;
 T score=T(scores[k]);
 for(uint c=0;c<4;c++){
  T product=T(x[offset+c]*score);
  sums[c]=T(product+sums[c]);
 }
}
for(uint c=0;c<4;c++)partial[(ly*32+lx)*4+c]=sums[c];
threadgroup_barrier(mem_flags::mem_threadgroup);
if(ly==0){
 for(uint c=0;c<4;c++){
  T total=partial[lx*4+c];
  for(uint lane=1;lane<8;lane++)total=T(partial[(lane*32+lx)*4+c]+total);
  out[row*DIM+col+c]=total;
 }
}
''')
def weighted_sum(x,inv,scores):
 assert scores.shape[-1]==10 and x.shape[-1]%128==0
 d=x.shape[-1];rows=scores.size//10
 return _kernel(inputs=[x,inv,scores],template=[('T',x.dtype),('DIM',d)],grid=(d//4,rows*8,1),threadgroup=(32,8,1),output_shapes=[(*scores.shape[:-1],d)],output_dtypes=[x.dtype])[0]
