# Local prefill implementation of established grouped RMS normalization.
# Related HC fusion: MTPLX PR391. See PROVENANCE.md.
"""FP32 grouped RMS normalization with unchanged residual-gamma weights."""
import mlx.core as mx
_kernel=mx.fast.metal_kernel(name='flash_hc_group_rms',input_names=['x','weight'],output_names=['out'],source=r'''
uint row=threadgroup_position_in_grid.x;
uint tid=thread_position_in_threadgroup.x;
float sum=0;
for(uint c=tid;c<2560;c+=THREADS){float v=float(x[row*2560+c]);sum+=v*v;}
sum=simd_sum(sum);
threadgroup float partial[32];
if((tid%32)==0)partial[(tid/32)]=sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
float v=tid<THREADS/32?partial[tid]:0;
if((tid/32)==0){v=simd_sum(v);if(tid==0)partial[0]=metal::rsqrt(v/2560.0f+1e-6f);}
threadgroup_barrier(mem_flags::mem_threadgroup);
float inv=partial[0];
for(uint c=tid;c<2560;c+=THREADS){
 float norm=float(x[row*2560+c])*inv;
 out[row*2560+c]=T(norm*(1.0f+float(weight[(row%4)*2560+c])));
}
''')
def norm(x,w,eps,threads=256):
 assert x.shape[-1]==10240 and w.size==10240 and eps==1e-6
 return _kernel(inputs=[x,w],template=[('T',x.dtype),('THREADS',threads)],grid=(x.size//2560*threads,1,1),threadgroup=(threads,1,1),output_shapes=[x.shape],output_dtypes=[x.dtype])[0]
