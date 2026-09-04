"""Lossless temporary widening of packed 6-bit integers; scales/biases unchanged."""
import mlx.core as mx
_kernel=mx.fast.metal_kernel(name='flash_six_to_eight',input_names=['w'],output_names=['out'],source=r'''
uint i=thread_position_in_grid.x;
if(i>=GROUPS) return;
uint a=w[3*i],b=w[3*i+1],c=w[3*i+2];
uint p[4]={a&0xffffffu,((a>>24)|(b<<8))&0xffffffu,((b>>16)|(c<<16))&0xffffffu,c>>8};
for(uint j=0;j<4;j++){
 uint v=p[j];
 out[4*i+j]=(v&63u)|(((v>>6)&63u)<<8)|(((v>>12)&63u)<<16)|(((v>>18)&63u)<<24);
}
''')
def repack(w):
 assert w.dtype==mx.uint32 and w.shape[-1]%3==0
 return _kernel(inputs=[w],template=[("GROUPS",w.size//3)],grid=(w.size//3,1,1),threadgroup=(256,1,1),output_shapes=[(*w.shape[:-1],w.shape[-1]*4//3)],output_dtypes=[mx.uint32])[0]
