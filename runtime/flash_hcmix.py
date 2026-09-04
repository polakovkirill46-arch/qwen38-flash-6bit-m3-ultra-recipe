# Follows MLX sigmoid and BF16 accumulation semantics (Apple Inc., MIT).
# See licenses/MLX-MIT.txt and PROVENANCE.md for related HC fusion work.
import mlx.core as mx
_kernel=mx.fast.metal_kernel(name='flash_hc_sigmoid_mix_mean',input_names=['up','normed'],output_names=['out'],source=r'''
uint i=thread_position_in_grid.x;
if(i>=ROWS*DIM)return;
uint row=i/DIM,col=i%DIM;
T total=T(0);
for(uint j=0;j<4;j++){
 uint offset=row*4*DIM+j*DIM+col;
 T raw=up[offset];
 auto y=1/(1+metal::exp(metal::abs(raw)));
 T gate=(raw<0)?y:1-y;
 T product=T(gate*normed[offset]);
 total=T(product+total);
}
out[i]=T(total*T(.25));
''')
def mix_mean(up,normed):
 assert up.shape==normed.shape and up.shape[-1]==10240
 d=2560;rows=up.size//10240
 return _kernel(inputs=[up,normed],template=[('T',up.dtype),('ROWS',rows),('DIM',d)],grid=(rows*d,1,1),threadgroup=(256,1,1),output_shapes=[(*up.shape[:-1],d)],output_dtypes=[up.dtype])[0]
