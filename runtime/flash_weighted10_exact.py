import mlx.core as mx
_kernel=mx.fast.metal_kernel(name='flash_weighted10_exact',input_names=['x','inv','scores'],output_names=['out'],source=r'''
uint i=thread_position_in_grid.x;
if(i>=ROWS*DIM) return;
uint row=i/DIM,col=i%DIM;
T sums[LANES];
for(uint lane=0;lane<LANES;lane++){
 T subtotal=T(0);
 for(uint j=lane;j<10;j+=LANES){
  uint k=row*10+j;
  T product=T(x[inv[k]*DIM+col]*T(scores[k]));
  subtotal=T(product+subtotal);
 }
 sums[lane]=subtotal;
}
T total=sums[0];
for(uint lane=1;lane<LANES;lane++)total=T(sums[lane]+total);
out[i]=total;
''')
def weighted_sum(x,inv,scores,lanes=1):
 assert scores.shape[-1]==10
 d=x.shape[-1];rows=scores.size//10
 return _kernel(inputs=[x,inv,scores],template=[('T',x.dtype),('ROWS',rows),('DIM',d),('LANES',lanes)],grid=(rows*d,1,1),threadgroup=(256,1,1),output_shapes=[(*scores.shape[:-1],d)],output_dtypes=[x.dtype])[0]
