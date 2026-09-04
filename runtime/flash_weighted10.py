import mlx.core as mx
_kernel=mx.fast.metal_kernel(name='flash_weighted10',input_names=['x','inv','scores'],output_names=['out'],source=r'''
uint i=thread_position_in_grid.x;
if(i>=ROWS*DIM) return;
uint row=i/DIM,col=i%DIM;
float sum=0;
for(uint j=0;j<10;j++){
 uint k=row*10+j;
 // Preserve the original BF16 product rounding before the FP32 reduction.
 T product=T(x[inv[k]*DIM+col]*T(scores[k]));
 sum+=float(product);
}
out[i]=T(sum);
''')
def weighted_sum(x,inv,scores):
 assert scores.shape[-1]==10
 d=x.shape[-1];rows=scores.size//10
 return _kernel(inputs=[x,inv,scores],template=[('T',x.dtype),('ROWS',rows),('DIM',d)],grid=(rows*d,1,1),threadgroup=(256,1,1),output_shapes=[(*scores.shape[:-1],d)],output_dtypes=[x.dtype])[0]
