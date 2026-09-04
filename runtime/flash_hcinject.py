import mlx.core as mx
_kernel=mx.fast.metal_kernel(name='flash_hc_inject_residual',input_names=['hyper','branch','weights'],output_names=['out'],source=r'''
uint i=thread_position_in_grid.x;
if(i>=ROWS*10240)return;
uint row=i/10240,col=i%10240;
T product=T(branch[row*2560+col%2560]*weights[row*4+col/2560]);
out[i]=T(hyper[i]+product);
''')
def inject(hyper,branch,weights):
 assert hyper.shape[-1]==10240 and branch.shape[-1]==2560 and weights.shape[-1]==4
 assert hyper.dtype==branch.dtype==weights.dtype
 return _kernel(inputs=[hyper,branch,weights],template=[('T',hyper.dtype),('ROWS',hyper.size//10240)],grid=(hyper.size,1,1),threadgroup=(256,1,1),output_shapes=[hyper.shape],output_dtypes=[hyper.dtype])[0]
