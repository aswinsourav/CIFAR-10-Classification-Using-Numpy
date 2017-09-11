def relu_gradient(cnn_relu,delta_pool):
    de_relu=np.zeros(cnn_relu.shape)
    i=j=0
    while(i<cnn_relu.shape[2]):
        j=0
        while(j<cnn_relu.shape[3]):
            a = cnn_relu[:,:,i:i+2,j:j+2]
            de_relu[:,:,i:i+2,j:j+2]=( a == np.nanmax(a,axis=(2,3))[:,:,None,None]).astype(int)*delta_pool[:,:,i//2,j//2][:,:,None,None]
            j+=2
        i+=2
    cnn_relu[cnn_relu!=0]=1
    return de_relu*cnn_relu
    
def CNN_gradient(de_relu,cnn):
    delta_cnn=np.zeros((cnn.cnn_weights.shape))
    k=j=0
    while(k<delta_cnn.shape[3]):
        j=0
        while(j<delta_cnn.shape[2]):
            stride_delta=np.einsum('icjk,iw->wcjk',cnn.input_image[:,:,k:k+cnn.stride,j:j+cnn.stride],de_relu[:,:,k,j])
            delta_cnn+=stride_delta
            j+=1
        k+=1
    return delta_cnn,np.sum(de_relu,axis=(0,2,3))
    
def pool_gradient(delta_relu,cnn):
    delta_pool=np.zeros((cnn.input_image.shape))
    k=j=0
    while(k<delta_relu.shape[2]-cnn.pad*2):
        j=0
        while(j<delta_relu.shape[3]-cnn.pad*2):
            stride_delta=np.einsum('fcjk,if->icjk',cnn.cnn_weights,delta_relu[:,:,k+cnn.pad,j+cnn.pad])
            delta_pool[:,:,k:k+5,j:j+5]+=stride_delta
            j+=1
        k+=1
    cnn.input_image[cnn.input_image!=0]=1
    return delta_pool*cnn.input_image
    
def sgd_momentum(w, dw, config=None):
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))
    next_w = None
    v = config['momentum']*v - config['learning_rate']*dw
    next_w = w + v
    config['velocity'] = v
    return next_w, config    