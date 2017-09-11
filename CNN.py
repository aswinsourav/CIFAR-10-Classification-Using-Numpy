class CNN:
    def __init__(self,input_image,filter_size,stride,filter_depth,channels,pad,dropout_p):
        self.input_image=input_image
        self.filter_size=filter_size
        self.stride=stride
        self.filter_depth=filter_depth
        self.channels=channels
        self.pad=pad
        self.dropout_p=dropout_p
    
    def weight_init(self):
        fan_in=self.filter_size*self.filter_size*self.channels
        self.cnn_weights=np.random.randn(self.filter_depth,self.channels,self.filter_size,self.filter_size)/np.sqrt(fan_in/2)
        self.bias=np.zeros(self.filter_depth)
        #print(self.cnn_weights.shape)
    
    
    def convolve(self,input_image,dropout=True):
        output_shape=int(((input_image.shape[2]-self.filter_size)/self.stride)+1)
        #print(output_shape)
        self.cnn_output=np.zeros((input_image.shape[0],self.filter_depth,output_shape,output_shape))
        #print(cnn_output.shape)
        i=j=k=r=c=0
        while(k<output_shape):
            j=c=0
            while(j<output_shape):
                self.cnn_output[:,:,r,c]=np.einsum('icjk,fcjk->if',input_image[:,:,k:k+self.filter_size,j:j+self.filter_size],self.cnn_weights)+self.bias
                j+=self.stride
                c+=1
            k+=self.stride
            r+=1
        npad=((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad))
        self.cnn_output=np.pad(self.cnn_output, pad_width=npad, mode='constant', constant_values=0)    
        self.cnn_output_relu=relu_op(self.cnn_output)
        if(dropout):
            dr=(np.random.rand(*self.cnn_output_relu.shape)<self.dropout_p)/self.dropout_p
        else:
            dr=1
            
        return self.cnn_output_relu*dr