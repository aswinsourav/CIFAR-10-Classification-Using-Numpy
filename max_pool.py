class Pool:
    
        def max_pool(self):
            output_shape=int(((self.input_cnn.shape[2]-self.filter_size)/self.stride)+1)
            pool_out=np.zeros((self.input_cnn.shape[0],self.input_cnn.shape[1],output_shape,output_shape))
            i=j=r=c=0
            while(i<self.input_cnn.shape[2]):
                j=c=0
                while(j<self.input_cnn.shape[3]):   
                    pool_out[:,:,r,c]=np.max(self.input_cnn[:,:,i:i+2,j:j+2],axis=(2,3))
                    j+=self.stride
                    c+=1
                i+=self.stride
                r+=1
            return pool_out
            
            
        def __init__(self,input_cnn,filter_size,stride):
            self.input_cnn=input_cnn
            self.filter_size=filter_size
            self.stride=stride