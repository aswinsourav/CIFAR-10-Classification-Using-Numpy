class FC:
    def __init__(self,pool_input,out_labels):
        self.pool_input=pool_input
        self.out_labels=out_labels
    
    def weight_init(self):
        fan_in=self.pool_input.shape[1]
        self.softmax_weights=np.random.randn(fan_in,self.out_labels)/np.sqrt(fan_in/2)
        self.bias=np.zeros(self.out_labels)
        return self.softmax_weights
        
    def FC_out(self,pool_input):
        return np.dot(pool_input,self.softmax_weights)+self.bias
        
        