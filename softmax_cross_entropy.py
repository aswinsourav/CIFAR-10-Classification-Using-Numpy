class Softmax_Loss:
    def __init__(self,logits,labels):
        self.logits=logits
        self.labels=labels
    
    def calc_softmax(self):
        exp_val=np.exp(self.logits)
        self.softmax_val=exp_val/np.sum(exp_val,axis=1)[:,None]
        return self.softmax_val
    
    def cross_entropy(self):
        loss=-np.sum(self.labels*np.log(self.softmax_val))
        return loss