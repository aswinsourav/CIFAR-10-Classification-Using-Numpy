channel=3
filter1=16
filter2=20
filter3=25
labels=10
batch_size=125
dropout_p=0.5

train_data=X_train[0:batch_size,:,:,:]
train_labels=Y_train[0:batch_size]

cnn1=CNN(train_data,5,1,filter1,channel,2,0.8)
cnn1.weight_init()
cnn1_out=cnn1.convolve(cnn1.input_image)
print("cnn1",cnn1_out.shape)

pool1=Pool(cnn1_out,2,2)
pool1_out=pool1.max_pool()
print("Pool1",pool1_out.shape)

cnn2=CNN(pool1_out,5,1,filter2,filter1,2,0.7)
cnn2.weight_init()
cnn2_out=cnn2.convolve(cnn2.input_image)
print("cnn2",cnn2_out.shape)

pool2=Pool(cnn2_out,2,2)
pool2_out=pool2.max_pool()
print("Pool2",pool2_out.shape)

cnn3=CNN(pool2_out,5,1,filter3,filter2,2,dropout_p)
cnn3.weight_init()
cnn3_out=cnn3.convolve(cnn3.input_image)
print("cnn3",cnn3_out.shape)

pool3=Pool(cnn3_out,2,2)
pool3_out=pool3.max_pool()
print("Pool3",pool3_out.shape)

pool_elongate=pool3_out.reshape(pool3_out.shape[0],pool3_out.shape[1]*pool3_out.shape[2]*pool3_out.shape[3])

fc1=FC(pool_elongate,labels)
fc1_weights=fc1.weight_init()

logits=fc1.FC_out(fc1.pool_input)
shift_logits=logits-np.max(logits,axis=1,keepdims=True)

softmax=Softmax_Loss(shift_logits,train_labels)
pred_prob=softmax.calc_softmax()
loss=softmax.cross_entropy()
print("Loss",loss)


def forward(input_image,input_label,dropout=True):
    cnn1_out=cnn1.convolve(input_image,dropout)
    pool1=Pool(cnn1_out,2,2)
    pool1_out=pool1.max_pool()
    
    cnn2_out=cnn2.convolve(pool1_out,dropout)
    pool2=Pool(cnn2_out,2,2)
    pool2_out=pool2.max_pool()
    
    cnn3_out=cnn3.convolve(pool2_out,dropout)
    pool3=Pool(cnn3_out,2,2)
    pool3_out=pool3.max_pool()
    
    pool_elongate=pool3_out.reshape(pool3_out.shape[0],pool3_out.shape[1]*pool3_out.shape[2]*pool3_out.shape[3])
    
    logits=fc1.FC_out(pool_elongate)
    shift_logits=logits-np.max(logits,axis=1,keepdims=True)
    softmax=Softmax_Loss(shift_logits,input_label)
    pred_prob=softmax.calc_softmax()
    loss=softmax.cross_entropy()
    accuracy=np.sum(np.argmax(pred_prob,axis=1)==np.argmax(input_label,axis=1))/input_label.shape[0]
    return pool_elongate,pool3_out,pred_prob,loss,accuracy
    
    
def backward(pool_elongate,pool3_out,pred_prob,input_label,lr):
    softmax_gradient=pred_prob-input_label
    delta_fc_weights=pool_elongate.T.dot(softmax_gradient)
    delta_fc_bias=np.sum(softmax_gradient)
    delta_pool3=np.einsum('ij,jk->ik',softmax_gradient,fc1.softmax_weights.T).reshape(pool3_out.shape)
    pool3_out[pool3_out!=0]=1
    delta_relu3=relu_gradient(cnn3.cnn_output_relu,pool3_out*delta_pool3)
    delta_cnn3_weights,delta_cnn3_bias=CNN_gradient(delta_relu3,cnn3)

    delta_pool2=pool_gradient(delta_relu3,cnn3)
    delta_relu2=relu_gradient(cnn2.cnn_output_relu,delta_pool2)
    delta_cnn2_weights,delta_cnn2_bias=CNN_gradient(delta_relu2,cnn2)

    delta_pool1=pool_gradient(delta_relu2,cnn2)
    delta_relu1=relu_gradient(cnn1.cnn_output_relu,delta_pool1)
    #print(delta_relu1.shape)
    delta_cnn1_weights,delta_cnn1_bias=CNN_gradient(delta_relu1,cnn1)
    return delta_fc_weights,delta_fc_bias,delta_cnn1_weights,delta_cnn1_bias,delta_cnn2_weights,delta_cnn2_bias,delta_cnn3_weights,delta_cnn3_bias
    
vel_fc_weights={'learning_rate':1e-3,'momentum':0.9,'velocity':np.zeros_like(fc1.softmax_weights)}
vel_fc_bias={'learning_rate':1e-3,'momentum':0.9,'velocity':np.zeros_like(fc1.bias)}

vel_cnn1_weights={'learning_rate':1e-2,'momentum':0.9,'velocity':np.zeros_like(cnn1.cnn_weights)}
vel_cnn1_bias={'learning_rate':1e-2,'momentum':0.9,'velocity':np.zeros_like(cnn1.bias)}

vel_cnn2_weights={'learning_rate':1e-2,'momentum':0.9,'velocity':np.zeros_like(cnn2.cnn_weights)}
vel_cnn2_bias={'learning_rate':1e-2,'momentum':0.9,'velocity':np.zeros_like(cnn2.bias)}

vel_cnn3_weights={'learning_rate':1e-2,'momentum':0.9,'velocity':np.zeros_like(cnn3.cnn_weights)}
vel_cnn3_bias={'learning_rate':1e-2,'momentum':0.9,'velocity':np.zeros_like(cnn3.bias)}
lr=0.00001

for i in range(6001):
    #offset=0
    offset = (i * batch_size) % (Y_train.shape[0] - batch_size)
    batch_data = X_train[offset:(offset + batch_size), :]
    batch_labels = Y_train[offset:(offset + batch_size), :]
    
    pool_elongate,pool3_out,pred_prob,train_loss,train_accuracy=forward(batch_data,batch_labels)
    delta_fc_weights,delta_fc_bias,delta_cnn1_weights,delta_cnn1_bias,delta_cnn2_weights,delta_cnn2_bias,delta_cnn3_weights,delta_cnn3_bias=backward(pool_elongate,pool3_out,pred_prob,batch_labels,0.00001)
    
    fc1.softmax_weights,vel_fc_weights=sgd_momentum(fc1.softmax_weights,delta_fc_weights,vel_fc_weights)
    fc1.bias,vel_fc_bias=sgd_momentum(fc1.bias,delta_fc_bias,vel_fc_bias)

    cnn1.cnn_weights,vel_cnn1_weights=sgd_momentum(cnn1.cnn_weights,delta_cnn1_weights,vel_cnn1_weights)
    cnn1.bias,vel_cnn1_bias=sgd_momentum(cnn1.bias,delta_cnn1_bias,vel_cnn1_bias)
    
    
    cnn2.cnn_weights,vel_cnn2_weights=sgd_momentum(cnn2.cnn_weights,delta_cnn2_weights,vel_cnn2_weights)
    cnn2.bias,vel_cnn2_bias=sgd_momentum(cnn2.bias,delta_cnn2_bias,vel_cnn2_bias)
    
   
    cnn3.cnn_weights,vel_cnn3_weights=sgd_momentum(cnn3.cnn_weights,delta_cnn3_weights,vel_cnn3_weights)
    cnn3.bias,vel_cnn3_bias=sgd_momentum(cnn3.bias,delta_cnn3_bias,vel_cnn3_bias)

    
    
    if(i%500==0):
        _,_,_,test_loss,test_accuracy=forward(batch_data,batch_labels,False)
        print("Step: ",i,"Train Loss: ",test_loss," Train Accuracy: ",test_accuracy)    
        
    if(i%1000==0):
        _,_,_,test_loss,test_accuracy=forward(X_test,Y_test,False)
        print("Step: ",i,"Train Loss: ",train_loss," Test Loss: ",test_loss," Test Accuracy:",test_accuracy)                