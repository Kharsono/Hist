import numpy as np
import scipy.io
from nn import *
from collections import Counter

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
##########################

input_sz = train_x.shape[1]
initialize_weights(input_sz, batch_size, params, "layer1")
initialize_weights(batch_size, batch_size, params, "layer2")
initialize_weights(batch_size, batch_size, params, "layer3")
initialize_weights(batch_size, batch_size, params, "output")

##########################

# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        ##########################
        
         # forward
        sig = forward(xb, params, "layer1", relu)
        sig2 = forward(sig, params, "layer2", relu)
        sig3 = forward(sig2, params, "layer3", relu)
        out = forward(sig3, params, "output", sigmoid)
        
        # loss
        # be sure to add loss and accuracy to epoch totals 
        loss = np.sum((out - xb)**2) #CANT FIGRE OUT BROADCAST
        total_loss += loss
        
        # backward
        diff = (out - xb) * 2
        diff2 = backwards(diff, params, "output", sigmoid_deriv)
        diff3 = backwards(diff2, params, "layer3", relu_deriv)
        diff4 = backwards(diff3, params, "layer2", relu_deriv)
        diff5 = backwards(diff4, params, "layer1", relu_deriv)
        
        # apply gradient --> learning rate
        params["Wlayer1"] = params["Wlayer1"] - params["grad_Wlayer1"] * learning_rate
        params["Wlayer2"] = params["Wlayer2"] - params["grad_Wlayer2"] * learning_rate
        params["Wlayer3"] = params["Wlayer3"] - params["grad_Wlayer3"] * learning_rate
        params["Woutput"] = params["Woutput"] - params["grad_Woutput"] * learning_rate
        
        params["blayer1"] = params["blayer1"] - params["grad_blayer1"] * learning_rate
        params["blayer2"] = params["blayer2"] - params["grad_blayer2"] * learning_rate
        params["blayer3"] = params["blayer3"] - params["grad_blayer3"] * learning_rate
        params["boutput"] = params["boutput"] - params["grad_boutput"] * learning_rate
        
    total_loss = total_loss/train_x.shape[0]
    loss_plt.append(total_loss)
    
        ##########################

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9
        
plt.figure()
plt.plot(np.arange(max_iters), loss_plt)
plt.show()
        
# Q5.3.1
import matplotlib.pyplot as plt
# visualize some results
##########################
#PARTIAL CREDS FOR THIS -> cant run

i = np.array([0, 10,20, 50])
xb = valid_x[i]

sig = forward(xb, params, "layer1", relu)
sig2 = forward(sig, params, "layer2", relu)
sig3 = forward(sig2, params, "layer3", relu)
out = forward(sig3, params, "output", sigmoid)

for i in range(5):
    plt.subplot(2,1,1)
    plt.imshow(xb[i])
    plt.subplot(2,1,2)
    plt.imshow(out[i])
    plt.show()
    
##########################


# Q5.3.2
from skimage.measure import compare_psnr as psnr
# evaluate PSNR
##########################
#partial cred -> cant run

PSNR = 0
sig = forward(valid_x, params, "layer1", relu)
sig2 = forward(sig, params, "layer2", relu)
sig3 = forward(sig2, params, "layer3", relu)
out = forward(sig3, params, "output", sigmoid)

for i in range(valid_x.shape[0]):
    temp = psnr(valid_x[i], out[i], None)
    PSNR += temp
    
PSNR = PSNR / valid_x.shape[0]
print(PSNR) # Answer for write up 5.3.2


##########################
