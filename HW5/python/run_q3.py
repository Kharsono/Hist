import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

##########################

max_iters = 40
# pick a batch size, learning rate
batch_size = 48
learning_rate = 1e-2
hidden_size = 64

##########################

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
##########################

# Train layers
inputlayers = train_x.shape[1]
outputlayers = train_y.shape[1]
initialize_weights(inputlayers, hidden_size, params, "layer1")
initialize_weights(hidden_size, outputlayers, params, "output")

train_accL = []; train_lossL = []; valid_accL = []; valid_lossL = []

# # Q3.3 --> Print the recent initialized weights
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import ImageGrid

# # visualize weights here
# ##########################

# W = params['Wlayer1']
# assert W.shape == (32 * 32, 64)
# fig = plt.figure()
# grid = ImageGrid(fig, 111, nrows_ncols = (8, 8))
# for i in range(64):
#     grid[i].imshow( W[:,i].reshape(32, 32) )
# plt.show()

# ##########################

##########################

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        
        # training loop can be exactly the same as q2!
        ##########################
        
        # forward
        sig = forward(xb, params, "layer1", sigmoid)
        pred = forward(sig, params, "output", softmax)
        
        # loss
        # be sure to add loss and accuracy to epoch totals 
        loss, acc = compute_loss_and_acc(yb, pred)
        total_loss += loss
        total_acc += acc

        # backward
        diff = pred - yb
        diff2 = backwards(diff, params, "output", linear_deriv)
        diff2 = backwards(diff2, params, "layer1", sigmoid_deriv)
        
        # apply gradient --> learning rate
        params["Wlayer1"] = params["Wlayer1"] - params["grad_Wlayer1"] * learning_rate
        params["Woutput"] = params["Woutput"] - params["grad_Woutput"] * learning_rate
        params["blayer1"] = params["blayer1"] - params["grad_blayer1"] * learning_rate
        params["boutput"] = params["boutput"] - params["grad_boutput"] * learning_rate

    avg_acc = total_acc / batch_num
    avg_loss = total_loss / batch_num
    
    # forward
    sig = forward(valid_x, params, "layer1", sigmoid)
    pred = forward(sig, params, "output", softmax)
    # loss
    valid_loss, valid_acc = compute_loss_and_acc(valid_y, pred)
    valid_loss /= valid_x.shape[0]
    
    train_accL.append(avg_acc)
    train_lossL.append(avg_loss)
    valid_accL.append(valid_acc)
    valid_lossL.append(valid_loss)
    
    ##########################

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

x = np.arange(max_iters)

plt.plot(x, train_accL, linewidth = 3, label = "Training Accuracy")
plt.plot(x, valid_accL, linewidth = 3, label = "Validation Accuracy")
plt.legend()
plt.show()

plt.plot(x, train_lossL, linewidth = 3, label = "Training Loss")
plt.plot(x, valid_lossL, linewidth = 3, label = "Validation Loss")
plt.legend()
plt.show()

# run on validation set and report accuracy! should be above 75%
valid_acc = None

##########################

sig = forward(valid_x, params, "layer1", sigmoid)
pred = forward(sig, params, "output", softmax)
# loss
_, valid_acc = compute_loss_and_acc(valid_y, pred)

##########################

print('Validation accuracy: ',valid_acc)
if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# visualize weights here
##########################

W = params['Wlayer1']
assert W.shape == (32 * 32, 64)
fig = plt.figure()
grid = ImageGrid(fig, 111, nrows_ncols = (8, 8))
for i in range(64):
    grid[i].imshow( W[:,i].reshape(32, 32) )
plt.show()

##########################


# Q3.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# compute comfusion matrix here
##########################

valid_pred_y = np.argmax(pred, axis=1)
for i in range(valid_pred_y.shape[0]):
    pred_max = valid_pred_y[i]
    y_max = np.argmax(valid_y[i])
    confusion_matrix[pred_max][y_max] += 1
    
##########################

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()