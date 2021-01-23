import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None
    
    ##########################
    
    bound = np.sqrt(6 / (in_size + out_size))
    W = np.random.uniform( -bound, bound, (in_size, out_size)) #Weights
    b = np.zeros(out_size) #bias
    
    ##########################

    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = None

    ##########################
    res = 1 / (1 + np.exp(-x))
    ##########################
    
    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    ##########################
    
    y = np.dot(X, W) + b
    post_act = activation(y)
    
    ##########################

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = None

    ##########################
    
    max_x = np.max(x, axis=1)
    c = np.expand_dims(-max_x, axis=1)
    res = np.exp(x + c) / np.expand_dims( np.sum( np.exp(x + c), axis=1), axis=1)
    
    ##########################

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    ##########################
    
    loss = -np.sum(y * np.log(probs))
    
    y_pred = np.argmax(probs, axis=1)
    corr = 0
    for i in range(y.shape[0]):
        if y[i, y_pred[i]] == 1.0:
            corr += 1
    acc = corr / y.shape[0]
    
    ##########################

    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    # then compute the derivative W,b, and X
    ##########################
    
    # Initialize size
    grad_W = np.zeros(W.shape) # d x k
    grad_X = np.zeros(X.shape) # d x 1 for each
    grad_b = np.zeros(b.shape) # k x 1
    
    deltad = delta * activation_deriv(post_act)
    for i in range(X.shape[0]):
        # Refer to Q 1.5 for calcs
        dW = np.dot(np.expand_dims(X[i, :], axis=1), np.expand_dims(deltad[i, :], axis=0)) # d x k
        dX = np.dot(W, deltad[i,:]) # d x 1
        db = deltad[i, :] # k x 1
        grad_W += dW
        grad_X[i,:] = dX
        grad_b += db
        
    ##########################

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    
    ##########################
    
    ind = np.arange(x.shape[0])
    np.random.shuffle(ind)
    for i in range(0, x.shape[0], batch_size):
        x_b = x[ ind[ i : i + batch_size ] ]
        #print(ind[i:i+batch_size])
        y_b = y[ ind[ i : i + batch_size ] ]
        batches.append((x_b, y_b))
    
    ##########################
    
    return batches
