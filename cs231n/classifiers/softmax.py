import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  N_sample = X.shape[0]
  for i in range(0, N_sample):
      X_i = X[i].reshape(1, X.shape[1]) #shape(1, D)
      class_i = np.dot(X_i, W) #shape(1, C)
      normalized_i = np.exp(class_i)/np.sum(np.exp(class_i)) #shape(1, C)
      li = -np.log(normalized_i[0,y[i]])
      loss += li
      dW[:, y[i]] -=  X_i.T.reshape(X_i.shape[1],)
      dW += np.dot(X_i.T, normalized_i)
  loss = loss/N_sample + reg * np.sum(W*W) 
  dW = dW/N_sample + 2 * reg * W
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  N_sample = X.shape[0]

  classes = np.dot(X, W) #shape(N, C)
  normalized = np.exp(classes)/(np.sum(np.exp(classes), axis=1).reshape(X.shape[0], 1)) #shape(N, C)
  l_list = -np.log(normalized[range(y.shape[0]),y])
  loss = np.sum(l_list) 
  normalized[range(N_sample), y] -= 1.0  # Sample i, class y_i subtracts by 1
  dW += np.dot(X.T, normalized) #shape(D, N)x(N, C)=(D,C)

  loss = loss/N_sample + reg * np.sum(W*W) 
  dW = dW/N_sample + 2 * reg * W

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

