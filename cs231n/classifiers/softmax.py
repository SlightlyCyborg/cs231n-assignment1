import numpy as np
from random import shuffle

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  #Normalize the scores to prevent blowup
  num_examples = X.shape[0]
  num_classes = W.shape[1]
  
  scores -= np.matrix(np.max(scores, axis=1)).T
  for example_index in xrange(X.shape[0]):
    the_sum = 0.0
    for class_index in xrange(W.shape[1]):
      the_sum += np.exp(scores[example_index][class_index])
    
    loss += np.log(the_sum)  
    loss -= scores[example_index][y[example_index]]

    for class_index in xrange(W.shape[1]):
      p =  np.exp(scores[example_index][class_index])/the_sum
      if (class_index == y[example_index]):
        dW[:, class_index] += (p-1) * X[example_index]
      else:
        dW[:, class_index] += p*X[example_index]

  #Average
  loss /= num_examples 
  dW /= num_examples 

  #Regularization
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #Scores Shape: N x C
  scores = X.dot(W)
  scores -= np.matrix(np.max(scores, axis=1)).T

  #Sums Shape: 1 x N  
  sums = np.matrix(np.sum(np.exp(scores), axis=1))


  n = X.shape[0]
  c = W.shape[1]

  print "softmax++"
  #Probs Shape: N x C
  probs = np.exp(scores) / sums.T
  correct_inds = np.zeros(probs.shape)
  correct_inds[np.arange(n), y] = 1
  probs -= correct_inds

  #Add in the sums to the loss
  loss += np.sum(np.log(sums))

  #Add in the numerator of the orignal log equation
  loss -= np.sum(scores[np.arange(len(scores)), y])

  #dW Shape: D x C
  dW += X.T.dot(probs)

  #Normalize
  loss /= n
  dW   /= n

  #Regularization
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

