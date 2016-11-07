import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    didnt_meet_margin_count = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        didnt_meet_margin_count += 1
        #The gradient for the incorrect class
        dW[:,j] += X[i]
    
    #The gradient for the correct class
    dW[:,y[i]] -= didnt_meet_margin_count * X[i]

    #Factor in the Regularization
    dW += 0.5 * reg * 2 * W

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW   /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  delta = 1
  #Get scores
  scores = X.dot(W)
  margins = np.maximum(0, scores - np.matrix(scores[range(len(scores)), y]).T + delta)
  margins_copy = margins
  #Set the yi = j position to 0
  margins[np.arange(len(scores)), y] = 0
  loss = np.sum(margins)/len(X) + 0.5 * reg * np.sum(W*W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  multiplier = margins
  print multiplier.shape
  #A ExampleSize x ClassSize array that shows what intersections contributed to the error
  contributors = np.logical_and(1, multiplier)
  correct_class_partial_ders = np.sum(contributors, axis=1).reshape(contributors.shape[0])
  contributors[np.arange(len(scores)), y] = correct_class_partial_ders
  # X = N x D
  # W = D x C
  # Cont = N x C
  # Need size D x C
  # X.T * Cont
  dW = X.T * contributors
  dW += 0.5 * reg * 2 * W
  dW   /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  delta = 1
  #Get scores
  scores = X.dot(W)
  margins = np.maximum(0, scores - np.matrix(scores[range(len(scores)), y]).T + delta)
  margins_copy = margins
  #Set the yi = j position to 0
  margins[np.arange(len(scores)), y] = 0
  loss = np.sum(margins)/len(X) + 0.5 * reg * np.sum(W*W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #A ExampleSize x ClassSize array that shows what intersections contributed to the error
  multiplier = margins
  multiplier[multiplier > 0] = 1
  correct_class_partial_ders = np.sum(multiplier, axis=1).reshape(multiplier.shape[0])
  multiplier[np.arange(len(scores)), y] = correct_class_partial_ders * -1
  # X = N x D
  # W = D x C
  # Cont = N x C
  # Need size D x C
  # X.T * Cont
  dW = X.T * multiplier
  dW += 0.5 * reg * 2 * W
  dW /= X.shape[0]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
