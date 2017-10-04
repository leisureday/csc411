# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import scipy

from sklearn.datasets import load_boston
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
y = boston['target']
N = x.shape[0] # number of data points
d = x.shape[1] # number of features with bias

idx = np.random.permutation(range(N))


#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist


#helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        predictions =  np.array([LRLS(x_test[i,:].reshape(d,1),x_train,y_train, tau) \
                        for i in range(N_test)])
        losses[j] = ((predictions.flatten()-y_test.flatten())**2).mean()
    return losses


#to implement
# locally reweighted least squares
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    n_train = x_train.shape[0]

    # compute norms and all the values inside exp()
    norms = l2(x_train, test_datum.transpose())
    values = (-1*norms)/(2*(tau**2))

    # compute A
    A_diagonal = np.exp(values - scipy.misc.logsumexp(values))
    A = np.diagflat(A_diagonal.transpose())

    # compute w and y_hat
    a = np.add(x_train.transpose().dot(A).dot(x_train), (np.identity(d)*lam))
    b = x_train.transpose().dot(A).dot(y_train)
    w = np.linalg.solve(a, b)
    y_hat = np.dot(test_datum.transpose(), w)

    return y_hat


def run_k_fold(x,y,taus,k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is k_losses: a vector of k-fold cross validation losses one for each tau value
    '''
    ## TODO
    permutation = np.random.permutation(N)
    fold_length = np.floor(N/k)
    k_losses = np.zeros((k, taus.shape[0]))

    for i in range(k):
        # divide input data k times
        testing_indices = permutation[int(fold_length*i): int(fold_length*(i+1))]
        training_indices = np.concatenate((permutation[0: int(fold_length*i)], permutation[int(fold_length*(i+1)): -1]))

        x_test = np.take(x, testing_indices, axis=0, out=None, mode='wrap')
        y_test = np.take(y, testing_indices, axis=0, out=None, mode='wrap')
        x_train = np.take(x, training_indices, axis=0, out=None, mode='wrap')
        y_train = np.take(y, training_indices, axis=0, out=None, mode='wrap')

        # get one fold losses
        losses = run_on_fold(x_test, y_test, x_train, y_train, taus)
        k_losses[i] = losses

    average_k_losses = np.average(k_losses, axis=0)
    return average_k_losses
    ## TODO


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    losses = run_k_fold(x,y,taus,k=5)
    plt.plot(taus, losses, 'b.')
    plt.title('Q2')
    plt.ylabel('average k losses')
    plt.xlabel('tau')
    plt.show()
    print("min loss = {}".format(losses.min()))
