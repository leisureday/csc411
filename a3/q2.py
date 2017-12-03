import numpy as np 

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

np.random.seed(1847)

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''
    
    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices 

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch  


class GDOptimizer(object):
    '''
    A gradient descent optimizer with momentum
    '''

    def __init__(self, lr, beta=0.0):
        # lr: learning rate; beta: momentum
        self.lr = lr
        self.beta = beta
        self.eta = 0

    def update_params(self, params, grad):
        # Update parameters using GD with momentum and return
        # the updated parameters
        eta = -self.lr*grad+self.beta*self.eta
        params = params+eta 
        self.eta = eta
        return params
        

class SVM(object):
    '''
    A Support Vector Machine
    Assume add bias feature 1 at the end of each x
    '''

    def __init__(self, c, feature_count):
        '''
        c: penalty parameter
        w: weight parameters
        '''
        self.c = c
        self.w = np.random.normal(0.0, 0.1, feature_count)
        
    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).
        Given y in {-1, 1}, X have bias feature incorporated.
        Returns a length-n vector containing the hinge-loss per data point.
        '''
        # Implement hinge loss
        num_data = X.shape[0]
        hinge_losses = np.zeros((num_data,))
        for i in range(num_data):
            x = X[i]
            yx = y[i]*np.dot(np.transpose(self.w), x)
            hinge_loss = max(1-yx, 0)
            hinge_losses[i] = hinge_loss
        return hinge_losses
        
    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))
        Given y in {-1, 1}, X have bias feature incorporated.
        Returns the gradient with respect to the SVM parameters (shape (m,)), vector.
        '''
        # Compute (sub-)gradient of SVM objective
        num_data = X.shape[0]
        grad_w = np.copy(self.w)
        grad_w[-1] = 0 # do not regularize the bias weight
        hinge_losses = self.hinge_loss(X, y)
        for i in range(num_data):
            if hinge_losses[i] != 0:
                grad_w -= y[i]*X[i]*self.c/num_data
        return grad_w
        
    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).
        Given X have bias feature incorporated.
        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1
        return np.array([1 if np.dot(np.transpose(self.w), x)>=0 else -1 for x in X])


def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets


def optimize_test_function(optimizer, w_init=10.0, steps=200):
    '''
    Optimize the simple quadratic test function and return the parameter history.
    '''
    def func(x):
        return 0.01 * x * x

    def func_grad(x):
        return 0.02 * x

    w = w_init
    w_history = [w_init]

    for step in range(steps):
        # Optimize and update the history
        w = w_history[-1]
        w = optimizer.update_params(w, func_grad(w))
        w_history.append(w)
        
    return w_history


def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.
    penalty: penalty parameter
    iters: number of iterations
    Assume train_data has bias feature 1 incorporated
    '''
    num_feature = train_data.shape[1]
    svm = SVM(penalty, num_feature)
    batchSampler = BatchSampler(train_data, train_targets, batchsize)
    
    for i in range(iters):
        batch_data, batch_targets = batchSampler.get_batch()
        grad = svm.grad(batch_data, batch_targets)
        svm.w = optimizer.update_params(svm.w, grad)
        
    return svm

    
if __name__ == '__main__':
    # q2.1 implement SGD with momentum and optimize test function
    '''
    test_optimizer = GDOptimizer(1.0, 0.9)
    w_history = optimize_test_function(test_optimizer)
    plt.plot(w_history)
    plt.ylabel('X_batct')
    plt.xlabel('steps') 
    plt.title('optimize test function')
    plt.show()
    '''

    # q2.3 apply svm on 4vs9 digits on MNIST
    # get data and add bias term to X
    train_data, train_targets, test_data, test_targets = load_data()
    train_data = np.append(train_data, np.ones((train_data.shape[0], 1)), axis=1)
    test_data = np.append(test_data, np.ones((test_data.shape[0], 1)), axis=1)
    
    # train svms, alpah=0.05, beta1=0.0, beta2=0.1, penalty=1.0, batch_size=100, iters=500
    optimizer1 = GDOptimizer(0.05, 0.0)
    optimizer2 = GDOptimizer(0.05, 0.1)
    svm1 = optimize_svm(train_data, train_targets, 1.0, optimizer1, 100, 500)
    svm2 = optimize_svm(train_data, train_targets, 1.0, optimizer2, 100, 500)
    
    # get predictions
    train_pred1 = svm1.classify(train_data)
    test_pred1 = svm1.classify(test_data)
    train_pred2 = svm2.classify(train_data)
    test_pred2 = svm2.classify(test_data)
    
    # get and print accuracies
    train_accuracy1 = (train_pred1==train_targets).mean()
    test_accuracy1 = (test_pred1==test_targets).mean()
    train_accuracy2 = (train_pred2==train_targets).mean()
    test_accuracy2 = (test_pred2==test_targets).mean()
    print("train accuracy with beta=0.0: {}\n".format(train_accuracy1))
    print("test accuracy with beta=0.0 : {}\n".format(test_accuracy1))
    print("train accuracy with beta=0.1: {}\n".format(train_accuracy2))
    print("test accuracy with beta=0.1 : {}\n".format(test_accuracy2))
    
    # get and print hinge loss
    train_hinge_losses1 = svm1.hinge_loss(train_data, train_targets)
    test_hinge_losses1 = svm1.hinge_loss(test_data, test_targets)
    train_hinge_losses2 = svm2.hinge_loss(train_data, train_targets)
    test_hinge_losses2 = svm2.hinge_loss(test_data, test_targets) 
    train_loss1 = np.mean(train_hinge_losses1)
    test_loss1 = np.mean(test_hinge_losses1)
    train_loss2 = np.mean(train_hinge_losses2)        
    test_loss2 = np.mean(test_hinge_losses2)
    print("train loss with beta=0.0: {}\n".format(train_loss1))
    print("test loss with beta=0.0 : {}\n".format(test_loss1))
    print("train loss with beta=0.1: {}\n".format(train_loss2))
    print("test loss with beta=0.1 : {}\n".format(test_loss2))
    
    # get w and plot, remove bias weight
    w1 = np.reshape(svm1.w[0:-1], (28,28))
    w2 = np.reshape(svm2.w[0:-1], (28,28))
    plt.ion()
    plt.imshow(w1, cmap='gray')
    plt.title("beta=0.0")
    plt.show()    
    plt.ion()
    plt.imshow(w2, cmap='gray')
    plt.title("beta=0.1")
    plt.show()     

