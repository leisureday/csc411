import numpy as np
from sklearn.datasets import load_boston

BATCHES = 50

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


def load_data_and_init_params():
    '''
    Load the Boston houses dataset and randomly initialise linear regression weights.
    '''
    print('------ Loading Boston Houses Dataset ------')
    X, y = load_boston(True)
    features = X.shape[1]

    # Initialize w
    w = np.random.randn(features)

    print("Loaded...")
    print("Total data points: {0}\nFeature count: {1}".format(X.shape[0], X.shape[1]))
    print("Random parameters, w: {0}".format(w))
    print('-------------------------------------------\n\n\n')

    return X, y, w


def cosine_similarity(vec1, vec2):
    '''
    Compute the cosine similarity (cos theta) between two vectors.
    '''
    dot = np.dot(vec1, vec2)
    sum1 = np.sqrt(np.dot(vec1, vec1))
    sum2 = np.sqrt(np.dot(vec2, vec2))

    return dot / (sum1 * sum2)


#TODO: implement linear regression gradient
def lin_reg_gradient(X, y, w):
    '''
    Compute gradient of linear regression model parameterized by w
    X: mxd
    y: mx1
    w: dx1
    '''
    gradient = 2*(X.transpose().dot(X).dot(w) - X.transpose().dot(y))
    return gradient


def k_lin_reg_gradient(batch_sampler, w, k):
    '''
    compute mean of gradient of k mini-batches
    '''
    d = w.shape[0]
    k_gradient = np.zeros((k, d))
    for i in range(k):
        X_b, y_b = batch_sampler.get_batch()
        batch_grad = lin_reg_gradient(X_b, y_b, w) 
        k_gradient[i] = batch_grad
    print("k_gradient.shape: {0}".format(k_gradient.shape))
    mean_gradient = np.average(k_gradient, axis=0)
    print("mean_gradient.shape: {0}".format(mean_gradient.shape))        
    return mean_gradient
    
    
def main():
    # Load data and randomly initialise weights
    X, y, w = load_data_and_init_params()
    # Create a batch sampler to generate random batches from data
    batch_sampler = BatchSampler(X, y, BATCHES)

    # Example usage
    # X_b, y_b = batch_sampler.get_batch()
    # batch_grad = lin_reg_gradient(X_b, y_b, w)
    mean_gradient = k_lin_reg_gradient(batch_sampler, w, 500)
    true_gradient = lin_reg_gradient(X, y, w)
    
    '''
    partial derivative estimator
    w1 = w.copy()
    w2 = w.copy()
    w1[0] = w1[0] + 0.0005
    w2[0] = w2[0] - 0.0005    
    from numpy.linalg import norm
    element_0_estimator = (norm(y - X.dot(w1))**2 - norm(y - X.dot(w2))**2)/0.001
    '''                       
    similarity = cosine_similarity(mean_gradient, true_gradient)
    print("cosine_similarity: {0}".format(similarity))
    
    
if __name__ == '__main__':
    main()