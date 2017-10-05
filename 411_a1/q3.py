from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np
import scipy

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
    batch_size = X.shape[0]
    gradient = (2*(X.transpose().dot(X).dot(w) - X.transpose().dot(y)))/batch_size
    return gradient


def k_lin_reg_gradient(batch_sampler, w, k):
    '''
    compute k gradients of k mini-batches
    return k_gradient: k x features
    '''
    d = w.shape[0]
    k_gradient = np.zeros((k, d))
    
    # get k random batches and compute gradient on them
    for i in range(k):
        X_b, y_b = batch_sampler.get_batch()
        batch_grad = lin_reg_gradient(X_b, y_b, w) 
        k_gradient[i] = batch_grad
        
    return k_gradient

    
def compute_m_variances(X, y, w, m, k):
    # return m_variances: (m) x features
    # m_variances[i] corresponds to m == i+1
    features = X.shape[1]
    m_variances = np.zeros((m, features))
    
    # for each batch size m, get k batches and compute variance of gradient based on k gradients
    # then add variance to m_variances
    for batch_size in range(1, 401, 1):
        batch_sampler = BatchSampler(X, y, batch_size)
        k_gradient = k_lin_reg_gradient(batch_sampler, w, k)
        variances = np.var(k_gradient, axis=0)
        m_variances[batch_size-1] = variances
    return m_variances

        
def main():
    # Load data and randomly initialise weights
    X, y, w = load_data_and_init_params()
    # Create a batch sampler to generate random batches from data
    batch_sampler = BatchSampler(X, y, BATCHES)
    # Example usage
    # X_b, y_b = batch_sampler.get_batch()
    # batch_grad = lin_reg_gradient(X_b, y_b, w)
     
    # compute mean_gradient, true_gradient, similarity
    from numpy.linalg import norm
    k_gradient = k_lin_reg_gradient(batch_sampler, w, 500)
    mean_gradient = np.average(k_gradient, axis=0)    
    true_gradient = lin_reg_gradient(X, y, w) 
    print("mean_gradient: {0}".format(mean_gradient))
    print("true_gradient: {0}".format(true_gradient))    
    
    # partial derivative estimator
    w1 = w.copy()
    w2 = w.copy()
    w1[0] = w1[0] + 0.0005
    w2[0] = w2[0] - 0.0005    
    element_0_estimator = ((norm(y - X.dot(w1))**2)/506 - (norm(y - X.dot(w2))**2)/506)/0.001
    print("element_0_estimator: {0}".format(element_0_estimator))
     
    # compute similarities   
    c_similarity = cosine_similarity(mean_gradient, true_gradient)
    s_similarity = norm(mean_gradient - true_gradient)**2
    print("cosine_similarity: {0}".format(c_similarity))
    print("square_distance_metric: {0}".format(s_similarity))  
    
    # compute m_variances
    m_variances = compute_m_variances(X, y, w, 400, 500)
    log_m_variances_0 = np.log(m_variances[:, 0])
    log_m = np.log(np.arange(1, 401, 1))
    
    # plot log(m_variances[:, 0]) against log(m)
    plt.plot(log_m, log_m_variances_0, 'b.')
    plt.title('Q3')
    plt.ylabel('log(variance(parameter_0))')
    plt.xlabel('log(m)')
    plt.show()    
    
if __name__ == '__main__':
    main()