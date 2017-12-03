'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for k in range(10):
        class_data = train_data[train_labels==k]
        class_mean = np.mean(class_data, axis=0)
        means[k] = class_mean
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    means = compute_mean_mles(train_data, train_labels)
    for k in range(10):
        class_data = train_data[train_labels==k]
        class_mean = means[k]
        class_size = class_data.shape[0]
        sum_covariance = np.zeros((64, 64))
        # compute covariance for each data point then take average
        # added 0.01I to each average covariance for regularization and numerical stability 
        for n in range(class_size):
            single_data = class_data[n]     
            sum_covariance += \
            np.dot((single_data - class_mean).reshape((64,1)), (single_data - class_mean).reshape((1, 64)))
            class_covariance = (sum_covariance/class_size) + 0.01*np.identity(64)
            covariances[k] = class_covariance
    return covariances

def plot_cov_diagonal(covariances):
    # Plot the diagonal of each covariance matrix side by side
    img_concat = np.zeros((8,0))
    for i in range(10):
        log_cov_diag = np.log(np.diag(covariances[i]))
        log_cov_img = log_cov_diag.reshape((8, 8))
        img_concat = np.append(img_concat, log_cov_img, axis=1)
    plt.ion()
    plt.imshow(img_concat, cmap='gray')
    plt.show()

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    import math
    from numpy.linalg import det, inv
    from numpy import log
    
    num_data_points = digits.shape[0]
    log_likelihood = np.zeros((num_data_points, 10))
    for n in range(num_data_points):
        digit = digits[n].reshape((64, 1))
        for k in range(10):
            mean = means[k].reshape((64, 1))
            covariance = covariances[k]
            log_likelihood_nk = \
                (-32)*log(math.pi*2) - 0.5*log(det(covariance)) - \
                0.5*(digit-mean).transpose().dot(inv(covariance)).dot(digit-mean)
            log_likelihood[n,k] = log_likelihood_nk
    return log_likelihood

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    from numpy import log, exp
    generative_log_likelihood = generative_likelihood(digits, means, covariances)
    num_data_points = digits.shape[0]
    conditional_log_likelihood = np.zeros((num_data_points, 10))
    py = 0.1
    for n in range(num_data_points):
        px = np.sum(exp(generative_log_likelihood[n]))*py
        for k in range(10):
            log_pxy = generative_log_likelihood[n, k]
            log_pyx = log_pxy + log(py) - log(px)
            conditional_log_likelihood[n, k] = log_pyx
    return conditional_log_likelihood

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    conditional_log_likelihood = conditional_likelihood(digits, means, covariances)
    num_data_points = digits.shape[0]
    sum_conditional_log_likelihood = 0
    for n in range(num_data_points):
        k = int(labels.item(n))
        sum_conditional_log_likelihood += conditional_log_likelihood[n, k]
    return sum_conditional_log_likelihood/num_data_points

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    return nx1 array
    '''
    conditional_log_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    num_data_points = digits.shape[0]
    classification = np.zeros((num_data_points, 1))
    for n in range(num_data_points):
        predicted_label = np.argmax(conditional_log_likelihood[n], axis=0)      
        classification[n] = predicted_label
    return classification

def classification_accuracy(means, covariances, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    accuracy: percentage of correct predication
    '''
    num_data = eval_data.shape[0]
    num_correct_predict = 0
    classification = classify_data(eval_data, means, covariances)
    for n in range(num_data):
        data = eval_data[n]
        true_label = eval_labels[n]
        predicted_label = classification.item(n)
        if predicted_label == true_label:
            num_correct_predict += 1
    return float(num_correct_predict)/num_data

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    
    # q2.2.1 plot diagonal of covariances
    plot_cov_diagonal(covariances)

    # q2.2.2 report average conditional log likelihood for both test and train
    train_avg_cond_log_likelihood = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    test_avg_cond_log_likelihood = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    print("average conditional log likelihood for train set: {:f}".format(train_avg_cond_log_likelihood))
    print("average conditional log likelihood for test  set: {:f}".format(test_avg_cond_log_likelihood))
    
    # q2.2.3 report classification accuracy for train and test set
    train_accuracy = classification_accuracy(means, covariances, train_data, train_labels)
    test_accuracy = classification_accuracy(means, covariances, test_data, test_labels)
    print('classification accuracy for train set: {:f}'.format(train_accuracy))
    print('classification accuracy for test  set: {:f}'.format(test_accuracy))    

if __name__ == '__main__':
    main()