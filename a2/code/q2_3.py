'''
Question 2.3 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    # compute using MLE with augmented data
    eta = np.zeros((10, 64))
    for k in range(10):
        class_train_data = train_data[train_labels == k]
        aug_class_train_data = np.concatenate((class_train_data, np.zeros((1, 64)), np.ones((1, 64))), axis=0)
        for j in range(64):
            features_j = aug_class_train_data[:,j]
            eta_kj = np.count_nonzero(features_j)/features_j.shape[0]
            eta[k,j] = eta_kj
    return eta

def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    img_concat = np.zeros((8,0))
    for i in range(10):
        img_i = class_images[i].reshape((8,8))
        img_concat = np.append(img_concat, img_i, axis=1)
    plt.ion()    
    plt.imshow(img_concat, cmap='gray')
    plt.show()

def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    # compute average value as pixel value
    generated_data = np.zeros((10, 64))
    generated_avg_data = np.zeros((10, 64))    
    for k in range(10):
        for j in range(64):
            p = eta[k,j]
            n = 100
            generated_pixel = np.random.binomial(1, p)
            generated_avg_pixel = np.sum(np.random.binomial(n, p))/n
            generated_data[k,j] = generated_pixel
            generated_avg_data[k,j] = generated_avg_pixel            
    plot_images(generated_data)
    plot_images(generated_avg_data)    

def generative_likelihood(digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array 
    '''
    from numpy import sum, log, exp
    num_data = digits.shape[0]
    gen_log_likelihood = np.zeros((num_data, 10))
    for n in range(num_data):
        digit = digits[n]
        for k in range(10):
            eta_k = eta[k]
            gen_log_likelihood[n,k] = sum(log(eta_k[digit==1]))+sum(log(1-eta_k[digit==0]))
    return gen_log_likelihood

def conditional_likelihood(digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    from numpy import log, exp
    gen_log_likelihood = generative_likelihood(digits, eta)
    num_data = digits.shape[0]
    cond_log_likelihood = np.zeros((num_data, 10))
    py = 0.1
    for n in range(num_data):
        px = np.sum(exp(gen_log_likelihood[n]))*py
        for k in range(10):
            log_pxy = gen_log_likelihood[n, k]
            log_pyx = log_pxy + log(py) - log(px)
            cond_log_likelihood[n, k] = log_pyx
    return cond_log_likelihood

def avg_conditional_likelihood(digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    # Compute as described above and return
    cond_log_likelihood = conditional_likelihood(digits, eta)
    num_data = digits.shape[0]
    sum_cond_log_likelihood = 0
    for n in range(num_data):
        k = int(labels.item(n))
        sum_cond_log_likelihood += cond_log_likelihood[n, k]
    return sum_cond_log_likelihood/num_data


def classify_data(digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_log_likelihood = conditional_likelihood(digits, eta)
    # Compute and return the most likely class
    num_data = digits.shape[0]
    classification = np.zeros((num_data, 1))
    for n in range(num_data):
        predicted_label = np.argmax(cond_log_likelihood[n], axis=0)      
        classification[n] = predicted_label
    return classification

def classification_accuracy(eta, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    accuracy: percentage of correct predication
    '''
    num_data = eval_data.shape[0]
    num_correct_predict = 0
    classification = classify_data(eval_data, eta)
    for n in range(num_data):
        data = eval_data[n]
        true_label = eval_labels[n]
        predicted_label = classification.item(n)
        if predicted_label == true_label:
            num_correct_predict += 1
    return float(num_correct_predict)/num_data

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)
    
    # q2.3.3 plot eta
    plot_images(eta)
    
    # q2.3.4 plot sampled data
    generate_new_data(eta)
    
    # q2.3.5 report average conditional log likelihood for both test and train
    train_avg_cond_log_likelihood = avg_conditional_likelihood(train_data, train_labels, eta)
    test_avg_cond_log_likelihood = avg_conditional_likelihood(test_data, test_labels, eta)
    print("average conditional log likelihood for train set: {:f}".format(train_avg_cond_log_likelihood))
    print("average conditional log likelihood for test  set: {:f}".format(test_avg_cond_log_likelihood))
    
    # q2.3.6 report classification accuracy for train and test set
    train_accuracy = classification_accuracy(eta, train_data, train_labels)
    test_accuracy = classification_accuracy(eta, test_data, test_labels)
    print('classification accuracy for train set: {:f}'.format(train_accuracy))
    print('classification accuracy for test  set: {:f}'.format(test_accuracy))    
    
if __name__ == '__main__':
    main()
