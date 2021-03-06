'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
from sklearn.model_selection import KFold
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point

        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        # get l2 distances between test_point and all train_data
        l2_dists = self.l2_distance(test_point)
        # get the k indices and corresponding training labels with the smallest distance
        k_indices = np.argpartition(l2_dists, k)[:k]
        k_labels = self.train_labels[k_indices]
        # get number of occurrence of each label among k labels, and find which label occurs most often
        uniques, counts = np.unique(k_labels, return_counts=True)
        max_count_index = np.argmax(counts)
        label = uniques[max_count_index]
        # check if there are ties, if there are, reduce k and call recursion.
        if np.argwhere(counts == counts[max_count_index]).shape[0] > 1:
            return self.query_knn(test_point, k - 1)
        else:
            return label


def cross_validation(input_data, input_labels, k_range=range(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    kf = KFold(n_splits=10, shuffle=True)
    optimal_k = 0
    optimal_avg_accuracy = 0
    for k in k_range:
        accuracies = np.zeros(0)
        # loop through 10 folds and compute average accuracy
        for train_index, test_index in kf.split(input_data):
            train_data = input_data[train_index]
            train_labels = input_labels[train_index]
            test_data = input_data[test_index]
            test_labels = input_labels[test_index]
            knn = KNearestNeighbor(train_data, train_labels)
            accuracy = classification_accuracy(knn, k, test_data, test_labels)
            accuracies = np.append(accuracies, accuracy)
        avg_accuracy = np.mean(accuracies)
        if optimal_avg_accuracy < avg_accuracy:
            optimal_avg_accuracy = avg_accuracy
            optimal_k = k
        
    return optimal_k, optimal_avg_accuracy


def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    accuracy: percentage of correct predication
    '''
    num_data = eval_data.shape[0]
    num_correct_predict = 0
    for i in range(num_data):
        data = eval_data[i]
        true_label = eval_labels[i]
        predicted_label = knn.query_knn(data, k)
        if predicted_label == true_label:
            num_correct_predict += 1
    return float(num_correct_predict)/num_data


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    # q2.1.1 report train, test classification accuracies for k = 1, 15
    for k in [1, 15]:
        train_accuracy = classification_accuracy(knn, k, train_data, train_labels)
        test_accuracy = classification_accuracy(knn, k, test_data, test_labels)
        print('k={:2d} | train accuracy: {:.5f} | test accuracy: {:.5f}'.format(k, train_accuracy, test_accuracy))    
    
    # q2.1.3    
    # use 10-fold cross validation to find optimal k
    # report train, average, test accuracies for optimal k
    optimal_k, optimal_avg_accuracies = cross_validation(train_data, train_labels)
    optimal_train_accuracy = classification_accuracy(knn, optimal_k, train_data, train_labels)
    optimal_test_accuracy = classification_accuracy(knn, optimal_k, test_data, test_labels)
    print('optimal_k={:2d} | train accuracy: {:f} | average accuracy: {:f} | test accuracy: {:f}'.format\
          (optimal_k, optimal_train_accuracy, optimal_avg_accuracies, optimal_test_accuracy))        

if __name__ == '__main__':
    main()