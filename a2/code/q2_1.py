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
            return self.query_knn(test_point, k-1)
        else:
            return label

def cross_validation(knn, k_range=np.arange(1,15)):
    for k in k_range:
        # Loop over folds
        # Evaluate k-NN
        # ...
        pass

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    pass

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    # Example usage:
    predicted_label = knn.query_knn(test_data[0], 2)

if __name__ == '__main__':
    main()