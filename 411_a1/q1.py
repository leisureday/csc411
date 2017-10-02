from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import math

# used to load the data
def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    # returns tuple
    return X,y,features


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        #TODO: Plot feature i against y
        # 'ro' to plot as red circle
        plt.plot(X[:, i], y, '.')
        plt.title(features[i])
        plt.ylabel('target')
    
    plt.tight_layout()
    plt.show()


def fit_regression(X,Y):
    #TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!    
    a = np.dot(np.transpose(X), X)
    b = np.dot(np.transpose(X), Y)
    w = np.linalg.solve(a, b)
    
    return w
    

def main():
    # Load the data
    X, y, features = load_data()

    print("Features: {}".format(features))
    
    # Visualize the features
    '''
    visualize(X, y, features)
    '''
    
    #TODO: Split data into train and test
    # get a randomed array of indices
    data_count = y.shape[0]
    permutation = np.random.permutation(data_count)
    
    # divide indicies into 80%, 20%
    break_point = math.ceil(data_count*0.8)
    training_indices = permutation[0 : break_point]
    testing_indices = permutation[break_point :]
    
    #add bias term to x
    unbiased_training_X = np.take(X, training_indices, axis=0, out=None, mode='wrap')
    unbiased_testing_X = np.take(X, testing_indices, axis=0, out=None, mode='wrap')
    training_X = np.insert(unbiased_training_X, 0, 1, axis=1)
    testing_X = np.insert(unbiased_testing_X, 0, 1, axis=1)
    
    training_y = np.take(y, training_indices, axis=0, out=None, mode='wrap')
    testing_y = np.take(y, testing_indices, axis=0, out=None, mode='wrap')    

    # Fit regression model
    w = fit_regression(training_X, training_y)
    print(w)

    # Compute fitted values, MSE(mean square error), etc.
    test_count = testing_y.shape[0]
    SE = 0
    for index in range(test_count):
        x = testing_X[index, :]
        fitted_y = np.dot(x, w)
        SE += (testing_y[index] - fitted_y)**2
    
    MSE = SE/test_count
    print(MSE)
        

if __name__ == "__main__":
    main()

