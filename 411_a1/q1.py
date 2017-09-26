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


#
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
    
    # get a randomed array of indices
    data_count = Y.shape[0]
    permutation = np.random.permutation(data_count)
    
    # divide into 80%, 20%
    breakPoint = math.ceil(data_count*0.8)
    trainingIndices = permutation[0 : breakPoint]
    testingIndices = permutation[breakPoint :]
    training_x = numpy.take(a, indices, axis=None, out=None, mode='raise')[source]
    
    
    # compute linear regression
    # w = numpy.linalg.solve(a, b)

def main():
    # Load the data
    X, y, features = load_data()
    
    '''
    print("Features: {}".format(features))
    '''
    
    # Visualize the features
    '''
    visualize(X, y, features)
    '''
    
    #TODO: Split data into train and test

    # Fit regression model
    w = fit_regression(X, y)

    # Compute fitted values, MSE, etc.


if __name__ == "__main__":
    main()

