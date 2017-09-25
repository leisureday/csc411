from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

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
        plt.plot(X[:, i], y, 'ro')
        plt.title(features[i])
        plt.ylabel('target')
    
    plt.tight_layout()
    plt.show()


def fit_regression(X,Y):
    #TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!
    raise NotImplementedError()

def main():
    # Load the data
    X, y, features = load_data()
    """
    print(type(X))
    print(type(y))
    print(type(features))
    print(X[1])
    print("X: {}".format(X))
    print("y: {}".format(y))
    """
    print("Features: {}".format(features))
    
    # Visualize the features
    visualize(X, y, features)

    #TODO: Split data into train and test

    # Fit regression model
    #w = fit_regression(X, y)

    # Compute fitted values, MSE, etc.


if __name__ == "__main__":
    main()

