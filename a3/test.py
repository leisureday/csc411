import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

def logistic_regression(train_dat, train_labels, test_data, test_labels):
    # turn down tolerance for short training time
    # tol: Tolerance for stopping criteria.
    # C: Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
    model = LogisticRegression(penalty='l2', tol=0.01)
    model.fit(train_data, test_data)
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(binary_test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model
    

plt.show()
Total running time of the script: ( 0 minutes 0.556 seconds)