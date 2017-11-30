'''
Question 1 Skeleton Code


'''

import sklearn
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree


def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))
    return newsgroups_train, newsgroups_test


def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer()
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names


def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    return tf_idf_train, tf_idf_test, feature_names


def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(binary_test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model


def logistic_regression(train_data, train_labels, test_data, test_labels):
    # turn down tolerance for short training time
    # tol: Tolerance for stopping criteria.
    # C: Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
    model = LogisticRegression(penalty='l2', tol=0.01)
    model.fit(train_data, train_labels)
    
    train_pred = model.predict(train_data)
    print('Logistic regression train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(test_data)
    print('Logistic regression test accuracy = {}'.format((test_pred == test_labels).mean()))

    cm = confusion_matrix(test_labels, test_pred)
    print(cm)

    class1, class2 = find_most_confused(cm)
    print(class1, class2)

    return model


def support_vector_machine(train_data, train_labels, test_data, test_labels):
    model = svm.LinearSVC()
    model.fit(train_data, train_labels)

    train_pred = model.predict(train_data)
    print('SVM train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(test_data)
    print('SVM test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

def decision_tree(train_data, train_labels, test_data, test_labels):
    model = tree.DecisionTreeClassifier()
    model.fit(train_data, train_labels)

    train_pred = model.predict(train_data)
    print('Decision tree train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(test_data)
    print('Decision tree test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model


def confusion_matrix(true_labels, pred_labels):
    result = np.zeros((20, 20))
    num_labels = true_labels.shape[0]
    for i in range(num_labels):
        true_label = true_labels[i]
        pred_label = pred_labels[i]
        result[pred_label, true_label] += 1
    return result


def find_most_confused(cm):
    num_classes = cm.shape[0]
    for i in range(num_classes):
        cm[i,i] = 0
    class1, class2 = np.unravel_index(cm.argmax(), cm.shape)
    return class1+1, class2+1


if __name__ == '__main__':
    train_data, test_data = load_data()
    train_bow, test_bow, feature_names = bow_features(train_data, test_data)
    # tf_idf_train, tf_idf_test, tf_idf_feature_names = tf_idf_features(train_data, test_data)

    # bnb_model = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)
    # lr_model = logistic_regression(train_bow, train_data.target, test_bow, test_data.target)
    # svm_model = support_vector_machine(train_bow, train_data.target, test_bow, test_data.target)
    # dt_model = decision_tree(train_bow, train_data.target, test_bow, test_data.target)