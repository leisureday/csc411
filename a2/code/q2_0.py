'''
Question 2.0 Skeleton Code

Here you should load the data and plot
the means for each of the digit classes.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def plot_means(train_data, train_labels):
    means = []
    for i in range(0, 10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        # Compute mean of class i
        i_digits_mean = np.mean(i_digits, 0).reshape((8,8))
        means.append(i_digits_mean)

    # Plot all means on same axis
    all_concat = np.concatenate(means, 1)
    plt.ion()    
    plt.imshow(all_concat, cmap='gray')
    plt.show()

if __name__ == '__main__':
    # extract from ../a2digits.zip to data
    train_data, train_labels, _, _ = data.load_all_data_from_zip('../a2digits.zip', 'data')
    plot_means(train_data, train_labels)
