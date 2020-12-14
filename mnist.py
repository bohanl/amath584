import numpy as np
from sklearn import linear_model
import idx2numpy
import matplotlib.pyplot as plt

N_train = 60000
N_test  = 10000

# Read training/test images and labels.
images_train = idx2numpy.convert_from_file('dataset/train-images.idx3-ubyte')
labels_train = idx2numpy.convert_from_file('dataset/train-labels.idx1-ubyte')
images_test = idx2numpy.convert_from_file('dataset/t10k-images.idx3-ubyte')
labels_test = idx2numpy.convert_from_file('dataset/t10k-labels.idx1-ubyte')

# Each image (28x28) is flattened into a vector as columns of `A_train` and `A_test`.
x_train = np.reshape(images_train.T, (28*28, N_train))  # 784x60000
x_test = np.reshape(images_test.T, (28*28, N_test))

# Set up label matrix where labels are transformed to columns
y_train = np.zeros((10, N_train))  # 10x60000
for i, val in enumerate(labels_train):
    y_train[val][i] = 1
y_test = np.zeros((10, N_test))
for i, val in enumerate(labels_test):
    y_test[val][i] = 1

def evaluate(y_predicted, y_test):
    n_correct = np.zeros((10,))
    n_total = np.zeros_like(n_correct)
    for j in range(N_test):
        # `y_predicted` won't be perfectly 0s and 1s
        # set the max of the column to 1 and zero out the rest
        # when making the prediction.
        i = np.argmax(y_predicted[:, j])
        y_predicted[:, j].fill(0)
        y_predicted[i, j] = 1
        n_total[labels_test[j]] += 1
        if (y_predicted[:, j] == y_test[:, j]).all():
            n_correct[labels_test[j]] += 1
    # Assuming test labels contain all digits 0-9
    # so there is no divide by 0 error.
    return n_correct / n_total


def plot_accuracy(accu, method=''):
    x = np.arange(0, 10)
    width = 0.35  # the width of the bars
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(x, accu, width, label=method)
    ax.set(xlabel='Digit', ylabel='Accuracy', title=f'MNIST')
    for i,j in zip(x, accu):
        ax.annotate("{:.2f}".format(j), xy=(i,j))
    ax.legend()
    ax.set_xticks(x)
    plt.savefig(f'/tmp/mnist-{method}.png')


# The goal is to find model parameter matrix `A` (10x784) where A * x_train = y_train

# Method 1 - least square fit via pseudo inverse.
A1 = y_train @ np.linalg.pinv(x_train)
y_predicted = A1 @ x_test
accu_lst = evaluate(y_predicted, y_test)
plot_accuracy(accu_lst, method='Least Square Fit')

# Method 2 - Lasso with a small L1 penalty `alpha`=0.1
clf = linear_model.Lasso(alpha=0.2)
clf.fit(x_train.T, y_train.T)
y_predicted = clf.predict(x_test.T).T
accu_lasso1 = evaluate(y_predicted, y_test)
plot_accuracy(accu_lasso1, method='Lasso (alpha=0.2)')

# Method 3 - Lasso with a bigger L1 penalty `alpha`=0.8
clf = linear_model.Lasso(alpha=0.8)
clf.fit(x_train.T, y_train.T)
y_predicted = clf.predict(x_test.T).T
accu_lasso2 = evaluate(y_predicted, y_test)
plot_accuracy(accu_lasso2, method='Lasso (alpha=0.8)')

# Method 4 - Ridge with an L2 penalty on weights `alpha`=0.2
clf = linear_model.Ridge(alpha=0.2)
clf.fit(x_train.T, y_train.T)
y_predicted = clf.predict(x_test.T).T
accu_ridge = evaluate(y_predicted, y_test)
plot_accuracy(accu_ridge, method='Ridge')
