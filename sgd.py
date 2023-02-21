#################################
# Your name: Eyal Grinberg
#################################


import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt
import scipy

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels



def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    w_t = np.zeros(data.shape[1]) # dim(w_t) = 784 , w_1 = [0,0,...,0]
    for t in range(1,T + 1):
        i = np.random.randint(0, len(data), dtype=int) # sample i from 0,1,...,69999
        eta_t = eta_0 / t
        if labels[i] * np.dot(data[i], w_t) < 1:
            w_t = (1 - eta_t) * w_t + eta_t * C * labels[i] * data[i] 
        else:
            w_t = (1 - eta_t) * w_t
    return w_t

def SGD_log(data, labels, eta_0, T):
    """
    Implements SGD for log loss.
    """
    w_t = np.zeros(data.shape[1]) # dim(w_t) = 784 , w_1 = [0,0,...,0]
    for t in range(1,T + 1):
        i = np.random.randint(0, len(data), dtype=int) # sample i from 0,1,...,69999
        eta_t = eta_0 / t
        gradient_Fi_Wt = calc_gradient_Fi_Wt(w_t, data[i], labels[i])
        w_t = w_t - eta_t * gradient_Fi_Wt
    return w_t

#################################

# Place for additional code

# auxilary functions for question 1 - SGD for Hinge loss

def calculate_accuracy(classifier, data, labels):
    """
    calculate the accuracy of the classifier w_t on the given data 
    """
    error_cnt = 0
    for i in range(len(data)):
        if np.dot(classifier, data[i]) >= 0:
            prediction = 1
        else:
            prediction = -1
        error_cnt += prediction != labels[i]
    return 1 - error_cnt / len(data)

def find_best_eta_0_section_1a(T, C, train_set, train_labels, validation_set, validation_labels):
    """
    train the classifier w_t that was returned by SGD with hinge-loss (using T=1000, C=1) on the training set 
    cross validate to find best eta_0
    assess the performance of eta_0 by averaging the accuracy on the valdition set 10 times
    plot the average accuracy on the valdition set as a function of eta_0
    """
    avg_accuracy_valid_set_vec_10 = []
    possible_etas = [pow(10, j) for j in range(-5, 4)] # eta_0 = 10^-5, 10^-4, ..., 10^2, 10^3 (10^4 gives overflow)
    #possible_etas = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2] # increase resolution around eta_0 = 1
    for eta_0 in possible_etas:
        w_t = SGD_hinge(train_set, train_labels, C, eta_0, T) # train classifier on the training set
        # calculate averaged accuracy on the validation set 10 times 
        accuracy_i = 0
        for i in range(10):
            accuracy_i += calculate_accuracy(w_t, validation_set, validation_labels) 
        avg_accuracy_valid_set_vec_10.append(accuracy_i / 10)
    # plot the average accuracy on the valdition set as a function of eta_0 
    plt.plot(possible_etas, avg_accuracy_valid_set_vec_10)
    plt.xlabel('eta_0')
    plt.ylabel('averaged accuracy')
    plt.xscale('log')
    plt.ylim(0.8, 1)
    plt.show()

def find_best_C_section_1b(T, best_eta_0, train_set, train_labels, validation_set, validation_labels):
    """
    train the classifier w_t that was returned by SGD (using T=1000, eta_0 = 0.8) on the training set 
    cross validate to find best C
    assess the performance of C by averaging the accuracy on the valdition set 10 times
    plot the average accuracy on the valdition set as a function of C
    """
    avg_accuracy_valid_set_vec_10 = []
    possible_C = [pow(10, j) for j in range(-5, 6)] # C = 10^-5, 10^-4, ..., 10^4, 10^5
    #possible_C = [0.00004, 0.00006, 0.00008,0.0001, 0.00012, 0.00014, 0.00016, 0.00018 ]
    for C in possible_C:
        w_t = SGD_hinge(train_set, train_labels, C, best_eta_0, T) # train classifier on the training set 
        # calculate averaged accuracy on the validation set 10 times 
        accuracy_i = 0
        for i in range(10):
            accuracy_i += calculate_accuracy(w_t, validation_set, validation_labels) 
        avg_accuracy_valid_set_vec_10.append(accuracy_i / 10)
    # plot the average accuracy on the valdition set as a function of eta_0 
    plt.plot(possible_C, avg_accuracy_valid_set_vec_10)
    plt.xlabel('C')
    plt.ylabel('averaged accuracy')
    plt.xscale('log')
    plt.ylim(0.97, 0.99)
    plt.show()

def show_w_as_image_section_1c(T, best_C, best_eta_0, train_set, train_labels):
    w = SGD_hinge(train_set, train_labels, best_C, best_eta_0, T)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.show()

def find_accuracy_of_best_classifier_section_1d(T, best_eta_0, best_C, train_set, train_labels, test_set, test_labels):
    best_w = SGD_hinge(train_set, train_labels, best_C, best_eta_0, T)
    return calculate_accuracy(best_w, test_set, test_labels)


# auxilary functions for question 2 - SGD for log-loss

def calc_gradient_Fi_Wt(W_t, X_i, Y_i):
    """
    calculates the gradient of the i'th sample based on log-loss function at W_t
    """
    input_soft = np.array([0, Y_i*np.dot(W_t, X_i)])
    output_soft = scipy.special.softmax(input_soft) # softmax returns an array of two elements, we need only the first
    gradient_Fi_Wt = (-Y_i) * output_soft[0] * X_i # the required calculation of the gradient
    return gradient_Fi_Wt

def find_best_eta_0_section_2a(T, train_set, train_labels, validation_set, validation_labels):
    """
    train the classifier w_t that was returned by SGD with log-loss (using T=1000) on the training set 
    cross validate to find best eta_0
    assess the performance of eta_0 by averaging the accuracy on the valdition set 10 times
    plot the average accuracy on the valdition set as a function of eta_0
    """
    avg_accuracy_valid_set_vec_10 = []
    possible_etas = [pow(10, j) for j in range(-5, 6)] # eta_0 = 10^-5, 10^-4, ..., 10^4, 10^5
    for eta_0 in possible_etas:
        w_t = SGD_log(train_set, train_labels, eta_0, T) # train classifier on the training set
        # calculate averaged accuracy on the validation set 10 times 
        accuracy_i = 0
        for i in range(10):
            accuracy_i += calculate_accuracy(w_t, validation_set, validation_labels) 
        avg_accuracy_valid_set_vec_10.append(accuracy_i / 10)
    # plot the average accuracy on the valdition set as a function of eta_0 
    plt.plot(possible_etas, avg_accuracy_valid_set_vec_10)
    plt.xlabel('eta_0')
    plt.ylabel('averaged accuracy')
    plt.xscale('log')
    #plt.ylim(0, 1)
    plt.show()

def show_w_as_image_and_calc_accuracy_on_test_section_2b(T, best_eta_0, train_set, train_labels, test_set, test_labels):
    best_w = SGD_log(train_set, train_labels, best_eta_0, T)
    accuracy_on_test = calculate_accuracy(best_w, test_set, test_labels)
    print("\nthe accuracy of the best classifier on the test set is: " , accuracy_on_test)
    plt.imshow(np.reshape(best_w, (28, 28)), interpolation='nearest')
    plt.show()

def norm_through_training_section_2c(data, labels, eta_0, T):
    W_norm_vec = np.zeros(T)
    iterations_vec = np.arange(1, T+1)
    w_t = np.zeros(data.shape[1]) # dim(w_t) = 784 , w_1 = [0,0,...,0]
    for t in range(1,T + 1):
        i = np.random.randint(0, len(data), dtype=int) # sample i from 0,1,...,69999
        eta_t = eta_0 / t
        gradient_Fi_Wt = calc_gradient_Fi_Wt(w_t, data[i], labels[i])
        w_t = w_t - eta_t * gradient_Fi_Wt
        W_norm_vec[t - 1] = np.linalg.norm(w_t)
    plt.plot(iterations_vec, W_norm_vec)
    plt.xlabel('iteration')
    plt.ylabel('norm W_t')
    plt.show()

#################################

if __name__ == '__main__':
    #print("start\n")
    #train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    """
    print("\ntrain_data " , train_data.shape)
    print("\ntrain_labels " , train_labels.shape)
    print("\nvalidation_data ", validation_data.shape)
    print("\nvalidation_labels " , validation_labels.shape)
    print("\ntest_data ", test_data.shape)
    print("\ntest_labels ", test_labels.shape)
    """
    #find_best_eta_0_section_1a(1000, 1, train_data, train_labels, validation_data, validation_labels)
    #find_best_C_section_1b(1000, 1, train_data, train_labels, validation_data, validation_labels)
    #show_w_as_image(20000, 1, 0.0001, train_data, train_labels)
    #res = find_accuracy_of_best_classifier(20000, 1, 0.0001, train_data, train_labels, test_data, test_labels)
    #print("\nthe accuracy of the best classifier on the test set is: " , res )

    #find_best_eta_0_section_2a(1000, train_data, train_labels, validation_data, validation_labels)
    #show_w_as_image_and_calc_accuracy_on_test_section_2b(20000, 1, train_data, train_labels, test_data, test_labels)
    #norm_through_training_section_2c(train_data, train_labels, 0.00001, 20000)