import sys
import numpy as np
import math
import csv


# read a csv file and output label, data, num_sample
# label: y, data: x (x_0 = 1 for bias), num_sample: number of samples
def csv_reader(file_name):
    with open(file_name, 'r') as file:
        data = csv.reader(file, delimiter=',')
        matrix = []
        for row in data:
            matrix.append(row)
    np_data = np.asfarray(matrix, float)  # from string to float
    [label, data] = np.split(np_data, [1], axis=1)
    num_sample = np.size(label)
    one = np.ones((num_sample, 1))  # bias column set to 1
    data = np.concatenate([one, data], axis=1)  # concatenate (one, data)
    return label, data, num_sample


# Turn the label into one-hot matrix
def one_hot_label(label):
    one_hot = np.zeros((label.size, num_class))
    row = 0
    for ele in label:
        one_hot[row][int(ele)] = 1
        row += 1
    return one_hot


# initialize alpha and beta
def para_init(num_feature, num_hidden, num_class, init_flag):
    zero_alpha = np.zeros((num_hidden, 1))  # bias column: all set to 0
    zero_beta = np.zeros((num_class, 1))  # bias column: all set to 0
    if init_flag == 1:  # random initialization from (-0.1,0.1)
        alpha = np.random.uniform(-0.1,0.1,(num_hidden, num_feature))
        beta = np.random.uniform(-0.1,0.1,(num_class, num_hidden))
    else:  # zero initialization
        alpha = np.zeros((num_hidden, num_feature))
        beta = np.zeros((num_class, num_hidden))
    # combine
    alpha = np.concatenate([zero_alpha, alpha], axis=1)
    beta = np.concatenate([zero_beta, beta], axis=1)
    return alpha, beta


# Stochastic Gradient Descent (SGD)
def SGD(training_data, training_label, test_data, test_label):
    [alpha, beta] = para_init(num_feature, num_hidden, num_class, init_flag)  # initialize parameters
    J_D_mean = list();
    J_Dt_mean = list()
    training_label_hot = one_hot_label(training_label)  # matrix of train label
    test_label_hot = one_hot_label(test_label)  # matrix of test label
    for epoch in range(num_epoch):  # for each epoch
        for x, y in zip(training_data, training_label_hot):
            o = NNForward(x, y, alpha, beta)  # Compute neural networks layers
            [g_a, g_b] = NNBackward(x, y, alpha, beta, o)  # Compute gradients via backprop
            # Update parameters
            alpha -= learning_rate * g_a
            beta -= learning_rate * g_b
        # Evaluate training mean cross-entropy J_D(alpha,beta)
        J_D = 0
        num_sample_train = training_label.size  # number of samples in training data
        for x, y in zip(training_data, training_label_hot):
            J_D += NNForward(x, y, alpha, beta)[5]
        J_D_mean.append(J_D / num_sample_train)
        # Evaluate test mean cross-entropy J_Dt(alpha,beta)
        J_Dt = 0
        num_sample_test = test_label.size  # number of samples in test data
        for x, y in zip(test_data, test_label_hot):
            J_Dt += NNForward(x, y, alpha, beta)[5]
        J_Dt_mean.append(J_Dt / num_sample_test)
    return alpha, beta, J_D_mean, J_Dt_mean


# Prediction after SGD given data, output a list of predicted label
def PREDICT(data, label, alpha, beta, label_out):
    num_sample = label.size
    mismatch = 0
    label_hot = one_hot_label(label)
    with open(label_out, 'w') as out:
        for sample, actual_out, actual_m in zip(data, label, label_hot):
            o = NNForward(sample, actual_m, alpha, beta)
            y_hat = o[4];
            index = 0;
            max_prob = 0;
            max_index = 0
            max_index = np.argmax(y_hat)
            if max_index != actual_out:  # compare the predicted LABEL with the real LABEL
                mismatch += 1
            print(max_index, file=out)
    error = mismatch / num_sample
    return error


# Forward Computation: Training Example (x,y), Parameters (alpha,beta)
def NNForward(x, y, alpha, beta):
    a = LinearForward(x.T, alpha)
    z = SigmoidForward(a)
    b = LinearForward(z, beta)
    y_hat = SoftmaxForward(b)
    J = CrossEntropyForward(y, y_hat)
    o = [x, a, z, b, y_hat, J]
    return o


def NNBackward(x, y, alpha, beta, o):
    x = o[0];
    a = o[1];
    z = o[2];
    b = o[3];
    y_hat = o[4];
    J = o[5];
    g_J = 1
    # g_y_hat = CrossEntropyBackward(y, y_hat, J, g_J)
    # g_b = SoftmaxBackward(b, y_hat, g_y_hat)
    g_b = y_hat
    g_b -= y
    (g_beta, g_z) = LinearBackward(z, beta, b, g_b)
    g_z = g_z[1:, :]
    z = z[1:]
    g_a = SigmoidBackward(a, z, g_z)
    (g_alpha, g_x) = LinearBackward(x, alpha, a, g_a)
    return g_alpha, g_beta


def LinearForward(a, w):
    b = np.dot(w, a)
    return b


def LinearBackward(a, w, b, g_b):
    g_b = np.reshape(g_b, [g_b.size, 1])  # reshape to matrix
    a = np.reshape(a, [a.size, 1])  # reshape to matrix
    g_w = np.dot(g_b, a.T)
    g_a = np.dot(w.T, g_b)
    return g_w, g_a


def SigmoidForward(a):
    one = np.ones(a.size)
    exp = np.exp(-a)
    b = one / (1 + exp)
    b = np.concatenate((np.ones(1), b))  # add a bias term in front
    return b


def SigmoidBackward(a, b, g_b):
    b_m = np.multiply(b, 1 - b)
    b_m = np.reshape(b_m, [b_m.size, 1])
    g_a = np.multiply(g_b, b_m)
    return g_a


def SoftmaxForward(b):
    exp = np.exp(b)
    sum_exp = np.zeros(b.size) + sum(exp)
    softmax = exp / sum_exp
    return softmax


def SoftmaxBackward(a, b, g_b):
    diag = np.diag(b.T)
    g_a = np.dot(g_b.T, (diag - b * b.T))
    return g_a.T


def CrossEntropyForward(a, a_hat):
    log_a_hat = np.log(a_hat)
    b = - np.dot(a.T, log_a_hat)
    return b  # make the return as a float value


def CrossEntropyBackward(a, a_hat, b, g_b):
    g_a_hat = -g_b * (a / a_hat)
    return g_a_hat


# main
if __name__ == "__main__":
    # Command line arguments
    train_input = sys.argv[1]  # path to the training input .csv file
    test_input = sys.argv[2]  # path to the validation input .csv file
    train_out = sys.argv[3]  # path to the training output .labels file
    test_out = sys.argv[4]  # path to the test output .labels file
    metrics_out = sys.argv[5]  # path to the output .txt to which metrics like train and test error should be written
    num_epoch = int(sys.argv[6])  # integer specifying # of times backpropagation loops through all of the training data
    num_hidden = int(sys.argv[7])  # positive integer specifying the number of hidden units
    init_flag = int(
        sys.argv[8])  # integer taking value 1 or 2 specifies whether to use RANDOM-1 or ZERO-2 initialization
    learning_rate = float(sys.argv[9])  # float value specifying the learning rate for SGD

    # global variable
    num_feature = 128
    num_class = 10

    # start
    (train_label, train_data, train_sample) = csv_reader(train_input)  # read train data
    (test_label, test_data, test_sample) = csv_reader(test_input)  # read test data
    (alpha, beta, J_D_mean, J_Dt_mean) = SGD(train_data, train_label, test_data, test_label)  # SGD
    train_error = PREDICT(train_data, train_label, alpha, beta, train_out)  # predict train data
    test_error = PREDICT(test_data, test_label, alpha, beta, test_out)  # predict test data

    with open(metrics_out, 'w') as metrics:  # Output metrics
        for epoch in range(num_epoch):
            print("epoch=" + str(epoch + 1) + " crossentropy(train): ", J_D_mean[epoch], end='\n', file=metrics)
            print("epoch=" + str(epoch + 1) + " crossentropy(test): ", J_Dt_mean[epoch], end='\n', file=metrics)
        print("error(train): ", train_error, end='\n', file=metrics)
        print("error(test): ", test_error, end='\n', file=metrics)
