import numpy as np
import test_file_creator


# runs voted perceptron on given training set and outputs predictions of test data set
# into pred_file
def run(Xtrain_file, Ytrain_file, test_data_file, pred_file):
    # read in xtrainfile, ytrainfile, and testdatafile
    train_data = np.loadtxt(Xtrain_file, delimiter=",", dtype=float)
    train_label = np.loadtxt(Ytrain_file, delimiter=",", dtype=float)

    # change train_label from 0,1 to -1, 1 to work with our algorithms
    for i in range(len(train_label)):
        if train_label[i] == 0:
            train_label[i] = -1

    test_data = np.loadtxt(test_data_file, delimiter=",", dtype=float)
    epochs = 2
    weights = voted_perceptron(train_data, train_label, epochs)
    prediction = np.array([my_prediction(weights, point) for point in test_data], dtype=object)

    np.savetxt(pred_file, prediction, fmt='%1d', delimiter=",")
    # save predictions (copy from run.py in hw1)
    return 0

# ---------------------------------------------------------------------------------------

# training the model

# input: training files, T - number of epochs
# output: list of weighted vectors produced by perceptrons
# produces multiple perceptron vectors, stored in v[],
# each with an associated weight, stored in c[]
# weights are determined by lifetime - the longer a perceptron
# lasts without making a misclassification, the heavier the weight
def voted_perceptron(train_data, train_label, T):
    # initialize variables and lists
    dimensions = len(train_data[0])
    k = 0                              # index for v and c
    v = np.array([[0] * dimensions])   # initialize first perceptron vector
    c = [0]                            # initialize first weight
    t = 0

    while t <= T:
        for i in range(len(train_data)):
            x = train_data[i]
            y_hat = sign(np.dot(v[k], x))
            y = train_label[i]
            if y_hat == y:
                c[k] = c[k] + 1
            else:  # misclassification
                v_prime = np.add(v[k], y * x)              # update perceptron (as new one)
                v = np.append(v, [v_prime], axis=0)        # add new perceptron vector to our list
                c.append(1)                                # begin new weight
                k += 1
        t = t + 1
    weights = []
    for i in range(k):
        weights.append([v[i], c[i]])
    return weights


def sign(x):
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        return 0


# predicts the class of a data point by voting amongst the predictions
# from the learned weighted perceptrons
# input: list of perceptrons and their weights, data point to classify
# output: predicted label of data point
def my_prediction(weighted_perceptrons, point):
    s = 0
    for i in range(len(weighted_perceptrons)):
        weight_pair = weighted_perceptrons[i]
        v_i = weighted_perceptrons[i][0]
        c_i = weighted_perceptrons[i][1]
        s = s + c_i * sign(np.dot(v_i, point))

    if sign(s) <= 0:
        return 0
    else:
        return 1


if __name__ == "__main__":
    test_file_creator.create_test_sets('Xtrain.csv', 'Ytrain.csv')
    numbers = [1, 2, 5, 10, 20, 100]
    for n in numbers:
        train_input_dir = 'data/train_data_%i.csv' % n
        train_label_dir = 'data/train_labels_%i.csv' % n
        test_input_dir = 'data/test_data.csv'
        test_label_dir = 'data/test_labels.csv'
        pred_file = 'predictions_%i.txt' % n
        run(train_input_dir, train_label_dir, test_input_dir, pred_file)

        predicted = np.loadtxt(pred_file, skiprows=0)
        actual = np.loadtxt(test_label_dir, skiprows=0)

        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for a, p in zip(actual, predicted):
            if a == 1 and p == 1:
                tp += 1
            elif a == 1 and p == 0:
                fn += 1
            elif a == 0 and p == 1:
                fp += 1
            else:
                tn += 1

        accuracy = round(100 * (tp + tn) / (tp + fp + tn + fn), 4)
        f1_score = round(100 * (2 * tp) / (tp + fn + tp + fp), 4)

        print("---- Dataset %i ----" % n)
        print("Accuracy: %s" % accuracy)
        print("F1 score: %s" % f1_score)
        print("TP : %i ; FP : %i" % (tp, fp))
        print("TN : %i ; FN : %i" % (tn, fn))
        print('')