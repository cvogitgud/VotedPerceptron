import numpy as np

def create_test_sets(train_input_dir, train_label_dir):
    xtrain_orig = np.loadtxt('Xtrain.csv', delimiter=",", dtype=float)
    ytrain_orig = np.loadtxt('Ytrain.csv', delimiter=",", dtype=float)

    split_mark = int(len(xtrain_orig)/10)

    test_data = xtrain_orig[-split_mark:]
    test_labels = ytrain_orig[-split_mark:]
    np.savetxt('test_data.csv', test_data, fmt='%1d', delimiter=",")
    np.savetxt('test_labels.csv', test_labels, fmt='%1d', delimiter=",")

    split_mark = len(xtrain_orig) - split_mark

    train_data_100 = xtrain_orig[:split_mark]
    train_labels_100 = ytrain_orig[:split_mark]
    np.savetxt('train_data_100.csv', train_data_100, fmt='%1d', delimiter=",")
    np.savetxt('train_labels_100.csv', train_labels_100, fmt='%1d', delimiter=",")

    train_data_20 = xtrain_orig[:int(split_mark/5)]
    train_labels_20 = ytrain_orig[:int(split_mark/5)]
    np.savetxt('train_data_20.csv', train_data_20, fmt='%1d', delimiter=",")
    np.savetxt('train_labels_20.csv', train_labels_20, fmt='%1d', delimiter=",")

    train_data_10 = xtrain_orig[:int(split_mark/10)]
    train_labels_10 = ytrain_orig[:int(split_mark/10)]
    np.savetxt('train_data_10.csv', train_data_10, fmt='%1d', delimiter=",")
    np.savetxt('train_labels_10.csv', train_labels_10, fmt='%1d', delimiter=",")

    train_data_5 = xtrain_orig[:int(split_mark/20)]
    train_labels_5 = ytrain_orig[:int(split_mark/20)]
    np.savetxt('train_data_5.csv', train_data_5, fmt='%1d', delimiter=",")
    np.savetxt('train_labels_5.csv', train_labels_5, fmt='%1d', delimiter=",")

    train_data_2 = xtrain_orig[:int(split_mark/50)]
    train_labels_2 = ytrain_orig[:int(split_mark/50)]
    np.savetxt('train_data_2.csv', train_data_2, fmt='%1d', delimiter=",")
    np.savetxt('train_labels_2.csv', train_labels_2, fmt='%1d', delimiter=",")

    train_data_1 = xtrain_orig[:int(split_mark/100)]
    train_labels_1 = ytrain_orig[:int(split_mark/100)]
    np.savetxt('train_data_1.csv', train_data_1, fmt='%1d', delimiter=",")
    np.savetxt('train_labels_1.csv', train_labels_1, fmt='%1d', delimiter=",")