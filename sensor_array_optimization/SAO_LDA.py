# LDA-based sensor array optimization
import glob
import math
import os
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
"""
The implementation of SAO algorithm
"""

def OP_LDA_2class(data, label):
    label_list = list(set(label))
    X1 = np.array([data[i] for i in range(len(data)) if label[i] == 0])
    X2 = np.array([data[i] for i in range(len(data)) if label[i] == 1])

    len1 = len(X1)
    len2 = len(X2)

    # mean vector u1,u2
    miu1 = np.mean(X1, axis=0)
    miu2 = np.mean(X2, axis=0)

    # calculate S_w
    # \sum_0
    conv1 = np.dot((X1 - miu1).T, (X1 - miu1))
    # \sum_1
    conv2 = np.dot((X2 - miu2).T, (X2 - miu2))
    Sw = conv1 + conv2

    # calculate w
    w = np.dot(np.mat(Sw).I, (miu1 - miu2).reshape((len(miu1), 1)))

def OP_LDA(data, label):
    label_list = list(set(label))

    np.set_printoptions(precision=4)
    mean_vectors = []
    for c1 in label_list:
        # The mean value of each channel under each category
        mean_vectors.append(np.mean(data[label == c1], axis=0))

    # calculate Sw Sb
    S_W = np.zeros((22, 22))
    for c1, mv in zip(label_list, mean_vectors):
        # scatter matrix for every class
        class_sc_mat = np.zeros((22, 22))
        for row in data[label == c1]:
            # make column vectors
            row, mv = row.reshape(22, 1), mv.reshape(22, 1)
            class_sc_mat += (row - mv).dot((row - mv).T)
        # sum class scatter metrices
        S_W += class_sc_mat

    # 2 class
    S_B = np.zeros((22, 22))
    miu0 = mean_vectors[0].reshape(22,1)
    miu1 = mean_vectors[1].reshape(22, 1)
    S_B += (miu0 - miu1).dot((miu0 - miu1).T)
    # print('between-class Scatter matrix:\n', pd.DataFrame(S_B))

    # c class
    # overall_mean = np.mean(data, axis=0)
    # S_B = np.zeros((22, 22))
    # for i, mean_vec in enumerate(mean_vectors):
    #     n = data[label == i + 1, :].shape[0]
    #     # make column vector
    #     mean_vec = mean_vec.reshape(22, 1)
    #     # make column vector
    #     overall_mean = overall_mean.reshape(22, 1)
    #     S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    # print('between-class Scatter matrix:\n', pd.DataFrame(S_B))

    # Solve the eigenvalues and eigenvectors of Sw(-1)*Sb
    print(S_W)
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))# 22./22*22

    for i in range(len(eig_vals)):
        eigvec_sc = eig_vecs[:, i].reshape(22, 1)
        # print('\n Eigenvector {}: \n {}'.format(i + 1, eigvec_sc.real))
        # print('Eigenvalue {: }: {:.2e}'.format(i + 1, eig_vals[i].real))

    # make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

    # sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

    W = eig_pairs[0][1].reshape(22, 1)
    # print(W.shape)
    # print('Matrix W: \n', pd.DataFrame(W.real))

    W_sum = np.sum(W, axis=1)
    W_abs = np.abs(W_sum)
    # print(W_abs)

    indices = np.argsort(- W_abs)
    w_sort = np.sort(W_abs)[::-1]

    return indices, w_sort


def svm_optimize_train(x, y, feature_name, method="LDA"):
    """
    Array optimization training based on SVM model
    :param x: training data
    :param y: training label
    :param method: LDA
    """

    sensors = np.arange(22)

    train_X, test_X, train_Y, test_Y = train_test_split(x, y, test_size=0.1, random_state=0)
    zscore = StandardScaler()
    train_X = zscore.fit_transform(train_X)
    test_X = zscore.transform(test_X)

    train_Y = train_Y.astype('int')
    test_Y = test_Y.astype('int')

    if method == "LDA":
        print("LDA in using")
        indices, w_sort = OP_LDA(train_X, train_Y)# the order of sensors

    # sen = np.array(sensors)
    # id_sensors = sen[indices]
    # pd.DataFrame([id_sensors, w_sort]).to_excel(feature_name + "_sensors.xlsx", index=False, header=False)

    C_para = []
    G_para = []
    train_acc = []
    ACC = []

    # The minimum number of statistical labels, used as the fold number of cross-validation（<=10）
    cnt_label_min = 10
    for label in set(train_Y):
        cnt_label_min = min(cnt_label_min, train_Y.to_list().count(label))
    print("Currently" + str(cnt_label_min) + " -fold cross validation")

    for i in range(1, len(indices) + 1, 1):
        # print(i) # nuumbers of selected subset

        a = indices[0:i]#the selected subset
        print("the selected subset is:",a)
        train_data = train_X[:, a]
        # print(train_data.shape, train_X.shape) #(136, 1) (136, 22)
        test_data = test_X[:, a]

        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)

        max_auc = 0
        best_C = 0
        best_G = 0

        c_list = [22, 24, 26, 28, 30] #gridsearch
        g_list = [-5, -4, -3, -2, -1]

        for C in c_list:
            for g in g_list:
                gamma = math.pow(2, g)
                auc = cross_val_score(SVC(C=C, kernel='rbf', gamma=gamma), train_data, train_Y, cv=cnt_label_min,
                                      scoring='accuracy').mean()  # 'roc_auc' or 'accuracy'
                if auc > max_auc:
                    max_auc = auc
                    best_C = C
                    best_G = gamma
        train_acc.append(max_auc)
        best_classifier = SVC(C=best_C, kernel='rbf', gamma=best_G)
        best_classifier.fit(train_data, train_Y)
        acc = best_classifier.score(test_data, test_Y)
        ACC.append(acc)  # best accuracy
        C_para.append(best_C)
        G_para.append(best_G)

    print(len(train_acc), len(ACC))
    x = range(1, len(indices)+1)
    pd.DataFrame([x, ACC]).to_excel(feature_name + ".xlsx", index=False, header=False)
    plt.plot(x,train_acc, label='train_acc')
    plt.plot(x, ACC, label='valid_acc')
    plt.title(feature_name +"_lda_optim.png")
    plt.legend()
    plt.savefig(feature_name +"_lda_optim.png")
    plt.show()

    best_sensors = []
    c = ACC.index(max(ACC))  #
    print("ACC: ", max(ACC))
    model_c = C_para[c]
    model_g = G_para[c]
    sel = indices[0:c + 1]
    # joblib.dump(sel, 'sel.pkl')

    for j in range(len(sel)):
        d = sel[j]
        best_sensors.append(sensors[d])
        # best_sensors.append(d)
    print("lda selected sensors：", best_sensors)

    model_data = test_X[:, sel]
    # joblib.dump(model_data, 'model_data.pkl')
    model_svm = SVC(C=model_c, kernel='rbf', gamma=model_g, probability=True, break_ties=True, random_state=0)
    model_svm.fit(model_data, test_Y)
    # joblib.dump(model_svm, 'model_svm.pkl')
    return zscore, sel, best_sensors, model_svm, indices


def array_optimization_output(sensors, feature_name,indices):
    f = open('array_optimization_result.txt', 'a')
    f.write("\n")
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    f.write('time:' + str(t) + '\n')
    f.write('indices:' + str(indices) + '\n')
    f.write(feature_name + "lda best sensors: " + str(len(sensors))+"\n")
    # best_sensors = "; ".join(sensors)
    # f.write(best_sensors)


def load_lc_data(path, feature_name):
    txt_Paths = glob.glob(os.path.join(path, '*.txt'))
    txt_Paths.sort()
    sensor_data = []
    sensor_label = []
    features = []

    for txt_item in txt_Paths:
        # print(excelPaths_item)
        label = txt_item.split('-')[1]
        if (label == 'health'):
            label_c = 0
        else:
            label_c = 1
        data_z = pd.read_csv(txt_item, delimiter=',', header=None)

        data_z = np.asarray(data_z) #(22, 360)
        data_t = np.transpose(data_z)#(360, 22)

        # mean filter
        sensor_data2 = data_t.copy()
        t = 5
        m = len(data_t)
        for j in range(t - 1, m):
            for k in range(22):
                sum = np.sum(data_t[j - t + 1:j + 1, k])
                min = np.amin(data_t[j - t + 1:j + 1, k])
                max = np.amax(data_t[j - t + 1:j + 1, k])
                sensor_data2[j, k] = (sum - max - min) / (t - 2)

        feature = []
        feature.append(label_c)
        # print(sensor_data2.shape) #360*22
        if feature_name =="MAX":
            fea = np.max(sensor_data2, axis=0).tolist()
        if feature_name == "MEAN":
            fea = np.mean(sensor_data2, axis=0).tolist()
        if feature_name == "AUC":
            fea = np.trapz(sensor_data2, axis=0).tolist()
        feature.extend(fea)
        features.append(feature)

    features1 = np.array(features)
    features2 = pd.DataFrame(features1)

    return features2


def optimize_train(path, feature_name):
    data_train = load_lc_data(path, feature_name)
    fea_train = data_train.iloc[:, 1:23]
    label_train = data_train.iloc[:,0]

    root = "./wsd/"
    writein = root + "results.txt"
    output_re = open(writein, 'a')
    output_re.write('\r\n')

    start = time.time()
    scaler, sel_sensors, best_sensors, classifier, indices = svm_optimize_train(fea_train, label_train,feature_name, method="LDA")
    end = time.time()

    lda_time = end-start
    print(lda_time)
    output_re.write('lda_time=' + str(lda_time) + '\n')

    joblib.dump(data_train, "LDA_SVM_OP_data_train.pkl")
    joblib.dump(scaler, "LDA_SVM_OP_scaler.pkl")
    joblib.dump(classifier, "LDA_SVM_OP_classifier.pkl")
    joblib.dump(sel_sensors, "LDA_SVM_OP_sel.pkl")

    array_optimization_output(best_sensors, feature_name, indices)

# for i in range(5):
train_dir = r'./sensor_data/train'
optimize_train(train_dir, feature_name="MAX")