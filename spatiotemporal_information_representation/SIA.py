import glob
import os
import time
from itertools import permutations

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score


def FPG(num):
    _permutation_list = list(permutations(range(num)))
    filter_p = []
    for _perm in _permutation_list:
        if list(_perm[::-1]) not in filter_p:
            filter_p.append(list(_perm))

    return filter_p


def calculate_MIMTS(p, U_matrix):
    MIMTS = 0
    for e in range(len(p) - 1):
        MI_item = U_matrix[p[e]][p[e + 1]]
        if not MI_item:  # if the MI is 0
            MI_item = U_matrix[p[e + 1]][p[e]]
        print(p, MI_item)
        MIMTS = MIMTS + MI_item

    return MIMTS


def load_lc_data(path, indices=0, split=False):
    txt_Paths = glob.glob(os.path.join(path, '*.txt'))
    txt_Paths.sort()
    sensor_data = []
    sensor_label = []

    for txt_item in txt_Paths:
        label = txt_item.split('-')[1]
        data_z = pd.read_csv(txt_item, delimiter=',', header=None)
        data_z = np.asarray(data_z)
        # print(data_z.shape)
        data = data_z.reshape((1, 22, 181, 180))

        # sensor array optimization
        if indices:
            data = data[0][indices]
            data = data.reshape((1, len(indices), 181, 180))

        # print(data.shape, type(data))
        if (label == 'health'):
            label_c = 0
        else:
            label_c = 1

        sensor_data.append(data)
        sensor_label.append(label_c)

    _data = np.array(sensor_data, dtype="float")
    _label = np.array(sensor_label, dtype="float")
    print(_data.shape)

    if split == True:
        # train,val-array,list
        from sklearn.model_selection import train_test_split
        x_train, x_val, y_train, y_val = train_test_split(
            _data, _label,
            test_size=0.1,
            random_state=20,  # results can re
            shuffle=True,
            stratify=_label
        )
        return x_train, x_val, y_train, y_val
    else:
        return _data, _label


def calculate_U(optim_param, indices, x_train):
    # indices = indices[:optim_param]
    # indices.sort()
    print('indices: ', indices)
    Umatrix = np.zeros([optim_param, optim_param])

    for i in range(len(x_train)):
        # for i in range(1):
        _data = x_train[i][0]
        Umatrix_i = np.zeros([optim_param, optim_param])
        # calculate the MI between 2 sensors
        for m in range(optim_param):
            Xm = np.reshape(_data[m], -1)
            for n in range(m + 1, optim_param):
                Xn = np.reshape(_data[n], -1)
                # print(mutual_info_score(Xm, Xn))
                Umatrix_i[m, n] = normalized_mutual_info_score(Xm, Xn)
        Umatrix = Umatrix + Umatrix_i

    return Umatrix


if __name__ == "__main__":
    root = "./wsd/"
    train_dir = root + "sensor_data/train"
    optim_param = 8
    indices = [3, 21, 9, 10, 14, 6, 0, 20, 7, 2, 13, 15, 12, 18, 4, 1, 16, 19, 8, 11, 17, 5]  # MAX
    indices = indices[:optim_param]
    indices.sort()
    x_train, y_train = load_lc_data(train_dir, indices, split=False)
    # calculate the sum of the NMI
    start = time.time()
    U_matrix = calculate_U(optim_param, indices, x_train)

    # calculate the filtered permutation list
    filter_p = FPG(optim_param)
    filter_p_arr = pd.DataFrame(filter_p)
    # filter_p_arr.to_csv('filter_p_8.csv')

    A = []
    for p in filter_p:
        SIFMI = calculate_SIFMI(p, U_matrix)
        A.append(SIFMI)

    A_arr = pd.DataFrame(A)

    max_A = max(A)
    max_index = A.index(max_A)
    _pc = filter_p[max_index]
    end = time.time()
    sir_time = end-start
    print(sir_time)

    root = "./wsd/"
    writein = root + "results.txt"
    output_re = open(writein, 'a')
    output_re.write('\r\n')
    output_re.write('sir_time=' + str(sir_time) + '\n')
