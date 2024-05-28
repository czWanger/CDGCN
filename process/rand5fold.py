import random
from pathlib import Path
from random import shuffle
import os
import numpy as np

cwd=os.getcwd()


def loadDataList(obj):
    fold0_x_test = []
    fold0_x_train = []
    if 'PHEME' in obj:
        train_list = os.path.join(cwd, "data/" + obj + "/" + 'data.train.txt')
        test_list = os.path.join(cwd, "data/" + obj + "/" + 'data.test.txt')
        print("loading PHEME label:")
        for line in open(train_list):
            print(int(line))
            fold0_x_train.append(int(line))
        for line in open(test_list):
            print(int(line))
            fold0_x_test.append(int(line))

    fold0_test = list(fold0_x_test)
    shuffle(fold0_test)
    fold0_train = list(fold0_x_train)
    shuffle(fold0_train)

    return list(fold0_test), list(fold0_train)


def load5foldData(obj, shuffle_flag = True, seed=2020):
    random.seed(seed)
    if 'Twitter' in obj:
        labelPath = os.path.join(cwd,"data/" +obj+"/"+ obj + "_label_All.txt")
        labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']
        print("loading tree label" )
        NR,F,T,U = [],[],[],[]
        l1=l2=l3=l4=0
        labelDic = {}
        for line in open(labelPath):
            line = line.rstrip() # false	E272	656955120626880512	2	143	536	532	IJCAI	0.73
            label, eid = line.split('\t')[0], line.split('\t')[2]
            current_working_directory = os.getcwd()
            if Path('./data/Twitter15/Twitter15graph/{}.npz'.format(eid)).exists():
                labelDic[eid] = label.lower()
                if label in labelset_nonR:
                    NR.append(eid)
                    l1 += 1
                if labelDic[eid] in labelset_f:
                    F.append(eid)
                    l2 += 1
                if labelDic[eid] in labelset_t:
                    T.append(eid)
                    l3 += 1
                if labelDic[eid] in labelset_u:
                    U.append(eid)
                    l4 += 1
        print(len(labelDic))
        # print(labelDic)
        print(l1,l2,l3,l4)
        if shuffle_flag:
            print("Shuffle 5 fold...")
            random.shuffle(NR)
            random.shuffle(F)
            random.shuffle(T)
            random.shuffle(U)

        fold0_x_test,fold1_x_test,fold2_x_test,fold3_x_test,fold4_x_test=[],[],[],[],[]
        fold0_x_train, fold1_x_train, fold2_x_train, fold3_x_train, fold4_x_train = [], [], [], [], []
        leng1 = int(l1 * 0.2)
        leng2 = int(l2 * 0.2)
        leng3 = int(l3 * 0.2)
        leng4 = int(l4 * 0.2)

        fold0_x_test.extend(NR[0:leng1])
        fold0_x_test.extend(F[0:leng2])
        fold0_x_test.extend(T[0:leng3])
        fold0_x_test.extend(U[0:leng4])
        fold0_x_train.extend(NR[leng1:])
        fold0_x_train.extend(F[leng2:])
        fold0_x_train.extend(T[leng3:])
        fold0_x_train.extend(U[leng4:])
        fold1_x_train.extend(NR[0:leng1])
        fold1_x_train.extend(NR[leng1 * 2:])
        fold1_x_train.extend(F[0:leng2])
        fold1_x_train.extend(F[leng2 * 2:])
        fold1_x_train.extend(T[0:leng3])
        fold1_x_train.extend(T[leng3 * 2:])
        fold1_x_train.extend(U[0:leng4])
        fold1_x_train.extend(U[leng4 * 2:])
        fold1_x_test.extend(NR[leng1:leng1*2])
        fold1_x_test.extend(F[leng2:leng2*2])
        fold1_x_test.extend(T[leng3:leng3*2])
        fold1_x_test.extend(U[leng4:leng4*2])
        fold2_x_train.extend(NR[0:leng1*2])
        fold2_x_train.extend(NR[leng1*3:])
        fold2_x_train.extend(F[0:leng2*2])
        fold2_x_train.extend(F[leng2*3:])
        fold2_x_train.extend(T[0:leng3*2])
        fold2_x_train.extend(T[leng3*3:])
        fold2_x_train.extend(U[0:leng4*2])
        fold2_x_train.extend(U[leng4*3:])
        fold2_x_test.extend(NR[leng1*2:leng1*3])
        fold2_x_test.extend(F[leng2*2:leng2*3])
        fold2_x_test.extend(T[leng3*2:leng3*3])
        fold2_x_test.extend(U[leng4*2:leng4*3])
        fold3_x_train.extend(NR[0:leng1*3])
        fold3_x_train.extend(NR[leng1*4:])
        fold3_x_train.extend(F[0:leng2*3])
        fold3_x_train.extend(F[leng2*4:])
        fold3_x_train.extend(T[0:leng3*3])
        fold3_x_train.extend(T[leng3*4:])
        fold3_x_train.extend(U[0:leng4*3])
        fold3_x_train.extend(U[leng4*4:])
        fold3_x_test.extend(NR[leng1*3:leng1*4])
        fold3_x_test.extend(F[leng2*3:leng2*4])
        fold3_x_test.extend(T[leng3*3:leng3*4])
        fold3_x_test.extend(U[leng4*3:leng4*4])
        fold4_x_train.extend(NR[0:leng1*4])
        fold4_x_train.extend(NR[leng1*5:])
        fold4_x_train.extend(F[0:leng2*4])
        fold4_x_train.extend(F[leng2*5:])
        fold4_x_train.extend(T[0:leng3*4])
        fold4_x_train.extend(T[leng3*5:])
        fold4_x_train.extend(U[0:leng4*4])
        fold4_x_train.extend(U[leng4*5:])
        fold4_x_test.extend(NR[leng1*4:leng1*5])
        fold4_x_test.extend(F[leng2*4:leng2*5])
        fold4_x_test.extend(T[leng3*4:leng3*5])
        fold4_x_test.extend(U[leng4*4:leng4*5])

    if obj == "PHEME":
        # labelPath = os.path.join(cwd,"data/" +obj+"/"+ "data.label.txt")
        labelPath = os.path.join(cwd,"data/" +obj+"/"+ 'label.npy')
        # labelset_NR, labelset_R = ['non-rumours'], ['rumours']
        # print("loading PHEME label")
        # F, T = [], []
        # l1 = l2 = 0
        # labelDic = {}
        # for line in open(labelPath):
        #     line = line.rstrip()  # false	E272	656955120626880512	2	143	536	532	IJCAI	0.73
        #     label, eid = line.split('\t')[0], line.split('\t')[2]
        #     if Path('./data/PHEME/PHEMEgraph/{}.npz'.format(eid)).exists():
        #         labelDic[eid] = label.lower()
        #         if label in labelset_NR:
        #             T.append(eid)
        #             l1 += 1
        #         if labelDic[eid] in labelset_R:
        #             F.append(eid)
        #             l2 += 1
        # print(len(labelDic))
        # # print(labelDic)
        # print(l1, l2)
        # random.shuffle(F)
        # random.shuffle(T)

        print("loading PHEME label:")
        F, T = [], []
        l1 = l2 = 0
        labelDic = np.load(labelPath, allow_pickle=True).item()
        for eid in labelDic:
            if labelDic[eid] == 0:
                eid = int(eid)
                F.append(eid)
                l1 += 1
            if labelDic[str(eid)] == 1:
                eid = int(eid)
                T.append(eid)
                l2 += 1
        print(len(labelDic))
        print(l1, l2)
        random.shuffle(F)
        random.shuffle(T)

        fold0_x_test, fold1_x_test, fold2_x_test, fold3_x_test, fold4_x_test = [], [], [], [], []
        fold0_x_train, fold1_x_train, fold2_x_train, fold3_x_train, fold4_x_train = [], [], [], [], []
        leng1 = int(l1 * 0.2)
        leng2 = int(l2 * 0.2)
        fold0_x_test.extend(F[0:leng1])
        fold0_x_test.extend(T[0:leng2])
        fold0_x_train.extend(F[leng1:])
        fold0_x_train.extend(T[leng2:])
        fold1_x_train.extend(F[0:leng1])
        fold1_x_train.extend(F[leng1 * 2:])
        fold1_x_train.extend(T[0:leng2])
        fold1_x_train.extend(T[leng2 * 2:])
        fold1_x_test.extend(F[leng1:leng1 * 2])
        fold1_x_test.extend(T[leng2:leng2 * 2])
        fold2_x_train.extend(F[0:leng1 * 2])
        fold2_x_train.extend(F[leng1 * 3:])
        fold2_x_train.extend(T[0:leng2 * 2])
        fold2_x_train.extend(T[leng2 * 3:])
        fold2_x_test.extend(F[leng1 * 2:leng1 * 3])
        fold2_x_test.extend(T[leng2 * 2:leng2 * 3])
        fold3_x_train.extend(F[0:leng1 * 3])
        fold3_x_train.extend(F[leng1 * 4:])
        fold3_x_train.extend(T[0:leng2 * 3])
        fold3_x_train.extend(T[leng2 * 4:])
        fold3_x_test.extend(F[leng1 * 3:leng1 * 4])
        fold3_x_test.extend(T[leng2 * 3:leng2 * 4])
        fold4_x_train.extend(F[0:leng1 * 4])
        fold4_x_train.extend(F[leng1 * 5:])
        fold4_x_train.extend(T[0:leng2 * 4])
        fold4_x_train.extend(T[leng2 * 5:])
        fold4_x_test.extend(F[leng1 * 4:leng1 * 5])
        fold4_x_test.extend(T[leng2 * 4:leng2 * 5])

    if obj == "Weibo":
        labelPath = os.path.join(cwd,"data/Weibo/weibo_id_label.txt")
        # labelPath = '../data/Weibo/weibo_id_label.txt'
        print("loading weibo label:")
        F, T = [], []
        l1 = l2 = 0
        labelDic = {}
        for line in open(labelPath):
            line = line.rstrip()
            eid,label = line.split(' ')[0], line.split(' ')[1]
            segmentedData = np.load(os.path.join(cwd,'data/Weibo/segmentedData.npz'))
            # segmentedData = np.load('../data/Weibo/segmentedData.npz')
            num_1000 = segmentedData['num_1000']
            if eid in num_1000:
                labelDic[eid] = int(label)
                if labelDic[eid]==0:
                    F.append(eid)
                    l1 += 1
                if labelDic[eid]==1:
                    T.append(eid)
                    l2 += 1
            # labelDic[eid] = int(label)
            # if labelDic[eid] == 0:
            #     F.append(eid)
            #     l1 += 1
            # if labelDic[eid]==1:
            #     T.append(eid)
            #     l2 += 1
        print(len(labelDic))
        print(l1, l2)
        random.shuffle(F)
        random.shuffle(T)

        fold0_x_test, fold1_x_test, fold2_x_test, fold3_x_test, fold4_x_test = [], [], [], [], []
        fold0_x_train, fold1_x_train, fold2_x_train, fold3_x_train, fold4_x_train = [], [], [], [], []
        leng1 = int(l1 * 0.2)
        leng2 = int(l2 * 0.2)
        fold0_x_test.extend(F[0:leng1])
        fold0_x_test.extend(T[0:leng2])
        fold0_x_train.extend(F[leng1:])
        fold0_x_train.extend(T[leng2:])
        fold1_x_train.extend(F[0:leng1])
        fold1_x_train.extend(F[leng1 * 2:])
        fold1_x_train.extend(T[0:leng2])
        fold1_x_train.extend(T[leng2 * 2:])
        fold1_x_test.extend(F[leng1:leng1 * 2])
        fold1_x_test.extend(T[leng2:leng2 * 2])
        fold2_x_train.extend(F[0:leng1 * 2])
        fold2_x_train.extend(F[leng1 * 3:])
        fold2_x_train.extend(T[0:leng2 * 2])
        fold2_x_train.extend(T[leng2 * 3:])
        fold2_x_test.extend(F[leng1 * 2:leng1 * 3])
        fold2_x_test.extend(T[leng2 * 2:leng2 * 3])
        fold3_x_train.extend(F[0:leng1 * 3])
        fold3_x_train.extend(F[leng1 * 4:])
        fold3_x_train.extend(T[0:leng2 * 3])
        fold3_x_train.extend(T[leng2 * 4:])
        fold3_x_test.extend(F[leng1 * 3:leng1 * 4])
        fold3_x_test.extend(T[leng2 * 3:leng2 * 4])
        fold4_x_train.extend(F[0:leng1 * 4])
        fold4_x_train.extend(F[leng1 * 5:])
        fold4_x_train.extend(T[0:leng2 * 4])
        fold4_x_train.extend(T[leng2 * 5:])
        fold4_x_test.extend(F[leng1 * 4:leng1 * 5])
        fold4_x_test.extend(T[leng2 * 4:leng2 * 5])

    fold0_test = list(fold0_x_test)
    shuffle(fold0_test)
    fold0_train = list(fold0_x_train)
    shuffle(fold0_train)
    fold1_test = list(fold1_x_test)
    shuffle(fold1_test)
    fold1_train = list(fold1_x_train)
    shuffle(fold1_train)
    fold2_test = list(fold2_x_test)
    shuffle(fold2_test)
    fold2_train = list(fold2_x_train)
    shuffle(fold2_train)
    fold3_test = list(fold3_x_test)
    shuffle(fold3_test)
    fold3_train = list(fold3_x_train)
    shuffle(fold3_train)
    fold4_test = list(fold4_x_test)
    shuffle(fold4_test)
    fold4_train = list(fold4_x_train)
    shuffle(fold4_train)

    return [list(fold0_test), list(fold1_test), list(fold2_test), list(fold3_test), list(fold4_test)], \
           [list(fold0_train), list(fold1_train), list(fold2_train), list(fold3_train), list(fold4_train)]
