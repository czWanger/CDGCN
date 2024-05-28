'''
这段代码的功能是：

从指定路径加载PHEME数据集的标签信息，标签信息保存在Numpy格式的文件 label.npy 中。
将标签分为两类：'rumours'（谣言）和 'non-rumours'（非谣言）。
从标签信息中获取每个事件的标签，并根据标签将事件ID分别添加到 'rumours' 和 'non-rumours' 的列表中。
随机打乱 'rumours' 和 'non-rumours' 列表。
将数据集划分为训练集和测试集，其中80%的数据作为训练集，20%的数据作为测试集。
将训练集和测试集中的事件ID分别写入到 data.train.txt 和 data.test.txt 文件中。
整体来说，这段代码是在准备PHEME数据集的训练集和测试集，以便后续的模型训练和评估。

'''

import random
from pathlib import Path
from random import shuffle
import os
import numpy as np

cwd=os.getcwd()

labelPath = '../data/PHEME/label.npy'
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
        F.append(eid)
        l1 += 1
    if labelDic[eid] == 1:
        T.append(eid)
        l2 += 1
print(len(labelDic))
print(l1, l2)
random.shuffle(F)
random.shuffle(T)

x_test = []

x_train = []
leng1 = int(l1 * 0.8)
leng2 = int(l2 * 0.8)
x_train.extend(F[0:leng1])
x_train.extend(T[0:leng2])
x_test.extend(F[leng1:])
x_test.extend(T[leng2:])

with open('../data/PHEME/data.train.txt', 'w') as file:
    for element in x_train:
        file.write(element + '\n')

with open('../data/PHEME/data.test.txt', 'w') as file:
    for element in x_test:
        file.write(element + '\n')





