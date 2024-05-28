import sys, os, fnmatch, datetime, time
import sqlite3
import numpy as np
from scipy import misc
from sklearn import preprocessing
# from dgl import load_graphs
from sklearn.model_selection import StratifiedKFold


def read(file_path):
    if not os.path.exists(file_path):
        return None

    f = open(file_path, 'r', encoding='utf-8', errors='ignore')
    con = f.read()
    f.close()
    return con


def read_lines(file_path):
    if not os.path.exists(file_path):
        print('file not exists')
        return None
    f = open(file_path, "r", encoding='utf-8', errors='ignore')
    ls = f.readlines()
    ls2 = []
    for line in ls:
        if line.strip():
            ls2.append(line.strip())
    f.close()
    return ls2


def save(var, file_path):
    f = open(file_path, "w", encoding='utf-8', errors='ignore')
    f.write(str(var))
    f.close()


def ls(folder, pattern='*'):
    fs = []
    for root, dir, files in os.walk(folder):
        for f in fnmatch.filter(files, pattern):
            fs.append(f)
    return fs


def str_to_timestamp(string):
    structured_time = time.strptime(string, "%a %b %d %H:%M:%S +0000 %Y")
    timestamp = time.mktime(structured_time)
    return timestamp


def db_connect(db_file):
    try:
        dbc = sqlite3.connect(db_file, timeout=10)
        return dbc
    except sqlite3.Error as e:
        print("Error {}:".format(e.args[0]))
        return None


def normalize(x, positions):
    num_columns = x.shape[1]
    for i in range(num_columns):
        if i in positions:
            x[:, i:i + 1] = np.copy(preprocessing.robust_scale(x[:, i:i + 1]))
    return x


def load_img(path, length, height):
    img = misc.imread(path)
    img = misc.imresize(img, [height, length])
    img = img[:, :, 0:3]
    return img
#
# def load_data(dataset):
#     print('loading data...')
#     g_list, labels = load_graphs("../process/graphData.bin")
#     return g_list, labels



def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 5, "fold_idx must be from 0 to 4."
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list