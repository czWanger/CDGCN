import json
import os
import re
import time
import traceback

import numpy as np
from GN import *
from joblib import Parallel, delayed
from tqdm import tqdm
from communities.algorithms import louvain_method
import jieba
from gensim import corpora, models

# 文本内容清理和分词
def clean_str_cut(str):
    """
    除SST外的所有数据集的标记化/字符串清洗。
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # str = re.sub(r'@\w+\s', '', str)
    str = re.sub(r"\'s", " \'s", str)
    str = re.sub(r"\'ve", " \'ve", str)
    str = re.sub(r"n\'t", " n\'t", str)
    str = re.sub(r"\'re", " \'re", str)
    str = re.sub(r"\'d", " \'d", str)
    str = re.sub(r"\'ll", " \'ll", str)
    str = re.sub(r",", " , ", str)
    str = re.sub(r"!", " ! ", str)
    str = re.sub(r"\(", " \( ", str)
    str = re.sub(r"\)", " \) ", str)
    str = re.sub(r"\[", " \[ ", str)
    str = re.sub(r"\]", " \] ", str)
    str = re.sub(r"\?", " \? ", str)
    str = re.sub(r"\s{2,}", " ", str)

    return str

def read(file_path):
    if not os.path.exists(file_path):
        return None

    f = open(file_path, 'r', encoding='utf-8', errors='ignore')
    con = f.read()
    f.close()
    return con

class Post(object):
    def __init__(self, eid=None, idx=None):
        self.eid = eid
        self.idx = idx
        self.content = []
        self.parent = None
        self.user = []


class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        self.parent = None
        self.user = []


def constructMat(tree, id):
    index2node = {}
    for i in tree:
        node = Node_tweet(idx=i)
        index2node[i] = node
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']  # 父节点索引
        nodeC = index2node[indexC]  # 孩子节点内容
        nodeC.word = tree[j]['vec']
        nodeC.user = tree[j]['user']
        if not indexP is None:
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        else:
            rootindex = indexC - 1
            root_index = nodeC.index
            root_word = nodeC.word
    rootfeat = np.zeros([1, 5000])
    if len(root_index) > 0:
        rootfeat[0, np.array(root_index)] = np.array(root_word)
    row = []
    col = []
    x_x = []
    user_matrix = []
    for index_i in range(len(index2node)):
        for index_j in range(len(index2node)):
            try:
                if index2node[index_i + 1].children != None and index2node[index_j + 1] in index2node[index_i + 1].children:
                    row.append(index_i)
                    col.append(index_j)
            except Exception as e:
                print(id)
                traceback.print_exc()
        x_x.append(index2node[index_i + 1].word)
        user_matrix.append(index2node[index_i + 1].user)
    edgematrix = [row, col]

    return x_x, edgematrix, rootfeat, rootindex,user_matrix


def louvainDiv(tree):
    row = tree[0]
    col = tree[1]
    # networkX构图
    G = load_graph(row, col)
    A = np.array(nx.adjacency_matrix(G).todense())
    communities, _ = louvain_method(A)
    new_row = []
    new_col = []
    for group in communities:
        nodes = list(group)
        subG = nx.subgraph(G, nodes)
        for e in subG.edges():
            new_row.append(e[0])
            new_col.append(e[1])
    return [new_row, new_col]

def format_time(str):
    """
    :param str: 如"Sat Aug 09 23:28:03 +0000 2014"
    :return: 自1970年1月1日以来持续时间的秒数
    """
    str = re.sub(r'\+\w*', '', str)
    time_structure = time.strptime(str, '%a %b %d %H:%M:%S %Y')
    ts = time.mktime(time_structure)
    return ts


def getContent(str):
    # 停用词表
    stop_words = ['的', '了', '是', '在', '我', '有', '和', '就', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
                  '你', '会', '着', '没有', '看', '好']
    words = [word for word in jieba.cut(str) if word not in stop_words]
    # 定义词表
    dictionary = corpora.Dictionary()
    # 将分词后的文本添加到词表中
    dictionary.add_documents([words])
    # 将分词后的文本列表合并成字符串列表
    corpus = ''.join(words)
    # 加载预训练的TF-IDF模型
    tfidf_model = models.TfidfModel.load('tfidf_model')
    # 将分词后的文本转换为TF-IDF向量表示
    corpus = [tfidf_model[dictionary.doc2bow(words)] for words in [words]]
    print(corpus)


def getPostDic(file, eid):
    jsonFile = read(file)
    weibo = json.loads(jsonFile)
    posts = {}
    order_dic = {}
    idx = 1
    rootMid = weibo[0]['mid']
    for postDic in weibo:
        pid = postDic['id']
        post = Post(eid=eid, idx=idx)
        post.parent = postDic['parent']
        post.content = getContent(postDic['text'])
    # 返回pid的dic
    pass


def main():

    labelPath = "../data/Weibo_all/weibo_id_label.txt"
    print("loading weibo label:")
    eids, y = [], []
    l1 = l2 = 0
    labelDic = {}
    for line in open(labelPath):
        line = line.rstrip()
        eid, label = line.split(' ')[0], line.split(' ')[1]
        labelDic[eid] = int(label)
        y.append(labelDic[eid])
        eids.append(eid)
        if labelDic[eid] == 0:
            l1 += 1
        if labelDic[eid] == 1:
            l2 += 1

    print(len(labelDic), len(eids), len(y))
    print(l1, l2)

    eventsDic = {}
    jsonDir = '../data/Weibo/Weibo/'
    for eid in eids:
        eventsDic[eid] = getPostDic(jsonDir+str(eid)+'.json', eid)


    eventsPath = '../data/PHEME/PHEME_TFIDF.npy'
    print('reading pheme tree')
    eventsDic = np.load(eventsPath, allow_pickle=True).item()
    user_FeatureDic = np.load('../data/PHEME/PHEME_userFeature.npy',allow_pickle=True).item()
    treeDic = {}
    for eid in eventsDic:
        for pid in eventsDic[eid]:
            indexP = eventsDic[eid][pid].parent
            indexC = eventsDic[eid][pid].idx
            Vec = eventsDic[eid][pid].content
            user = user_FeatureDic[eid][pid]
            if not treeDic.__contains__(eid):
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP, 'vec': Vec, 'user': user}
    print('tree no:', len(treeDic))

    labelPath = '../data/PHEME/label.npy'
    print('loading tree label')
    labelDic = np.load(labelPath, allow_pickle=True).item()
    event, y = [], []
    l1 = l2 = 0
    for eid in labelDic:
        event.append(eid)
        label = labelDic[eid]
        if label == 0:
            l1 += 1
        if label == 1:
            l2 += 1
    print(len(labelDic))
    print(l1, l2)

    def loadEid(event, id, y):
        if event is None:
            print('eid is none')
            return None
        if len(event)<2:
            return None
        if len(event)>1:
            x_x, tree, rootfeat, rootindex,userFeature = constructMat(event, id)
            userMapDic = np.load('../data/PHEME/PHEME_userMap.npy',allow_pickle=True).item()
            userMap = userMapDic[int(id)][2]
            rootfeat, tree, x_x, rootindex, y = np.array(rootfeat), np.array(tree), np.array(x_x), np.array(
                rootindex), np.array(y)
            np.savez('../data/PHEME/PHEMEgraph_user/' + id + '.npz', x=x_x, root=rootfeat, edgeindex=tree,
                     rootindex=rootindex, y=y,userMap = userMap,userFeature = userFeature)
            return None

    print('loading dataset')
    Parallel(n_jobs=5, backend='threading')(
        delayed(loadEid)(treeDic[int(eid)] if int(eid) in treeDic else None, eid, labelDic[eid]) for eid in tqdm(event)
    )
    return


if __name__ == '__main__':
    main()
