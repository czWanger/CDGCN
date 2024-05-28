import traceback

import numpy as np
from GN import *
from joblib import Parallel, delayed
from tqdm import tqdm
from communities.algorithms import louvain_method


class Post(object):
    def __init__(self, eid=None, idx=None):
        self.eid = eid
        self.idx = idx
        self.content = []
        self.parent = None
        self.timespan = None
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
        # nodeC.user = tree[j]['user']
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


def main():
    eventsPath = '../data/PHEME/PHEME_TFIDF.npy'
    print('reading pheme tree')
    eventsDic = np.load(eventsPath, allow_pickle=True).item()
    # user_FeatureDic = np.load('../data/PHEME/PHEME_userFeature.npy',allow_pickle=True).item()
    treeDic = {}
    for eid in eventsDic:
        for pid in eventsDic[eid]:
            indexP = eventsDic[eid][pid].parent
            indexC = eventsDic[eid][pid].idx
            Vec = eventsDic[eid][pid].content
            # user = user_FeatureDic[eid][pid]
            if not treeDic.__contains__(eid):
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP, 'vec': Vec}
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
            np.savez('../data/PHEME/PHEMEgraph/' + id + '.npz', x=x_x, root=rootfeat, edgeindex=tree,
                     rootindex=rootindex, y=y,userMap = userMap,userFeature = userFeature)
            return None

    print('loading dataset')
    Parallel(n_jobs=5, backend='threading')(
        delayed(loadEid)(treeDic[int(eid)] if int(eid) in treeDic else None, eid, labelDic[eid]) for eid in tqdm(event)
    )
    return


if __name__ == '__main__':
    main()
