import os
from process.dataset_user import BiGraphDataset
import numpy as np
#获取当前路径
cwd=os.getcwd()


################################### load tree#####################################
#加载树形结构的数据
def loadTree(dataname):
    if 'Twitter' in dataname:
        #这个是树的路径
        treePath = os.path.join(cwd,'data/'+dataname+'/data.TD_RvNN.vol_5000.txt')
        print("reading twitter tree")
        treeDic = {}
        for line in open(treePath):
            # '656955120626880512	None	1	2	9	1:1 3:1 164:1 5:1 2282:1 11:1 431:1 473:1 729:1'
            #  eid, indexP, index_C, max_degree maxL Vec
            line = line.rstrip()
            eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
            max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]
            if not treeDic.__contains__(eid):
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}
        print('tree no:', len(treeDic))

    # if 'PHEME' in dataname:
    #     treePath = os.path.join(cwd, 'data/'+dataname+'/data.TD_RvNN.vol_5000.txt')
    #     print('reading PHEME tree')
    #     treeDic = {}
    #     for line in open(treePath):
    #         # '656955120626880512	None	1	2	9	1:1 3:1 164:1 5:1 2282:1 11:1 431:1 473:1 729:1'
    #         #  eid, indexP, index_C, max_degree maxL Vec
    #         line = line.rstrip()
    #         eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
    #         max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[6]
    #         if not treeDic.__contains__(eid):
    #             treeDic[eid] = {}
    #         treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}
    #     print('tree no:', len(treeDic))

    if 'PHEME' in dataname:
        eventsPath = os.path.join(cwd, 'data/' + dataname + '/PHEME_TFIDF.npy')
        print('reading pheme tree')
        eventsDic = np.load(eventsPath, allow_pickle=True).item()
        user_FeatureDic = np.load(os.path.join(cwd, 'data/' + dataname +'/PHEME_userFeature.npy'), allow_pickle=True).item()
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

    if dataname == "Weibo":
        treePath = os.path.join(cwd,'data/Weibo_all/weibotree.txt')
        print("reading Weibo tree")
        treeDic = {}
        for line in open(treePath):
            line = line.rstrip()
            eid, indexP, indexC,Vec = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2]),line.split('\t')[3]
            if not treeDic.__contains__(eid):
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP, 'vec': Vec}
        print('tree no:', len(treeDic))
    #返回树形数据的结构
    return treeDic


################################# load data ###################################
def loadData(dataname, treeDic, fold_x_train, fold_x_test, TDdroprate, BUdroprate, k):
    data_path = os.path.join(cwd,'data', dataname + 'graph')
    print("loading train set", )
    traindata_list = BiGraphDataset(fold_x_train, treeDic, k, tddroprate=TDdroprate, budroprate=BUdroprate, data_path=data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = BiGraphDataset(fold_x_test, treeDic, k, data_path=data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list


