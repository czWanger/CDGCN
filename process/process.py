import os
from process.dataset import BiGraphDataset, DNAconvDataset
import numpy as np
cwd=os.getcwd()


################################### load tree#####################################
def loadTree(dataname):
    if 'Twitter' in dataname:
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
        print('reading PHEME tree')
        eventsDic = np.load(eventsPath, allow_pickle=True).item()
        treeDic = {}
        for eid in eventsDic:
            for pid in eventsDic[eid]:
                indexP = eventsDic[eid][pid].parent
                indexC = eventsDic[eid][pid].idx
                Vec = eventsDic[eid][pid].content
                if not treeDic.__contains__(eid):
                    treeDic[eid] = {}
                treeDic[eid][indexC] = {'parent': indexP, 'vec': Vec}
        print('tree no:', len(treeDic))

    if dataname == "Weibo":
        treePath = os.path.join(cwd,'data/Weibo/weibotree.txt')
        print("reading Weibo tree")
        treeDic = {}
        segmentedData = os.path.join(cwd,'data/Weibo/segmentedData.npz')
        segmentedData = np.load(segmentedData,allow_pickle=True)
        num_1000, num_10000, num_59318 = segmentedData['num_1000'], segmentedData['num_10000'], segmentedData['num_59318']
        for line in open(treePath):
            line = line.rstrip()
            eid, indexP, indexC,Vec = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2]),line.split('\t')[3]
            if eid in num_10000 or eid in num_59318:
                pass
            else:
                if not treeDic.__contains__(eid):
                    treeDic[eid] = {}
                treeDic[eid][indexC] = {'parent': indexP, 'vec': Vec}
        print('tree no:', len(treeDic))

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


def loadDNAData(dataname, treeDic, fold_x_train, fold_x_test, TDdroprate, BUdroprate, k):
    data_path = os.path.join(cwd, 'data', dataname + 'graph')
    print("loading train set", )
    traindata_list = DNAconvDataset(fold_x_train, treeDic, tddroprate=TDdroprate, budroprate=BUdroprate, k=k,
                                    data_path=data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = DNAconvDataset(fold_x_test, treeDic, data_path=data_path, k=k)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list

