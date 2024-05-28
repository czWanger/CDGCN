import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data
import copy

class BiGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic, k, lower=2, upper=100000, tddroprate=0, budroprate=0,
                 data_path=os.path.join('..', '..', 'data', 'Twitter15graph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate
        self.k = k

    def __len__(self):
        return len(self.fold_x)

    def TD2BU(self, edgeindex):
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow, bucol]
        return bunew_edgeindex

    def __getitem__(self, index):
        id = self.fold_x[index]
        data = np.load('./data/Twitter15/Twitter15graph/' + str(id) + '.npz', allow_pickle=True)
        # threshold = np.load('./data/Weibo/timeWindowLen.npy', allow_pickle=True)

        # data = np.load("./data/PHEME/PHEMEgraph/" + str(id) + '.npz', allow_pickle=True)
        # threshold = np.load('./data/PHEME/timeWindowLen.npy', allow_pickle=True)

        # data = np.load('./data/Twitter16/' + str(id) + '.npz', allow_pickle=True)
        # threshold = np.load('./data/Twitter15/timeWindowLen.npy', allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex

        bunew_edgeindex = self.TD2BU(edgeindex)
        # budiv_edgeindex = self.TD2BU(data['division'])

        x = torch.tensor(data['x'], dtype=torch.float32)
        # # time
        # if threshold.__contains__(str(id)):
        #     pri_index = threshold[self.k]
        # else:
        #     pri_index = int(x.size(0) * (self.k + 1)*0.15)
        # # len
        # pri_index = (self.k+1)*10
        # x[pri_index:, :] = 0
        return Data(x=x,
                    edge_index=torch.LongTensor(new_edgeindex),
                    BU_edge_index=torch.LongTensor(bunew_edgeindex),
                    y=torch.LongTensor([int(data['y'])]),
                    root=torch.LongTensor(data['root']),
                    # TDdiv_edge_index=torch.LongTensor(data['division']),
                    # BUdiv_edge_index=torch.LongTensor(budiv_edgeindex),
                    rootindex=torch.LongTensor([int(data['rootindex'])]))



class DNAconvDataset(Dataset):
    def __init__(self, fold_x, treeDic, lower=2, upper=100000, tddroprate=0, budroprate=0, k=0.5,
                 data_path=os.path.join('..', '..', 'data', 'Weibograph')):
        self.fold_x = list(
            filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate
        self.k = k

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id = self.fold_x[index]
        data = np.load('./data/Twitter15/louvaindiv_cos' + id + '.npz', allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex

        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow, bucol]

        x = torch.tensor(data['x'], dtype=torch.float32)
        x1 = copy.deepcopy(x)
        pri_index = int(x.size(0) * self.k)
        x1[pri_index:, :] = 0
        x = torch.stack((x, x1), 1)
        return Data(x=x, edge_index=torch.LongTensor(new_edgeindex), BU_edge_index=torch.LongTensor(bunew_edgeindex),
                    y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
                    rootindex=torch.LongTensor([int(data['rootindex'])]), Division_edge_index=torch.LongTensor(data['division']))
