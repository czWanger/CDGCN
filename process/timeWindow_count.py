import json
import os
import re
import numpy as np
from datetime import datetime


def clean_convert(line):
    line = re.sub(r"\[", "", line)
    line = re.sub(r"\]", "", line)
    line = re.sub(r"->", ",", line)
    line = re.sub(r", ", ",", line)
    line = re.sub(r"\'", "", line)

    return line


def read(file_path):
    if not os.path.exists(file_path):
        print(file_path)
        return None

    f = open(file_path, 'r', encoding='utf-8', errors='ignore')
    con = f.read()
    f.close()
    return con


def getTimeStamp(dataset, label_path, tree_path):
    timeStampDic = {}
    if 'PHEME' in dataset:
        labelSet = np.load(label_path, allow_pickle=True).item()
        for eid in labelSet.keys():
            timeStampDic[eid] = []
            if labelSet[eid] == 0:
                label = 'non-rumors'
            elif labelSet[eid] == 1:
                label = 'rumors'
            tree = tree_path + label + '/{}/'.format(eid)
            sourcePost = read(tree + 'source-tweet/' + '{}.json'.format(eid))
            if sourcePost is None:
                pass
            else:
                content = json.loads(sourcePost)
                root_time = datetime.strptime(content['created_at'], '%a %b %d %H:%M:%S %z %Y').timestamp()
                print(root_time)
                timeStampDic[eid].append(0)
            for _, _, reactions in os.walk(tree + 'reactions/'):
                for reaction in reactions:
                    rePost = read(tree + 'reactions/' + reaction)
                    if rePost is None:
                        pass
                    else:
                        content = json.loads(rePost)
                        creatTime = datetime.strptime(content['created_at'], '%a %b %d %H:%M:%S %z %Y').timestamp()
                        timeStampDic[eid].append((creatTime - root_time) / 60)

    if 'Twitter' in dataset:
        for event in open(label_path):
            event = event.rstrip()
            label, eid = event.split('\t')[0], event.split('\t')[2]
            tree = tree_path + '{}.txt'.format(eid)
            timeStampDic[eid] = []
            for line in open(tree):
                line = clean_convert(line)
                timeStamp = float(line.split(',')[5])
                timeStampDic[eid].append(timeStamp)

    if 'Weibo' in dataset:
        segmentedData = np.load('../data/Weibo/segmentedData.npz')
        num_1000 = segmentedData['num_1000']
        for eid in num_1000:
            tree = read(tree_path + '{}.json'.format(eid))
            if tree is None:
                pass
            else:
                content = json.loads(tree)
                root_time = content[0]['t']
                timeStampDic[eid] = []
                for post in content:
                    # 特征信息转换为值
                    timeStampDic[eid].append((post['t'] - root_time) / 60)

    return timeStampDic


def getLen(threshold, timeStampDic):
    lenDic = {}
    for eid in timeStampDic:
        lenDic[eid] = []
        for deadline in threshold:
            lenDic[eid].append(len([i for i in timeStampDic[eid] if i <= deadline]))

    return lenDic


if __name__ == '__main__':
    dataset = 'PHEME'  # Twitter15 Twitter16 Weibo PHEME
    label_path, tree_path = None, None
    if 'Twitter' in dataset:
        label_path = '../data/{}/{}_label_All.txt'.format(dataset, dataset)
        tree_path = '../data/{}/tree/'.format(dataset)
    elif 'Weibo' in dataset:
        label_path = '../data/{}/{}_label_All.txt'.format(dataset, dataset)
        tree_path = '../data/Weibo/Weibo/'
    elif 'PHEME' in dataset:
        label_path = '../data/PHEME/label.npy'
        tree_path = '../data/PHEME/pheme-rnr-dataset/'
    timeStampDic = getTimeStamp(dataset, label_path, tree_path)  # eid:[t0, t1, ..., tn]
    threshold = [600, 1200, 1800, 2400, 3000, 3600]
    lenDic = getLen(threshold, timeStampDic)  # eid: [10min len, 20min len, ... , 60min len]
    save_path = '../data/{}/timeWindowLen.npy'.format(dataset)
    np.save(save_path, lenDic)
    print(lenDic)
