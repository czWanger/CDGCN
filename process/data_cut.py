#这段代码的核心功能是对树形数据进行处理，限制每个事件最多节点个数为 k，并将处理后的数据保存到新文件中。
treePath = '../data/Twitter16/data.TD_RvNN.vol_5000.txt'
k = 10
count = 0
treePath_out = '../data/Twitter16/data.TD_RvNN.vol_5000.{}.txt'.format(k)
eid_last = ''
for line in open(treePath):
    line = line.rstrip()
    eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
    max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]
    if eid == eid_last:
        count += 1
    else:
        count = 0
    if count < k:
        fo = open(treePath_out,'a')
        fo.write(line+'\n')
        fo.close()
    else:
        pass
    eid_last = eid

for line in open(treePath_out):
    line = line.rstrip()
    eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
    max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]
    print(eid, indexP, indexC, max_degree, maxL, Vec)