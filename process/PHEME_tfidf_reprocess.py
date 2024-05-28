import numpy as np
import copy
import os
import re
import time


class Post(object):
    def __init__(self, eid=None, idx=None):
        self.eid = eid
        self.idx = idx
        self.content = []
        self.parent = None
        self.timespan = None


def format_time(str):
    """
    :param str: 如"Sat Aug 09 23:28:03 +0000 2014"
    :return: 自1970年1月1日以来持续时间的秒数
    """
    str = re.sub(r'\+\w*', '', str)
    time_structure = time.strptime(str, '%a %b %d %H:%M:%S %Y')
    ts = time.mktime(time_structure)
    return ts


def get_events_dic(source_path, label_path):
    # 事件id-标签
    label_dic = np.load(label_path, allow_pickle=True).item()
    # 推文id-句向量
    contents_dic = np.load('../data/PHEME/PHEME_tfidf_5000.npy', allow_pickle=True).item()
    # 推文id-推文各项属性的词典-推文属性通过属性名称访问
    posts_dic = np.load('../data/PHEME/posts_dic.npy', allow_pickle=True).item()
    # 空内容帖子id列表
    empty_post = np.load('../data/PHEME/empty_id.npy', allow_pickle=True).tolist()
    # 写入路径
    labels = copy.deepcopy(list(label_dic.keys()))
    events = {}
    for eid in labels:
        idx = 1  # 推文在事件中的索引
        print(eid)
        if eid == '544457324211879937':
            pass
        else:
            # 当前事件推文列表
            posts = {}
            order_dic = {}
            if label_dic[eid] == 0:
                label = 'non-rumors'
            else:
                label = 'rumors'
            # 转推
            for root, dirs, files in os.walk(source_path+label+'/'+eid+'/reactions/', topdown=False):
                for pid in files:
                    print('pid', pid)
                    pid = int(re.sub(r"\.\w*", "", str(pid)))
                    # 内容不为空
                    if empty_post.count(pid) == 0:
                        # 推文id-时间戳
                        order_dic[pid] = format_time(posts_dic[pid]['created_at'])
            eid = int(eid)
            # 源推
            order_dic[eid] = format_time(posts_dic[eid]['created_at'])
            # 按时间戳从小到大排序
            ordered = sorted(order_dic.items(), key=lambda x: x[1], reverse=False)
            for tpl in ordered:
                pid = (list(tpl))[0]
                post = Post(eid=eid, idx=idx)
                posts[pid] = post
                posts[pid].content = contents_dic[pid]
                if pid == eid:
                    pass
                else:
                    if posts.__contains__((posts_dic[pid]['in_reply_to_status_id'])):
                        posts[pid].parent = posts[posts_dic[pid]['in_reply_to_status_id']].idx
                    else:
                    # 若上一个为空就将父节点设为根节点
                        print(pid, eid)
                        posts[pid].parent = posts[eid].idx
                idx += 1
            events[eid] = posts
    np.save('PHEME_TFIDF.npy', events)

if __name__ == '__main__':
    source_path = '../data/PHEME/pheme-rnr-dataset/'
    label_path = '../data/PHEME/label.npy'
    get_events_dic(source_path, label_path)