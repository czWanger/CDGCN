import copy
import json
import os
import re
import time

import jieba
import numpy as np
import utils
import gensim.downloader

non_rumors_path = '../source/pheme-rnr-dataset/non-rumors'
rumors_path = '../source/pheme-rnr-dataset/rumors'


class Post(object):
    def __init__(self, eid=None, idx=None):
        self.eid = eid
        self.idx = idx
        self.content = []
        self.parent = None
        self.timespan = None


def get_eids(file_path):
    """
    :param file_path: 数据集路径
    :return: 路径下所有事件id
    """
    for root, dirs, files in os.walk(file_path, topdown=False):
        pass
    return dirs


def contract_eids_labels():
    """
    :param label_path: 生成字典存放地址
    :return: 事件-标签字典
    """
    lable_dic = {}
    for eid in get_eids(non_rumors_path):
        lable_dic[eid] = 0
    for eid in get_eids(rumors_path):
        lable_dic[eid] = 1
    # return lable_dic
    np.save(label_path, lable_dic)
    # read_dic = np.load(label_path, allow_pickle=True)
    # print(read_dic)


def create_posts_dic():
    """
    :return:事件id-标签字典
    """
    posts_dic = {}
    for root, dirs, files in os.walk(non_rumors_path, topdown=False):
        for file in files:
            json_file = utils.read(os.path.join(root, file))
            # json.loads函数将字符串转化为字典
            post_dic = json.loads(json_file)
            posts_dic[post_dic['id']] = post_dic
    for root, dirs, files in os.walk(rumors_path, topdown=False):
        for file in files:
            json_file = utils.read(os.path.join(root, file))
            post_dic = json.loads(json_file)
            posts_dic[post_dic['id']] = post_dic
    np.save('posts_dic.npy', posts_dic)


# 文本内容清理和分词
def clean_str_cut(content, filename):
    """
    除SST外的所有数据集的标记化/字符串清洗。
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    if filename != "Weibo":
        content = re.sub(r'@\w*:', '', content)
        content = re.sub(r'@\w*', '', content)
        content = re.sub(r'#\w*', '', content)
        content = re.sub(r"[^A-Za-z0-9(),!?#@\'\`]", " ", content)
        content = re.sub(r"\'s", " \'s", content)
        content = re.sub(r"\'ve", " \'ve", content)
        content = re.sub(r"n\'t", " n\'t", content)
        content = re.sub(r"\'re", " \'re", content)
        content = re.sub(r"\'d", " \'d", content)
        content = re.sub(r"\'ll", " \'ll", content)
        content = re.sub(r",", " ", content)
        content = re.sub(r"\\", "", content)

    content = re.sub(r",", " , ", content)
    content = re.sub(r"!", " ! ", content)
    content = re.sub(r"\(", " \( ", content)
    content = re.sub(r"\)", " \) ", content)
    content = re.sub(r"\[", " \[ ", content)
    content = re.sub(r"\]", " \] ", content)
    content = re.sub(r"\?", " \? ", content)
    content = re.sub(r"\s{2,}", " ", content)

    words = list(jieba.cut(content.strip().lower())) if filename == "Weibo" else content.strip().lower().split()

    return words


def get_word2vec():
    """
    生成每条post的字典
    """
    contents_dic = {}
    count = 0
    for post_dic in np.load('posts_dic.npy', allow_pickle=True).item().values():
        count += 1
        contents_dic[post_dic['id']] = clean_str_cut(post_dic['text'], 'PHEME')
        print(count)
    np.save('./contents_dic.npy', contents_dic)


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
    contents_dic = np.load('post_sentence2vec.npy', allow_pickle=True).item()
    # 推文id-推文各项属性的词典-推文属性通过属性名称访问
    posts_dic = np.load('posts_dic.npy', allow_pickle=True).item()
    # 空内容帖子id列表
    empty_post = np.load('empty_id.npy', allow_pickle=True).tolist()
    # 写入路径
    save_f = './PHEME_TD.vol_300.txt'
    labels = copy.deepcopy(list(label_dic.keys()))
    events = {}
    for eid in labels:
        print(eid)
        # 当前事件推文列表
        posts = {}
        order_dic = {}
        # 推文在事件中的索引
        idx = 1
        if label_dic[eid] == 0:
            label = 'non-rumors'
        else:
            label = 'rumors'
        # 转推
        for root, dirs, files in os.walk(os.path.join(source_path, label, eid, 'reactions'), topdown=False):
            pass
        for pid in files:
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
        # 源推时间戳
        src_timestamp = list(ordered[0])[1]
        # 传播总时长
        total_time = list(ordered[len(list(ordered)) - 1])[1] - src_timestamp
        # if total_time == 0:
        #     print(eid)
        #     label_dic.pop(str(eid))
        for tpl in ordered:
            item = list(tpl)
            post = Post(eid=eid, idx=idx)
            posts[item[0]] = post
            posts[item[0]].content = contents_dic[item[0]]
            posts[item[0]].timespan = (item[1] - src_timestamp) / total_time
            if item[0] == eid:
                pass
            else:
                if posts.__contains__(posts_dic[item[0]]['in_reply_to_status_id']):
                    posts[item[0]].parent = posts[posts_dic[item[0]]['in_reply_to_status_id']].idx
                else:
                    # 若上一个为空就将父节点设为根节点
                    posts[item[0]].parent = 1
            idx += 1
            with open(save_f, 'a', encoding='utf8') as f:
                f.write(str(posts[item[0]].eid) + '\t' + str(posts[item[0]].parent) + '\t' + str(posts[item[0]].idx) + '\t' +
                        str(posts[item[0]].timespan) + '\t')
                for i in posts[item[0]].content:
                    f.write(str(i)+' ')
                f.write('\n')
        events[eid] = posts
    np.save('PHEME.npy', events)

source_path = '../source/pheme-rnr-dataset'
label_path = source_path + '/label.npy'
# create_posts_dic()
# contract_eids_labels()
# get_word2vec()
get_events_dic(source_path, label_path)
