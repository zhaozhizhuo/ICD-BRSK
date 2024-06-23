import re
from random import sample
import h5py
import gensim
import torch
import os
from torch.utils.data import Dataset
from constant import DATA_DIR, MIMIC_2_DIR, MIMIC_3_DIR
import sys
import pandas as pd
import numpy as np
import math
import csv
from collections import defaultdict
import warnings
import json, ujson
warnings.filterwarnings('ignore', category=FutureWarning)

maxInt = sys.maxsize

while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


def create_main_code(ind2c):
    mc = list(set([c.split('.')[0] for c in set(ind2c.values())]))
    mc.sort()   #获得上级标签
    ind2mc = {ind:mc for ind, mc in enumerate(mc)}
    mc2ind = {mc:ind for ind, mc in ind2mc.items()}
    return ind2mc, mc2ind   #构建上级标签的索引


class MimicFullDataset(Dataset):
    def __init__(self, version, mode, vocab_path, truncate_length,
                 label_truncate_length=30, term_count=1, sort_method='max', one_test='text'):
        self.version = version
        self.mode = mode
        self.df = []

        if version == 'mimic2':
            raise NotImplementedError
        if version in ['mimic3', 'mimic3-50','mimic3-50l']:
            self.path = os.path.join(MIMIC_3_DIR, f"{version}_{mode}.json")

        if version in ['mimic3']:
            self.train_path = os.path.join(MIMIC_3_DIR, "train_full.csv")
            self.path = os.path.join(MIMIC_3_DIR, f"{mode}_full.csv")
        if version in ['mimic3-50']:
            self.train_path = os.path.join(MIMIC_3_DIR, "train_50.csv")
        if version in ['mimic3-50l']:
            self.train_path = os.path.join(MIMIC_3_DIR, "train_50l.csv")

        if version == 'mimic3':
            print(self.path)
            with open(self.path, 'r') as file:
                data = csv.reader(file)
                next(data)
                for line in data:
                    data_i = {}
                    data_i['subject_id'] = line[0]
                    data_i['hadm_id'] = line[1]
                    data_i['LABELS'] = line[2]
                    data_i['TEXT'] = line[3]
                    data_i['Addition'] = len(line[3].split())
                    self.df.append(data_i)
        else:
            if version == 'one_test':
                data_i = {}
                data_i['subject_id'] = 0
                data_i['hadm_id'] = 0
                data_i['LABELS'] = 0
                data_i['TEXT'] = one_test
                data_i['Addition'] = len(data_i['TEXT'].split())
                self.train_path = os.path.join(MIMIC_3_DIR, "train_50.csv")
                self.df.append(data_i)
            else:
                with open(self.path, "r") as f:
                    self.df = json.load(f) #加载数据集

        self.vocab_path = vocab_path
        self.word2id, self.id2word = load_vocab(self.vocab_path)

        self.truncate_length = truncate_length

        self.ind2c, _ = load_full_codes(self.train_path, version=version) #_表示的是code对应的描述，indc代表索引
        # self.part_icd_codes = list(self.ind2c.values())
        self.c2ind = {c: ind for ind, c in self.ind2c.items()}
        self.code_count = len(self.ind2c)
        if mode == "train":
            print(f'Code count: {self.code_count}')

        self.ind2mc, self.mc2ind = create_main_code(self.ind2c)
        self.main_code_count = len(self.ind2mc)
        if mode == "train":
            print(f'Main code count: {self.main_code_count}')

        from nltk.tokenize import RegexpTokenizer
        self.tokenizer = RegexpTokenizer(r'\w+')

        self.len = len(self.df) #查看训练集有多少条数据

        self.label_truncate_length = label_truncate_length  #标签截断长度，猜测应该是一篇文档最多对应的标签数量
        self.term_count = term_count
        self.sort_method = sort_method
        if self.mode == "train":
            self.prepare_label_feature(self.label_truncate_length)

    def check(self, word):
        for ch in word:
            if 'a' <= ch <= 'z' or 'A' <= ch <= "Z":
                return True
        return False

    def __len__(self):
        return self.len

    def gettext(self, index):
        return self.df[index]['TEXT']

    def split(self, text):
        sp = re.sub(r'\n\n+|  +', '\t', text.strip()).replace("\n",
                                                              " ").replace("!", "\t").replace("?", "\t").replace(".", "\t")
        return [s.strip() for s in sp.split("\t") if s.strip()]

    def tokenize(self, text):
        texts = self.split(text)
        all_text = []
        split_text = []
        sentence_index = []
        word_count = 0
        for note in texts:
            now_text = [w.lower() for w in self.tokenizer.tokenize(
                note) if not w.isnumeric()]
            if now_text:
                all_text.extend(now_text)
                split_text.append(now_text)
                word_count += len(now_text)
                sentence_index.append(word_count)
        return all_text, sentence_index, split_text

    def __get_text_label__(self, index):
        text = self.gettext(index)
        label = str(self.df[index]['LABELS']).split(';')
        return text, label

    def pad(self, l, pad_token_length, pad_token_id, reverse=False):
        if len(l) > pad_token_length:
            if not reverse:
                return l[0:pad_token_length]
            return l[-pad_token_length:]
        if not reverse:
            return l + [pad_token_id] * (pad_token_length - len(l))
        return [pad_token_id] * (pad_token_length - len(l)) + l

    def text2feature(self, text, t, truncate_length=None):
        if truncate_length is None:
            truncate_length = self.truncate_length
        #alltext代表的是所有单词组成的列表，sentence_index代表的是句子的程度，split_text代表的是有多少个同义词组
        all_text, sentence_index, split_text = self.tokenize(text)
        if t:
            def is_folder_empty(folder_path):
                return len(os.listdir(folder_path)) == 0
            # 示例用法
            folder_path = './output/text/'
            data_text = {}
            data_text['text'] = all_text
            if is_folder_empty(folder_path):
                json_file = './output/text/1.json'
                with open(json_file, 'w') as outfile:
                    json.dump(data_text, outfile)
                # np.save('./output/tensor/1.json', data_text)
                # print("文件夹为空")
            else:
                file_names = os.listdir(folder_path)
                sorted_file_names = sorted(file_names, key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))
                file_names = sorted_file_names[-1].split('.')[0]
                json_file = './output/text/{}.json'.format(int(file_names) + 1)
                with open(json_file, 'w') as outfile:
                    json.dump(data_text, outfile)

        input_word = [self.word2id.get(
            w, self.word2id["**UNK**"]) for w in all_text]  #将文字转换为对应idx
        word_mask = [1] * len(input_word)   #标记mask长度，将文本的长度标为1
        input_word = self.pad(
            input_word, truncate_length, self.word2id["**PAD**"])   #将文本长度填充为30
        word_mask = self.pad(word_mask, truncate_length, 0) #mask也进行填充
        word_sent = self.pad(sentence_index, truncate_length, -1)   #将句子的长度也进行填充
        return input_word, word_mask, word_sent

    def process(self, text, label):
        input_word, word_mask, word_sent = self.text2feature(text,t=True)

        binary_label = [0] * self.code_count
        for l in label:
            if l in self.c2ind:
                binary_label[self.c2ind[l]] = 1

        main_label = [0] * self.main_code_count
        for l in label:
            if l.split('.')[0] in self.mc2ind:
                main_label[self.mc2ind[l.split('.')[0]]] = 1

        return input_word, word_mask, word_sent, \
               binary_label, main_label

    def __getitem__(self, index):
        text, label = self.__get_text_label__(index)
        processed = self.process(text, label)
        return processed

    def extract_label_desc(self, ind2c):
        if not hasattr(self, 'desc_dict'):
            self.desc_dict = load_code_descriptions()

        desc_list = []
        for i in ind2c:
            code = ind2c[i]
            if not code in self.desc_dict:
                print(f'Not find desc of {code}')
            desc = self.desc_dict.get(code, code)   #如果在字典中存在code则输出code对应的值，否则就输出code
            desc_list.append(desc)
        return desc_list    #获得当前所有标签对应的描述文档

    def process_label(self, ind2c, truncate_length, term_count=1, method='max'):
        desc_list = self.extract_label_desc(ind2c)  #未加入同义词的代码描述
        if term_count == 1:
            c_desc_list = desc_list
        else: #term_count的意义是选取多少个同义词
            c_desc_list = []
            with open(f'./embedding/icd_mimic3_{method}_sort.json', 'r') as f:
                icd_syn = ujson.load(f)
            for i in ind2c:
                code = ind2c[i]
                tmp_desc = [desc_list[i]]
                new_terms = icd_syn.get(code, [])
                if len(new_terms) >= term_count - 1:
                    tmp_desc.extend(new_terms[0:term_count - 1])
                else:
                    tmp_desc.extend(new_terms)
                    repeat_count = int (term_count / len(tmp_desc)) + 1
                    tmp_desc = (tmp_desc * repeat_count)[0:term_count]
                if i < 5:
                    print(code, tmp_desc,'同义词没有五个')
                c_desc_list.extend(tmp_desc)    #所有出现代码的同义词

        c_input_word = []   #对于同义词文本来说，将文本转换为索引格式，之后填充或者截断为30长度
        c_word_mask = []    #同理对于上述c_input_word来说，将真实存在的变为1，其余填充为0
        c_word_sent = []    #将句子的长度填充为对应30，第一个为句子的长度，其余为-1

        for i, desc in enumerate(c_desc_list):
            input_word, word_mask, word_sent = self.text2feature(desc, truncate_length=truncate_length, t=False)
            c_input_word.append(input_word)
            c_word_mask.append(word_mask)
            c_word_sent.append(word_sent)
        #最终这三者会存储所有的同义词的编码（list的长度为 （编码数量*取同义词数量））
        return c_input_word, c_word_mask, c_word_sent

    def prepare_label_feature(self, truncate_length):
        print('Prepare Label Feature')
        if hasattr(self, 'term_count'):
            term_count = self.term_count
        else:
            term_count = 1
        if hasattr(self, 'sort_method'):
            sort_method = self.sort_method
        else:
            sort_method = 'max'
        c_input_word, c_word_mask, c_word_sent = self.process_label(self.ind2c, truncate_length,
                                                                    term_count=term_count,
                                                                    method=sort_method)
        # mc_input_word, mc_word_mask, mc_word_sent = self.process_label(self.ind2mc, truncate_length)
        self.c_input_word = torch.LongTensor(c_input_word)
        self.c_word_mask = torch.LongTensor(c_word_mask)
        self.c_word_sent = torch.LongTensor(c_word_sent)
        # self.mc_input_word = torch.LongTensor(mc_input_word)
        # self.mc_word_mask = torch.LongTensor(mc_word_mask)
        # self.mc_word_sent = torch.LongTensor(mc_word_sent)


def my_collate_fn(batch):
    type_count = len(batch[0])
    batch_size = len(batch)
    output = ()
    for i in range(type_count):
        tmp = []
        for item in batch:
            tmp.extend(item[i])
        if len(tmp) <= batch_size:
            output += (torch.LongTensor(tmp),)
        elif isinstance(tmp[0], int):
            output += (torch.LongTensor(tmp).reshape(batch_size, -1),)
        elif isinstance(tmp[0], float):
            output += (torch.FloatTensor(tmp).reshape(batch_size, -1),)
        elif isinstance(tmp[0], list):
            dim_y = len(tmp[0])
            if isinstance(tmp[0][0], int):
                output += (torch.LongTensor(tmp).reshape(batch_size, -1, dim_y),)
            elif isinstance(tmp[0][0], float):
                output += (torch.FloatTensor(tmp).reshape(batch_size, -1, dim_y),)
    return output


def load_vocab(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        words = [line.strip().split()[0] for line in lines]
    except BaseException:
        if path.endswith('.model'):
            model = gensim.models.Word2Vec.load(path)
        if path.endswith('.bin'):
            model = gensim.models.KeyedVectors.load_word2vec_format(
                path, binary=True)
        words = list(model.wv.key_to_index) #看在word2vec中存在的词的索引
        del model

    # hard code to trim word embedding size
    try:
        with open('./embedding/word_count_dict.json', 'r') as f:    #查看某一个单词出现的次数
            word_count_dict = ujson.load(f)
    except BaseException:
        with open('../embedding/word_count_dict.json', 'r') as f:
            word_count_dict = ujson.load(f)
    words = [w for w in words if w in word_count_dict]  #所有在数据中出现的单词

    for w in ["**UNK**", "**PAD**", "**MASK**"]:    #加入一些未登录词
        if not w in words:
            words = words + [w]
    word2id = {word: idx for idx, word in enumerate(words)} #构建词的索引
    id2word = {idx: word for idx, word in enumerate(words)} #构建索引到词
    return word2id, id2word


def load_full_codes(train_path, version='mimic3'):
    """
        Inputs:
            train_path: path to train dataset
            version: which (MIMIC) dataset
        Outputs:
            code lookup, description lookup
    """
    # get description lookup
    desc_dict = load_code_descriptions(version=version) #得到代码的描述
    # build code lookups from appropriate datasets
    if version == 'mimic2':
        ind2c = defaultdict(str)
        codes = set()
        with open('%s/proc_dsums.csv' % MIMIC_2_DIR, 'r') as f:
            r = csv.reader(f)
            # header
            next(r)
            for row in r:
                codes.update(set(row[-1].split(';')))
        codes = set([c for c in codes if c != ''])
        ind2c = defaultdict(str, {i: c for i, c in enumerate(sorted(codes))})
    else:
        codes = set()
        for split in ['train', 'dev', 'test']:
            with open(train_path.replace('train', split), 'r') as f:
                lr = csv.reader(f)
                next(lr)
                for row in lr:
                    for code in row[3].split(';'):
                        codes.add(code)
        codes = set([c for c in codes if c != ''])
        ind2c = defaultdict(str, {i: c for i, c in enumerate(sorted(codes))})
    return ind2c, desc_dict


def reformat(code, is_diag):
    """
        Put a period in the right place because the MIMIC-3 data files exclude them.
        Generally, procedure codes have dots after the first two digits,
        while diagnosis codes have dots after the first three digits.
    """
    code = ''.join(code.split('.'))
    if is_diag:
        if code.startswith('E'):
            if len(code) > 4:
                code = code[:4] + '.' + code[4:]
        else:
            if len(code) > 3:
                code = code[:3] + '.' + code[3:]
    else:
        code = code[:2] + '.' + code[2:]
    return code


def load_code_descriptions(version='mimic3'):
    # load description lookup from the appropriate data files
    desc_dict = defaultdict(str)
    if version == 'mimic2':
        with open('%s/MIMIC_ICD9_mapping' % MIMIC_2_DIR, 'r') as f:
            r = csv.reader(f)
            # header
            next(r)
            for row in r:
                desc_dict[str(row[1])] = str(row[2])
    else:
        with open("%s/D_ICD_DIAGNOSES.csv" % (DATA_DIR), 'r') as descfile:
            r = csv.reader(descfile)
            # header
            next(r)
            for row in r:
                code = row[1]
                desc = row[-1]
                desc_dict[reformat(code, True)] = desc
        with open("%s/D_ICD_PROCEDURES.csv" % (DATA_DIR), 'r') as descfile:
            r = csv.reader(descfile)
            # header
            next(r)
            for row in r:
                code = row[1]
                desc = row[-1]
                if code not in desc_dict.keys():
                    desc_dict[reformat(code, False)] = desc
        with open('%s/ICD9_descriptions' % DATA_DIR, 'r') as labelfile:
            for _, row in enumerate(labelfile):
                row = row.rstrip().split()
                code = row[0]
                if code not in desc_dict.keys():
                    desc_dict[code] = ' '.join(row[1:])
    return desc_dict    #构建所有的词典


def load_embeddings(embed_file):
    W = []
    word_list = []
    try:
        with open(embed_file,'rb') as ef:
            for line in ef:
                line = line.rstrip().split()
                word_list.append(line[0])
                vec = np.array(line[1:]).astype(np.float)
                # also normalizes the embeddings
                vec = vec / float(np.linalg.norm(vec) + 1e-6)
                W.append(vec)
        word2id, id2word = load_vocab(embed_file)
    except BaseException:
        if embed_file.endswith('.model'):
            model = gensim.models.Word2Vec.load(embed_file)
        if embed_file.endswith('.bin'):
            model = gensim.models.KeyedVectors.load_word2vec_format(
                embed_file, binary=True)
        words = list(model.wv.key_to_index)

        original_word_count = len(words)

        # hard code to trim word embedding size
        with open('./embedding/word_count_dict.json', 'r') as f:
            word_count_dict = ujson.load(f)
        words = [w for w in words if w in word_count_dict]

        for w in ["**UNK**", "**PAD**", "**MASK**"]:
            if not w in words:
                words = words + [w]
        word2id = {word: idx for idx, word in enumerate(words)}
        id2word = {idx: word for idx, word in enumerate(words)}
        new_W = []
        for i in range(len(id2word)):
            if not id2word[i] in ["**UNK**", "**PAD**", "**MASK**"]:
                new_W.append(model.wv.__getitem__(id2word[i]))
            elif id2word[i] == "**UNK**":
                print("adding unk embedding")
                new_W.append(np.random.randn(len(new_W[-1])))
            elif id2word[i] == "**MASK**":
                print("adding mask embedding")
                new_W.append(np.random.randn(len(new_W[-1])))
            elif id2word[i] == "**PAD**":
                print("adding pad embedding")
                new_W.append(np.zeros_like(new_W[-1]))
        new_W = np.array(new_W)
        print(f"Word count: {len(id2word)}")
        print(f"Load embedding count: {len(new_W)}")
        print(
            f"Original word count: {original_word_count}/{len(word_count_dict)}")
        del model
        return new_W

    if not "**UNK**" in word_list:
        # UNK embedding, gaussian randomly initialized
        print("adding unk embedding")
        word_list.append("**UNK**")
        vec = np.random.randn(len(W[-1]))
        vec = vec / float(np.linalg.norm(vec) + 1e-6)
        W.append(vec)
    if not "**MASK**" in word_list:
        # UNK embedding, gaussian randomly initialized
        print("adding unk embedding")
        word_list.append("**UNK**")
        vec = np.random.randn(len(W[-1]))
        vec = vec / float(np.linalg.norm(vec) + 1e-6)
        W.append(vec)
    if not "**PAD**" in word_list:
        print("adding pad embedding")
        word_list.append("**PAD**")
        vec = np.zeros_like(W[-1])
        W.append(vec)

    print(f"Word count: {len(id2word)}")
    print(f"Load embedding count: {len(W)}")
    print(f"Original word count: {original_word_count}/{len(word_count_dict)}")
    word2newid = {w: i for i, w in enumerate(word_list)}
    new_W = []
    for i in range(len(id2word)):
        new_W.append(W[word2newid[id2word[i]]])
    new_W = np.array(new_W)
    del model
    return new_W

