import csv
import json
import sys
import os
import torch.nn.functional as F
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import torch
from gensim.models import Word2Vec
word2vec_mode = Word2Vec.load(r'/home/zhaozhizhuo22/ICD-MSMN-master/embedding/word2vec_sg0_100.model')


CHAPTER = 1
THREE_CHARACTER = 2
FULL = 3
n_not_found = 0

from train_parser import generate_parser
parser = generate_parser()
args = parser.parse_args()

device = args.device

def pad(l, pad_token_length, pad_token_id, reverse=False):
    if len(l) > pad_token_length:
        if not reverse:
            return l[0:pad_token_length]
        return l[-pad_token_length:]
    if not reverse:
        return l + [pad_token_id] * (pad_token_length - len(l))
    return [pad_token_id] * (pad_token_length - len(l)) + l

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

def D_kb(c2ind,word2id):
    with open('/home/zhaozhizhuo22/ICD-MSMN-master/change_data/icd_kb.json', 'r', encoding='utf-8') as file:
        kb = json.load(file)
    over_label = []
    batch_word_idx = []
    batch_length = []
    batch_mask = []
    # 获得描述的文本，还需要转换为wordind
    all_right_code_des = []
    for code in c2ind:
        if code not in over_label:
            # right_code_des = []
            if code in kb:
                right_code_des_kb = kb[code]
            else:
                # print(code)
                right_code_des_kb = "**UNK**"
        all_right_code_des.append(right_code_des_kb)
    for right_code in all_right_code_des:

        words = right_code.lower().replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace('-',
                                                                                                               ' ').replace(
            ',', '').replace(';', '').replace('.', '').split(' ')
        length = len(words)
        word_idx = [word2id[word] if word in word2id else word2id["**UNK**"] for word in words]
        le = len(word_idx)
        word_mask = [1] * le
        word_mask = pad(word_mask,100,0)
        word_idx = pad(word_idx, 100, word2id["**PAD**"])
        batch_word_idx.append(torch.tensor(word_idx))
        batch_length.append(torch.tensor(length))
        batch_mask.append(torch.tensor(word_mask))

    batch_word_idx, batch_length,batch_mask = sort_batch1(batch_word_idx, batch_length,batch_mask)
    batch_word_idx = torch.LongTensor(pad_sequence(batch_word_idx, batch_first=True)).to(device)
    batch_length = torch.LongTensor(batch_length).to(device)
    batch_mask = torch.LongTensor(pad_sequence(batch_mask, batch_first=True)).to(device)

    return batch_word_idx, batch_length, batch_mask


def sort_batch1(features,  lengths, batch_mask):
    sorted_indices = sorted(range(len(features)), key=lambda i: features[i].size(0), reverse=True)
    sorted_features = []
    sorted_lengths = []
    sorted_mask = []

    for index in sorted_indices:
        sorted_features.append(features[index])
        sorted_mask.append(batch_mask[index])
        sorted_lengths.append(lengths[index])

    return sorted_features,sorted_lengths,sorted_mask


def text_encoder(c_label,text,word2id):
    batch_word_idx = []
    batch_length = []
    batch_mask = []
    # 获得描述的文本，还需要转换为wordind
    all_right_code_des = text
    for right_code in all_right_code_des:

        words = right_code.lower().replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace('-',
                                                                                                               ' ').replace(
            ',', '').replace(';', '').replace('.', '').split(' ')
        length = len(words)
        word_idx = [word2id[word] if word in word2id else word2id["**UNK**"] for word in words]
        le = len(word_idx)
        word_mask = [1] * le
        word_mask = pad(word_mask,2500,0)
        word_idx = pad(word_idx, 2500, word2id["**PAD**"])
        batch_word_idx.append(torch.tensor(word_idx))
        batch_length.append(torch.tensor(length))
        batch_mask.append(torch.tensor(word_mask))

    batch_word_idx, batch_length,batch_mask = sort_batch1(batch_word_idx, batch_length,batch_mask)
    batch_word_idx = torch.LongTensor(pad_sequence(batch_word_idx, batch_first=True)).to(device)
    batch_length = torch.LongTensor(batch_length).to(device)
    batch_mask = torch.LongTensor(pad_sequence(batch_mask, batch_first=True)).to(device)

    return batch_word_idx, batch_length, batch_mask