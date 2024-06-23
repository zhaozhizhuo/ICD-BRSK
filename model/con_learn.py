import random
import numpy as np
import torch


def pad(l, pad_token_length, pad_token_id, reverse=False):
    if len(l) > pad_token_length:
        if not reverse:
            return l[0:pad_token_length]
        return l[-pad_token_length:]
    if not reverse:
        return l + [pad_token_id] * (pad_token_length - len(l))
    return [pad_token_id] * (pad_token_length - len(l)) + l

def contrast_learn(i_l,des,label_list,right_note,neg_note,word_id,arg):
    pad_token_id = word_id["**PAD**"]
    neg_batchsize = arg.neg_sample_K
    # 随机选择负例code
    # 先找所有lable的兄弟进行，进行随机选择负样例
    label_b = [i_l[label_ind] for label_ind in label_list]
    # 存在一些label没有兄弟节点，跳过
    neg_label_note = []
    for ll in label_b:
        if ll in neg_note:
            x = neg_note[ll]
            neg_label_note.extend(neg_note[ll])
        else:
            continue
    neg_label_note = list(set(neg_label_note))
    neg_label_note = [x for x in neg_label_note if x not in label_b]
    neg_des_all = []
    # 存在一些lable没有代码描述，这个时候跳过
    for neg_ll in neg_label_note:
        if neg_ll in des:
            neg_des_all.append(
                des[neg_ll].replace('(', '').replace(')', '').replace('-', ' ').replace(';', '').replace('.', ''))
        else:
            neg_des_all.append('**UNK**')
    if len(neg_des_all) < neg_batchsize:
        neg_des = neg_des_all + ['**UNK**'] * (neg_batchsize - len(neg_des_all))
    else:
        neg_des = list(np.random.choice(neg_des_all, neg_batchsize))
    # 选择正样例描述
    num = 0
    right_des = []
    for right_ice in label_list:
        note_r = i_l[right_ice]
        if note_r in right_note:
            right_des.extend([
                ' '.join(right_note[note_r]).replace('(', '').replace(')', '').replace('-', ' ').replace('.', '')])
        else:
            right_des.extend(['**UNK**'])
    right_des = [' '.join(right_des)]
    # while (True):
    #     right_ice = random.randint(0, len(label_list) - 1)
    #     note_r = i_l[label_list[right_ice]]
    #     if note_r in right_note:
    #         right_des = [
    #             ' '.join(right_note[note_r]).replace('(', '').replace(')', '').replace('-', ' ').replace('.', '')]
    #         break
    #     else:
    #         num += 1
    #         if num == 10:
    #             # print('未找到对应的负样例')
    #             right_des = ['**UNK**']
    #             break
    neg_ind = []
    neg_mask = []
    for neg_des_i in neg_des:
        neg_des_i = neg_des_i.lower().replace(',', '').replace('[', ' ').replace(']', ' ').replace('(', ' ').replace(
            ')', ' ').replace('-', ' ').replace(';', ' ')
        neg_ind_i = []
        for wo in neg_des_i.split(' '):
            if wo not in word_id:
                neg_ind_i.append(word_id['**UNK**'])
            else:
                neg_ind_i.append(word_id[wo])
        neg_ind_i_mask = [1] * len(neg_ind_i) + [0] * (20 - len(neg_ind_i))
        neg_ind_i = pad(neg_ind_i, 20, pad_token_id)
        neg_ind.append(neg_ind_i)
        neg_mask.append(neg_ind_i_mask)

    for ri_des_i in right_des:
        ri_des_i = ri_des_i.lower().replace(',', '').replace('[', ' ').replace(']', ' ').replace('(', ' ').replace(')',
                                                                                                                   ' ').replace(
            '-', ' ').replace(';', ' ')
        ri_ind = []
        for wo in ri_des_i.split(' '):
            if wo in word_id:
                ri_ind.append(word_id[wo])
            else:
                ri_ind.append(word_id['**UNK**'])



    ri_ind_mask = [1] * len(ri_ind) + [0] * (500-len(ri_ind))
    ri_ind = pad(ri_ind,500,pad_token_id)


    return torch.LongTensor(ri_ind).to('cuda'),torch.LongTensor(ri_ind_mask).to('cuda'),neg_ind,neg_mask