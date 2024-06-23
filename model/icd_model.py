import gc

import torch
from torch import nn
import torch.nn.functional as F

from model.text_encoder import TextEncoder
from model.word_encoder import Word_Encoder
from model.decoder import create_decoder
from model.label_encoder import LabelEncoder
from model.losses import loss_fn,focalLoss
from evaluation import all_metrics
from transformers import (
    AdamW, 
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from data_util import MimicFullDataset
import json
from model.kb.cross_attention import D_kb,text_encoder
from model.kb.dot_attention import att1
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.attentions.par_attention import par_AttentionLayer

from laatmodel.use import laatrnn
import numpy as np
import os

from torch.nn.utils.rnn import pad_sequence
def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss

def fusion_first(input):
    return nn.Linear(50,input).to('cuda')

def fusion_second(input):
    return nn.Linear(50,input).to('cuda')

class IcdModel(nn.Module):
    def __init__(self, word_config={}, combine_config={},
                 decoder_config={}, label_config={}, loss_config={}, args=None):
        super().__init__()
        self.encoder = TextEncoder(word_config, combine_config)
        self.decoder = create_decoder(decoder_config)
        self.label_encoder = LabelEncoder(label_config)
        self.loss_config = loss_config
        self.args = args
        self.T = 0.07

        self.d_att1 = att1(self.args)
        self.lad = nn.Linear(100, 1)

        # #rare
        # self.lad = nn.Linear(1, 1,bias=True)

        self.dropout = nn.Dropout(0.2)

        self.MimicFullDataset = MimicFullDataset(self.args.version, "train", self.args.word_embedding_path , self.args.truncate_length,
                                                 self.args.label_truncate_length, self.args.term_count, self.args.sort_method)
        self.l = torch.nn.Linear(1024,512,bias=False)

        self.rnn = nn.LSTM(100, self.args.laat_rnn, num_layers=1,
                           bidirectional=True, dropout=0.3)

        self.word_encoder = Word_Encoder(word_config)

        self.neg_batchsize = args.neg_sample_K

        self.right_dropout = nn.Dropout(self.args.right_drop)

        self.par_att = par_AttentionLayer(self.args, n_labels=self.args.main_num)

        # self.first = nn.Linear(50,1)
        
    def calculate_text_hidden(self, input_word, word_mask):
        hidden = self.encoder(input_word, word_mask)
        return hidden
    
    def calculate_label_hidden(self):
        label_hidden = self.calculate_text_hidden(self.c_input_word,
                                                  self.c_word_mask)
        self.label_feats = self.label_encoder(label_hidden, self.c_word_mask)

    def forward(self, batch,rdrop=False,one_test=True):
        # print(rdrop)
        if rdrop:
            return self.forward_rdrop(batch)

        input_word = batch[0,:]
        word_mask = torch.randint(0, 2, (1, input_word.size(-1)), dtype=torch.long).to('cuda')
        hidden = self.calculate_text_hidden(input_word, word_mask)

        label_hidden = self.calculate_text_hidden(self.c_input_word,
                                                  self.c_word_mask)
        label_feats = self.label_encoder(label_hidden, self.c_word_mask)
        c_logits = self.decoder(hidden, word_mask, label_feats)

        # input_word, word_mask = batch[0:2]
        # hidden = self.calculate_text_hidden(input_word, word_mask)
        # # mc_logits = self.decoder(hidden, word_mask)
        #
        # label_hidden = self.calculate_text_hidden(self.c_input_word,
        #                                           self.c_word_mask)
        # label_feats = self.label_encoder(label_hidden, self.c_word_mask)
        # c_logits = self.decoder(hidden, word_mask, label_feats)

        # mc_label = batch[-1]
        # c_label = batch[-2]
        # # mc_loss = loss_fn(mc_logits, mc_label, self.loss_config)
        # mc_loss = 0.0
        #
        # c_loss = loss_fn(c_logits, c_label, self.loss_config)
        # loss = mc_loss * self.loss_config['main_code_loss_weight'] + \
        #        c_loss * self.loss_config['code_loss_weight']
        if one_test:
            c_logits = torch.softmax(c_logits, dim=1)
            return c_logits
        # return {'mc_loss':mc_loss, 'c_loss':c_loss, 'loss':loss}
               
    
    def forward_rdrop(self, batch):

        c_label = batch[-2]
        input_word, word_mask = batch[0:2]


        label_hidden = self.calculate_text_hidden(self.c_input_word,
                                                  self.c_word_mask)
        label_feats = self.label_encoder(label_hidden, self.c_word_mask)


        batch_word_idx_kb, batch_length_kb, batch_mask_kb = D_kb(self.c2ind,self.word2id)
        label_hidden_kb = self.calculate_text_hidden(batch_word_idx_kb,batch_mask_kb)
        kb_hidden_size = label_hidden_kb.size(0)
        label_feats_kb = self.lad(label_hidden_kb.permute(0, 2, 1)).reshape(kb_hidden_size, -1)
        
        hidden0 = self.calculate_text_hidden(input_word, word_mask)
        hidden1 = self.calculate_text_hidden(input_word, word_mask)

        # 加入层级知识
        weighted_outputs_par0, p_p_h_0 = self.par_att(hidden0)
        weighted_outputs_par1, p_p_h_1 = self.par_att(hidden1)

        # ignore mc_logits
        c_logits0 = self.decoder(hidden0, word_mask, label_feats)
        c_logits1 = self.decoder(hidden1, word_mask, label_feats)

        # ignore mc_logits
        c_logits0_kb = self.d_att1(hidden0, word_mask, label_feats_kb,weighted_outputs_par0,p_p_h_0)
        c_logits1_kb = self.d_att1(hidden1, word_mask, label_feats_kb,weighted_outputs_par1,p_p_h_1)
        c_logits0 = fusion_first(c_logits1.size(0)).weight.mul(c_logits0_kb) + c_logits0
        c_logits1 = fusion_second(c_logits1.size(0)).weight.mul(c_logits1_kb) + c_logits1



        # # 加入图知识
        # c_gcn0 = self.decoder(hidden0, word_mask, label_feats_fusion)
        # c_gcn1 = self.decoder(hidden1, word_mask, label_feats_fusion)

        # # mask
        # text_mask = torch.from_numpy(
        #     np.rint(text_mask.to('cpu').detach().numpy())).bool()
        # text_mask = torch.where(text_mask, 0.7, 0.5).to('cuda')

        # c_logits0 = fusion_first(c_logits1.size(0)).weight.mul(c_logits0_kb.mul(text_mask)) + c_logits0.mul(text_mask)
        # c_logits1 = fusion_second(c_logits1.size(0)).weight.mul(c_logits1_kb.mul(text_mask)) + c_logits1.mul(text_mask)

        # batch_size = c_logits0.size(0)
        # c_logits0 = self.first.weight.repeat(batch_size, 1).mul(c_logits0_kb) + c_logits0
        # c_logits1 = self.first.weight.repeat(batch_size, 1).mul(c_logits1_kb) + c_logits1




        # c_logits1 = fusion_second(c_logits1.size(0)).weight.mul(c_logits0_kb) + c_logits0

        # #使用r-dropout
        # _use_r_drop = True
        # if _use_r_drop:
        #     kld = nn.KLDivLoss(reduction='batchmean')
        #     pred2 = c_logits1
        #     kl_weight = 4.0
        #     ce_loss = (loss_fn(c_logits0, c_label,self.loss_config) + loss_fn(pred2, c_label,self.loss_config)) / 2
        #     kl_1 = kld(F.log_softmax(c_logits0, dim=-1), F.softmax(pred2, dim=-1)).sum(-1)
        #     kl_2 = kld(F.log_softmax(pred2, dim=-1), F.softmax(c_logits0, dim=-1)).sum(-1)
        #     c_loss = ce_loss + kl_weight * (kl_1 + kl_2) / 2


        # c_loss = (loss_fn(c_gcn0, c_label, self.loss_config) + loss_fn(c_gcn1, c_label,
        #                                                                           self.loss_config)) * 0.5

        # c_loss_kb = (loss_fn(c_logits0_kb, c_label, self.loss_config) + \
        #           loss_fn(c_logits1_kb, c_label, self.loss_config)) * 0.5


        # c_loss_kb = 0
        c_loss = (loss_fn(c_logits0, c_label, self.loss_config) + \
                  loss_fn(c_logits1, c_label, self.loss_config)) * 0.5


        # print("c_loss:{},c_loss_kb:{}".format(c_loss,c_loss_kb))
        # c_loss = (c_loss + c_loss_kb) * 0.5

        # c_loss = (loss_fn(c_logits0, c_label, self.loss_config) + \
        #           loss_fn(c_logits1, c_label, self.loss_config)) * 0.5 + (loss_fn(c_gcn0,c_label,self.loss_config)+loss_fn(c_gcn1,c_label,self.loss_config)) * 0.5

        #加入对比学习
        x = self.MimicFullDataset
        with open('/home/zhaozhizhuo22/ICD-MSMN-master/change_data/icd_mimic3_random_sort.json', 'r', encoding='utf-8') as file:
            icd_right = json.load(file)
        with open('/home/zhaozhizhuo22/ICD-MSMN-master/change_data/Negative Sample.json', 'r', encoding='utf-8') as file1:
            neg_icd_des = json.load(file1)  # 找到所有的关于兄弟节点的note的字典
        with open('/home/zhaozhizhuo22/ICD-MSMN-master/change_data/last_neg.json','r',encoding='utf-8') as file2:
            neg_icd_des_good = json.load(file2)
        i_l = x.ind2c
        des = x.desc_dict
        my_list = batch[3].tolist()
        batch_num = len(my_list)
        from model.con_learn import contrast_learn
        batch_size = batch[0].size(0)
        batch_right_des = []

        batch_neg_des = []
        neg_sample_batch = []
        for my_list_1 in my_list:
            indices = [i for i in range(len(my_list_1)) if my_list_1[i] == 1]
            right_des,right_des_mask, neg_des,neg_des_mask = contrast_learn(i_l, des, indices, icd_right, neg_icd_des_good,x.word2id,self.args)
            right_em = self.word_encoder(right_des)
            batch_right_des.append(right_em.unsqueeze(0))
            batch_neg_des.append(torch.tensor(neg_des))
        #使用同义词进行正样例的编码
        # batch_right_des = torch.cat(batch_right_des,dim=0)
        # right_em = self.dropout(batch_right_des)
        # np_lengths = torch.tensor([500]*batch_size)
        # right_em = pack_padded_sequence(right_em, np_lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        # des_rnn_output, des_rnn_hidden = self.rnn(right_em)
        # right_rnn_output = pad_packed_sequence(des_rnn_output)[0]
        # right_rnn_output = right_rnn_output.permute(1, 0, 2)
        for neg_i in batch_neg_des:
            neg_i = neg_i.to('cuda')
            neg_em = self.word_encoder(neg_i)
            neg_em = self.dropout(neg_em)
            neg_length = []
            for neg_ii in neg_i:
                neg_length.append(neg_ii.size()[-1])
            neg_length = torch.Tensor(neg_length)
            neg_em = pack_padded_sequence(neg_em, neg_length.to('cpu'), batch_first=True, enforce_sorted=False)
            neg_rnn_output, neg_hidden = self.rnn(neg_em)
            neg_rnn_output = pad_packed_sequence(neg_rnn_output)[0]
            neg_rnn_output = neg_rnn_output.permute(1, 0, 2)
            neg_sample_batch.append(torch.max(neg_rnn_output, 1).values)
        # 构建loss
        right_rnn_output = self.right_dropout(hidden1)
        q = torch.max(hidden1, 1).values
        k = torch.max(right_rnn_output, 1).values
        logits_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) #正样例
        # q = rnn_output
        # k = right_rnn_output
        logits_hard = None
        for iii in range(batch_size):
            qq = q[iii].reshape(1, -1)
            hardl = torch.einsum('nc,nc->n', [qq, neg_sample_batch[iii]]).unsqueeze(-1)
            if logits_hard is None:
                logits_hard = hardl
            else:
                logits_hard = torch.cat([logits_hard, hardl], dim=1)
        # sim_hard = torch.mean(logits_hard)
        logits_hard = logits_hard.transpose(1, 0)
        logits_hard = torch.cat([logits_pos, logits_hard], dim=1) / self.T
        hard_label = torch.zeros(batch_size, self.neg_batchsize + 1, dtype=torch.float).to('cuda')
        loss_hard = loss_fn(logits_hard, hard_label, self.loss_config)



        kl_loss = compute_kl_loss(c_logits0, c_logits1)
        # kl_loss = compute_kl_loss(c_gcn0,c_gcn1)

        # loss = self.loss_config['rdrop_alpha'] * kl_loss + \
        #        c_loss * self.loss_config['code_loss_weight']
        
        loss = self.loss_config['rdrop_alpha'] * kl_loss + \
               c_loss * self.loss_config['code_loss_weight'] + 0.1 * loss_hard
        return {'kl_loss':kl_loss, 'c_loss':c_loss, 'loss':loss}

    def predict(self, batch,threshold=None):
        input_word, word_mask = batch[0:2]
        hidden = self.calculate_text_hidden(input_word, word_mask)

        #save text
        word_lengths = torch.sum(word_mask, dim=1).tolist()[0]
        input_text = input_word[0:word_lengths].tolist()
        text = [self.id2word[i] for i in input_text[0] if i!=150695]
        def is_folder_empty(folder_path):
            return len(os.listdir(folder_path)) == 0
        # 示例用法
        folder_path = './output/text/'
        data_text = {}
        data_text['text'] = text
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


        # batch_word_idx_kb, batch_length_kb, batch_mask_kb = D_kb(self.c2ind, self.word2id)
        # label_hidden_kb = self.calculate_text_hidden(batch_word_idx_kb, batch_mask_kb)
        # kb_hidden_size = label_hidden_kb.size(0)
        # label_feats_kb = self.lad(label_hidden_kb.permute(0, 2, 1)).reshape(kb_hidden_size, -1)
        # weighted_outputs_par, p_p_h = self.par_att(hidden)
        # c_logits_kb = self.d_att1(hidden, word_mask, label_feats_kb, weighted_outputs_par, p_p_h)

        assert hasattr(self, 'label_feats')
        yhat_raw = self.decoder(hidden, word_mask, self.label_feats)
        # batch_size = yhat_raw.size(0)
        # yhat_raw = self.first.weight.repeat(batch_size, 1).mul(c_logits_kb) + yhat_raw

        if isinstance(yhat_raw, tuple):
            yhat_raw = yhat_raw[0]
        if threshold is None:
            threshold = self.args.prob_threshold
        yhat = yhat_raw >= threshold
        y = batch[-2]
        return {"yhat_raw": yhat_raw, "yhat": yhat, "y": y}

    def configure_optimizers(self, train_dataloader=None):
        args = self.args
        if self.args.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)
            return [optimizer], [None]
        if self.args.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=args.learning_rate)
            return [optimizer], [None]
        if self.args.optimizer == "AdamW":
            no_decay = ["bias", "LayerNorm.weight"]
            params = [
                {
                    "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": args.weight_decay,
                    "lr": args.learning_rate
                },
                {
                    "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": args.learning_rate
                }, 
            ]
            
            optimizer = AdamW(params, eps=args.adam_epsilon)
            
            self.total_steps = len(train_dataloader) * args.train_epoch
            
            if not hasattr(self.args, 'scheduler') or self.args.scheduler == "linear":
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=int(self.total_steps * args.warmup_ratio),
                    num_training_steps=self.total_steps,
                )
            elif self.args.scheduler == "constant":
                scheduler = get_constant_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=int(self.total_steps * args.warmup_ratio)
                )
            elif self.args.scheduler == "cosine":
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=int(self.total_steps * args.warmup_ratio),
                    num_training_steps=self.total_steps,
                )
            return [optimizer], [scheduler]
