import json

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from preprocess.coder_init_code_embedding import coder_init, load_code_embedding
from model.mlp import MLP
from torch.nn.init import xavier_uniform_ as xavier_uniform
from opt_einsum import contract
import math
import matplotlib.pyplot as plt
import os


class Decoder(nn.Module):
    def __init__(self, decoder_config={}):
        super(Decoder, self).__init__()
        self.decoder_config = decoder_config
        self.input_dim = self.decoder_config['input_dim']
        self.attention_dim = self.decoder_config['attention_dim']
        # self.label_count = self.decoder_config['label_count']
        self.code_embedding_path = self.decoder_config['code_embedding_path']
        self.ind2c = self.decoder_config['ind2c']
        self.ind2mc = self.decoder_config['ind2mc']
        
        # self.final = nn.Linear(self.input_dim, 1)
        self.final = nn.Linear(self.input_dim, len(self.ind2c))
        self.ignore_mask = False
        
        self.est_cls = self.decoder_config['est_cls']
        if self.est_cls > 0:
            # self.w_linear = nn.Linear(self.attention_dim, self.attention_dim)
            # self.b_linear = nn.Linear(self.attention_dim, 1)
            self.w_linear = MLP(self.attention_dim, self.attention_dim, self.attention_dim, self.est_cls)
            self.b_linear = MLP(self.attention_dim, self.attention_dim, 1, self.est_cls)
        elif self.est_cls == -1:
            self.w_linear = nn.Identity()
            self.b_linear = self.zero_first_dim

    def zero_first_dim(self, x):
        return torch.zeros_like(x)[:,0]

    def forward(self, h, word_mask, label_feat=None):
        m = self.get_label_queried_features(h, word_mask, label_feat)
        
        if hasattr(self, 'w_linear'):
            w = self.w_linear(label_feat) # label * hidden
            b = self.b_linear(label_feat) # label * 1
            logits = self.get_logits(m, w, b)
        else:
            logits = self.get_logits(m)
        return logits
    
    def get_logits(self, m, w=None, b=None):
        # logits = self.final(m).squeeze(-1)
        # m: batch * label * hidden
        if w is None:
            # logits = self.final(m).squeeze(-1)
            logits = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        else:
            logits = contract('blh,lh->bl', m, w) + b.squeeze(-1)
        return logits
    
    def get_label_queried_features(self, h, word_mask, label_feat):
        raise NotImplementedError
    
    def _code_emb_init(self, layer, version=None, init_code='ind2c'):
        if not self.code_embedding_path:
            return
        dim = layer.weight.shape[1]
        # if version is None:
        #     if self.label_count == 50 or self.label_count > 7000: 
        #         version = "mimic3"
        version = "mimic3"

        try:
            if init_code == "ind2c":
                ind2c = self.ind2c
            if init_code == "ind2mc":
                ind2c = self.ind2mc
            label_count = len(ind2c)
            code_embs = coder_init(self.code_embedding_path, ind2c, dim, 'cuda:0', version)
        except BaseException:
            code_embs = load_code_embedding(self.code_embedding_path)

        print(f'Init Layer {layer} using {self.code_embedding_path}')
            
        weights = np.zeros(layer.weight.size())
        for i in range(label_count):
            code = ind2c[i]
            weights[i] = code_embs[code]
        layer.weight.data[0:weights.shape[0]] = torch.Tensor(weights).clone()


class LAAT(Decoder):
    def __init__(self, decoder_config={}):
        super(LAAT, self).__init__(decoder_config)
        self.W = nn.Linear(self.input_dim, self.attention_dim)
        self.U = nn.Linear(self.attention_dim, len(self.ind2mc))
        
        self.xavier = self.decoder_config['xavier']
        
        if self.xavier:
            xavier_uniform(self.W.weight)
        self._code_emb_init(self.U, init_code='ind2mc')
    
    def get_label_queried_features(self, h, word_mask=None, label_feat=None):
        if word_mask is not None and not self.ignore_mask:
            l = word_mask.shape[-1]
            h = h[:,0:l]
        z = torch.tanh(self.W(h))
        if label_feat is None:
            label_feat = self.U.weight
        score = label_feat.matmul(z.transpose(1,2))
        if word_mask is not None and not self.ignore_mask:
            word_mask = word_mask.bool()
            score = score.masked_fill(mask=~word_mask[:,0:score.shape[-1]].unsqueeze(1).expand_as(score), value=float('-1e6'))
        alpha = F.softmax(score, dim=2)
        m = alpha.matmul(h) # Batch * Label * Hidden
        return m
    
 
class MultiLabelMultiHeadLAAT(LAAT):
    def __init__(self, decoder_config={}):
        super(MultiLabelMultiHeadLAAT, self).__init__(decoder_config)
        self.attention_head = self.decoder_config['attention_head']
        #assert self.input_dim % self.attention_head == 0
        #assert self.attention_dim % self.attention_head == 0
        self.rep_dropout = nn.Dropout(decoder_config['rep_dropout'])
        self.one = nn.Linear(self.attention_head,1,bias=True)
        self.head_pooling = decoder_config['head_pooling']
        if self.head_pooling == "concat":
            assert self.attention_dim % self.attention_head == 0
            self.reduce = nn.Linear(self.attention_dim,
                                     self.attention_dim // self.attention_head)
        # elif self.head_pooling == "attention":
        # TODO
        if decoder_config.get('att_dropout') > 0.0:
            self.att_dropout_rate = decoder_config['att_dropout']
            self.att_dropout = nn.Dropout(self.att_dropout_rate)
        
    def get_label_queried_features(self, h, word_mask=None, label_feat=None):
        if word_mask is not None:
            if not hasattr(self, 'ignore_mask') or not self.ignore_mask:
                l = word_mask.shape[-1]
                h = h[:,0:l]
        z = torch.tanh(self.W(h)) # batch_size * seq_length * att_dim
        batch_size, seq_length, att_dim = z.size()
        # z_reshape = z.reshape(batch_size, seq_length, self.attention_head, att_dim // self.attention_head)
        # batch_size, seq_length, att_head, sub_dim
        if label_feat is None:
            label_feat = self.U.weight
        label_count = label_feat.size(0) // self.attention_head
        # u_reshape = label_feat.reshape(label_count, self.attention_head, att_dim // self.attention_head)
        u_reshape = label_feat.reshape(label_count, self.attention_head, att_dim)
        # label_count * att_head * dim
        #score = u_reshape.matmul(z_reshape).squeeze(-1).squeeze(-1)
        # score = contract('abcd,ecd->aebc', z_reshape, u_reshape)
        score = contract('abd,ecd->aebc', z, u_reshape)
        # batch_size, label_count, seq_length, att_head
        
        if word_mask is not None:
            if not hasattr(self, 'ignore_mask') or not self.ignore_mask:
                word_mask = word_mask.bool()
                score = score.masked_fill(mask=~word_mask[:,0:score.shape[-2]].unsqueeze(1).unsqueeze(-1).expand_as(score),
                                        value=float('-1e6'))
        alpha = F.softmax(score, dim=2) # softmax on seq_length # batch_size, label_count, seq_length, att_head
        if hasattr(self, 'att_dropout'):
            alpha = self.att_dropout(alpha)
            if self.training:
                #alpha *= (1 - self.att_dropout_rate)
                alpha_sum = torch.clamp(alpha.sum(dim=2, keepdim=True), 1e-5)
                alpha = alpha / alpha_sum
        # alpha_permute = alpha.permute(0, 3, 1, 2)
        # # batch_size, attention_head, label_count, seq_length
        
        # h_reshape = h.reshape(batch_size, seq_length, self.attention_head, -1)
        # # batch_size, seq_length, attention_head, hidden // attention_head
        # h_reshape = h_reshape.permute(0, 2, 1, 3)
        # # batch_size, attention_head, seq_length, hidden // attention_head
        m = contract('abd,aebc->aedc', h, alpha)# .reshape(batch_size, label_count, -1)
        # h: batch * seq * hidden
        # a: batch * label * seq * head
        # test max pooling here / can be changed to mlp/concat/attention
        if not hasattr(self, 'head_pooling') or self.head_pooling == "max":
            m = self.one(m)
            # m = m.max(-1)[0]
        elif self.head_pooling == "concat":
            m = self.reduce(m.permute(0,1,3,2)) # batch * label * hidden // head * head
            m = m.reshape(batch_size, -1, att_dim)
        
        # m = alpha_permute.matmul(h_reshape).permute(0, 2, 1, 3).reshape(batch_size, label_count, -1)
        # batch_size, attention_head, label_count, hidden // attention_head
        # batch_size, label_count, attention_head, hidden // attention_head
        m = self.rep_dropout(m)
        return m
    
    def transform_label_feats(self, label_feat):
        if not hasattr(self, 'head_pooling') or self.head_pooling == "max":
            label_count = label_feat.shape[0] // self.attention_head
            label_feat = label_feat.reshape(label_count, self.attention_head, -1).max(1)[0]
        elif self.head_pooling == "concat":
            label_count = label_feat.shape[0] // self.attention_head
            label_feat = self.reduce(label_feat) # (label * head) * (hidden // head)
            label_feat = label_feat.reshape(label_count, self.attention_head, -1) # label * head * hidden
            label_feat = label_feat.reshape(label_count, -1)
        return label_feat

    def forward(self, h, word_mask, label_feat=None):
        m = self.get_label_queried_features(h, word_mask, label_feat)
        
        if hasattr(self, 'w_linear'):
            label_feat = self.transform_label_feats(label_feat)
            w = self.w_linear(label_feat) # label * hidden
            b = self.b_linear(label_feat) # label * 1
            logits = self.get_logits(m, w, b)
        else:
            logits = self.get_logits(m)
        return logits
    
ACT2FN = {'tanh':torch.tanh,
          'relu':torch.relu}
    
class MultiLabelMultiHeadLAATV2(MultiLabelMultiHeadLAAT):
    def __init__(self, decoder_config={}):
        super(MultiLabelMultiHeadLAATV2, self).__init__(decoder_config)
        self.act_fn_name = decoder_config['act_fn_name']
        self.act_fn = ACT2FN[self.act_fn_name]
        self.u_reduce = nn.Linear(self.attention_dim,
                                  self.attention_dim // self.attention_head)
        self.one = nn.Linear(self.attention_head, 1, bias=True)

        self.two_wp = nn.Linear(512, 49)

        # #使用自注意力来进行融合
        # self.query = nn.Linear()

    def get_label_queried_features(self, h, word_mask=None, label_feat=None,r_drop_prob=0):
        if word_mask is not None:
            if not hasattr(self, 'ignore_mask') or not self.ignore_mask:
                l = word_mask.shape[-1]
                h = h[:,0:l]
        z = self.act_fn(self.W(h)) # batch_size * seq_length * att_dim
        batch_size, seq_length, att_dim = z.size()
        z_reshape = z.reshape(batch_size, seq_length, self.attention_head, att_dim // self.attention_head)
        # batch_size, seq_length, att_head, sub_dim
        if label_feat is None:
            label_feat = self.U.weight
        label_count = label_feat.size(0) // self.attention_head
        u_reshape = self.u_reduce(label_feat.reshape(label_count, self.attention_head, att_dim))
        score = contract('abcd,ecd->aebc', z_reshape, u_reshape)
        if word_mask is not None:
            if not hasattr(self, 'ignore_mask') or not self.ignore_mask:
                word_mask = word_mask.bool()    #其中~表示取反，也就是说这句话代表要将其中所有为true的替换为-1e6，但是此时true代表的是之前的flase，因此要取反
                score = score.masked_fill(mask=~word_mask[:,0:score.shape[-2]].unsqueeze(1).unsqueeze(-1).expand_as(score),
                                        value=float('-1e6'))
        alpha = F.softmax(score, dim=2) # softmax on seq_length # batch_size, label_count, seq_length, att_head
        if hasattr(self, 'att_dropout'):
            alpha = self.att_dropout(alpha)
            if self.training:
                alpha *= (1 - self.att_dropout_rate)

        m = contract('abd,aebc->aedc', h, alpha)

        # #注意力可视化
        # #14对应的是50个代码中的icd code的位置
        # att_weight = alpha[:,14,:,:].max(-1)[0]
        # attention_weights_np = att_weight.detach().cpu().numpy()
        #
        # def is_folder_empty(folder_path):
        #     return len(os.listdir(folder_path)) == 0
        # # 示例用法
        # folder_path = './output/tensor/'
        # # for ii in att_weight.size(0):
        #
        # if is_folder_empty(folder_path):
        #     np.save('./output/tensor/1.npy', attention_weights_np)
        #     # print("文件夹为空")
        # else:
        #     file_names = os.listdir(folder_path)
        #     sorted_file_names = sorted(file_names, key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))
        #     file_names = sorted_file_names[-1].split('.')[0]
        #     # # 加载现有的 .npy 文件中的数据
        #     # existing_data = np.load('./output/tensor/{}}.npy')
        #     # # 将新的数据追加到现有数据中
        #     # combined_data = np.append(existing_data, attention_weights_np, axis=0)
        #     # 保存为 .npy 文件
        #     np.save('./output/tensor/{}.npy'.format(int(file_names)+1), attention_weights_np)


        # np.save('./output/tensor/tensor_vector.npy', attention_weights_np)

        # for i in range(batch_size):
        #
        #     normalized_data = F.normalize(att_weight, dim=1).cpu().numpy()
        #
        #     # # 获取需要绘制的注意力权重向量（这里以b维度为例）
        #     # b_dimension_vector = attention_weights_np[i, :]  # i为在b维度上的索引
        #     #
        #     # # 将注意力权重向量进行归一化，使得范围在0到1之间（方便后续绘图）
        #     # b_dimension_vector_normalized = (b_dimension_vector - b_dimension_vector.min()) / (
        #     #             b_dimension_vector.max() - b_dimension_vector.min())
        #
        #     # 绘制灰度图
        #     plt.figure(figsize=(6, 4))
        #     plt.imshow(normalized_data, cmap='gray', interpolation='nearest')
        #     plt.title('Attention Weight (b Dimension)')
        #     plt.colorbar()  # 添加颜色条，表示灰度值对应的权重值
        #     # 保存热图为图片
        #     plt.savefig('./attention_heatmap{}.png'.format(i), dpi=300, bbox_inches='tight')
        #     plt.show()

        if not hasattr(self, 'head_pooling') or self.head_pooling == "max":

            # #join linear fusion
            # m = self.one(m).max(-1)[0]

            #join att fusion
            m_score = m @ m.permute(0, 1, 3, 2)
            m_score = m_score / math.sqrt(self.attention_head)
            att_pre = F.softmax(m_score, dim=-1)

            m_fusion = torch.matmul(att_pre, m) + m
            m = self.one(m_fusion).max(-1)[0]
        elif self.head_pooling == "concat":
            # m = self.reduce(m.permute(0,1,3,2)) # batch * label * hidden // head * head
            # m = m.reshape(batch_size, -1, att_dim)
            m_score = m @ m.permute(0, 1, 3, 2)
            m_score = m_score / math.sqrt(self.attention_head)
            att_pre = F.softmax(m_score, dim=-1)

            m_fusion = torch.matmul(att_pre, m)
            m = self.one(m_fusion).max(-1)[0]

        # s_l_p = m @ torch.tanh(self.two_wp.weight.mul(p_p_h).transpose(1, 2))
        # att_l_p = torch.softmax(s_l_p, 2) @ weighted_outputs_par
        # m = m + att_l_p
        
        # batch_size, label_count, attention_head, hidden // attention_head
        if self.training and r_drop_prob > 0:
            batch_s, seq_len, hidden_size = m.size()
            mask = torch.bernoulli(torch.full((batch_s,seq_len),1 - r_drop_prob)).bool().to(m.device)
            m = m + mask.unsqueeze(-1)

        m = self.rep_dropout(m)
        return m


def create_decoder(decoder_config):
    if decoder_config['model'] == 'LAAT':
        decoder = LAAT(decoder_config)
    if decoder_config['model'] == 'MultiLabelMultiHeadLAAT':
        decoder = MultiLabelMultiHeadLAAT(decoder_config)
    if decoder_config['model'] == 'MultiLabelMultiHeadLAATV2':
        decoder = MultiLabelMultiHeadLAATV2(decoder_config)
    return decoder
