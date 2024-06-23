import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim,dropout=0.0, act="relu"):
        super().__init__()

        self.layers = nn.Linear(input_dim, output_dim)
        self.dropout = dropout
        self.dropouts = nn.Dropout(dropout)
        if act == "relu":
            # self.act_fn = nn.Tanh()
            self.act_fn = F.leaky_relu
        elif act == "gelu":
            self.act_fn = F.gelu

    def forward(self, x):
        x = self.act_fn(self.layers(x))
        x = self.dropouts(x)
        return x

class att(nn.Module):
    def __init__(self):
        super(att,self).__init__()
        self.d_rop = nn.Dropout(0.1)
        self.w_linear = MLP(512,512).to('cuda')
        self.b_linear = MLP(512,1).to('cuda')

    def forward(self,text_hidden,text_mask,des_hidden):
        label_count = 8
        batch_size, seq_length, att_dim = text_hidden.size()

        z = text_hidden.reshape(batch_size, seq_length, att_dim)
        label = des_hidden.reshape(50, label_count, att_dim)

        score = contract('abd,ecd->aebc', z, label)
        # score = score.masked_fill(mask=~text_mask[:, 0:score.shape[-2]].unsqueeze(1).unsqueeze(-1).expand_as(score),
        #                           value=float('-1e6'))
        alpha = F.softmax(score, dim=2)
        m = contract('abd,aebc->aedc', text_hidden, alpha)
        m = torch.sum(m,-1)
        # m = m.max(-1)[0]
        des_hidden = torch.max(label, 1).values


        # score = contract('abc,ec->aeb', text_hidden, des_hidden)
        # score = score.masked_fill(mask=~text_mask[:, 0:score.shape[-1]].unsqueeze(1).expand_as(score),
        #                           value=float('-1e6'))
        # alpha = F.softmax(score, dim=2)
        # m = contract('abn,aeb->aen', text_hidden, alpha)
        m = self.d_rop(m)

        w = self.w_linear(des_hidden)
        b = self.b_linear(des_hidden)

        logits = contract('blh,lh->bl', m, w) + b.squeeze(-1)

        return logits

class att1(nn.Module):
    def __init__(self,args):
        super(att1,self).__init__()
        self.d_rop = nn.Dropout(0.2)
        self.w_linear = MLP(args.rnn_dim,args.rnn_dim).to('cuda')
        self.b_linear = MLP(args.rnn_dim,1).to('cuda')
        self.tran = nn.Linear(args.rnn_dim*2,args.rnn_dim).to('cuda')

        self.two_wp = nn.Linear(args.rnn_dim,args.main_num)

        self.wlh = nn.Linear(args.code_num,args.rnn_dim)
        self.wlp = nn.Linear(args.code_num, args.rnn_dim)

    def forward(self,text_hidden,text_mask,des_hidden,weighted_outputs_par,p_p_h):
        score = contract('abc,ec->aeb', text_hidden, des_hidden)
        text_mask = text_mask.bool()
        score = score.masked_fill(mask=~text_mask[:, 0:score.shape[-1]].unsqueeze(1).expand_as(score),
                                  value=float('-1e6'))
        alpha = F.softmax(score, dim=2)
        m = contract('abn,aeb->aen', text_hidden, alpha)

        # L = des_hidden.unsqueeze(0).repeat(m.size(0),1,1)
        s_l_p = m @ torch.tanh(self.two_wp.weight.mul(p_p_h).transpose(1, 2))
        att_l_p = torch.softmax(s_l_p,2) @ weighted_outputs_par

        # if weighted_outputs_par is not None:
        #     temp = [m,
        #             weighted_outputs_par.repeat(1, 50).view(m.size(0), 50,
        #                                                     -1)]  # 在pytorch中view和tensor使用reshape是相同的作用，用来改变形状
        # temp = [m,att_l_p]
        # m = torch.cat(temp, dim=2)
        # m = self.tran(m)

        # m = m.mul(self.wlh.weight.T) + att_l_p.mul(self.wlp.weight.T)
        m = m + att_l_p

        m = self.d_rop(m)

        w = self.w_linear(des_hidden)
        b = self.b_linear(des_hidden)

        logits = contract('blh,lh->bl', m, w) + b.squeeze(-1)

        return logits

def d_att(text_hidden,text_mask,des_hidden):
    m_last = []
    ATT = att()
    m = ATT(text_hidden,text_mask,des_hidden)
    m_last.append(m)

    # weighted_output = []
    # des_hidden = des_hidden.permute(0, 2, 1)
    # A = text_hidden @ des_hidden
    # A = torch.softmax(A,1).permute(0, 2, 1)
    # weighted_output.append(W.weight.mul(A).sum(dim=2).add(W.bias))

    return m_last

def d_att1(text_hidden,text_mask,des_hidden):
    m_last = []
    ATT1 = att1()
    m = ATT1(text_hidden,text_mask,des_hidden)
    m_last.append(m)

    return m_last