import torch
import torch.nn as nn
import torch.nn.functional as F


class par_AttentionLayer(nn.Module):

    def __init__(self,
                 args,
                 n_labels=39,
                 n_level: int = 1
                 ):
        super(par_AttentionLayer, self).__init__()
        self.first_linears = nn.ModuleList([nn.Linear(args.rnn_dim, args.d_a, bias=False)])
        self.second_linears = nn.ModuleList([nn.Linear(args.d_a, args.main_num, bias=False)])
        self.third_linears = nn.ModuleList([nn.Linear(args.rnn_dim,args.main_num, bias=True)])
        self.projection_linears = nn.Linear(args.main_num,args.d_a,bias=False)
        self._init_weights(mean=0.0, std=0.03)

        self.twp_first_linears = nn.Linear(args.rnn_dim,args.rnn_dim,bias=False)
        self.twp_second_linears = nn.Linear(args.main_num,args.rnn_dim,bias=True)


    def _init_weights(self, mean=0.0, std=0.03) -> None:
        """
        Initialise the weights
        :param mean:
        :param std:
        :return: None
        """
        for first_linear in self.first_linears:
            torch.nn.init.normal_(first_linear.weight, mean, std)    #初始化第一层全连接的权重
            if first_linear.bias is not None:
                first_linear.bias.data.fill_(0)

        for linear in self.second_linears:
            torch.nn.init.normal_(linear.weight, mean, std)
            if linear.bias is not None:
                linear.bias.data.fill_(0)
        for linear in self.third_linears:
            torch.nn.init.normal_(linear.weight, mean, std)

    def forward(self, x, previous_level_projection=None, label_level=0):
        """
        :param x: [batch_size x max_len x dim (i.e., self.size)]

        :param previous_level_projection: the embeddings for the previous level output
        :param label_level: the current label level
        :return:
            Weighted average output: [batch_size x dim (i.e., self.size)]
            Attention weights
        """

        weights = torch.tanh(self.first_linears[label_level](x))

        att_weights = self.second_linears[label_level](weights)
        att_weights = F.softmax(att_weights, 1).transpose(1, 2) #交换位置（1,2）两个维度  #对应27中输出进行softmax（dim=1）是对列进行操作
        if len(att_weights.size()) != len(x.size()):
            att_weights = att_weights.squeeze()
        weighted_output = att_weights @ x      #pytorch中使用@进行相乘是正常的矩阵乘法，当使用*进行相乘的时候表示的是对应位置相乘
        #weighted_output 计算的是父节点

        Ht = self.twp_first_linears(x).transpose(1,2)
        s_P_H = weighted_output @ torch.tanh(Ht)   #s(p,h)
        att_p_h = torch.softmax(s_P_H,2)
        att_p_h = att_p_h @ x
        p_p_h = att_p_h   # 计算的是子节点
        # p_p_h = torch.sigmoid(torch.sum(torch.mul(self.twp_second_linears.weight.T,att_p_h),dim=1)).unsqueeze(1)
        # p_p_h = p_p_h.repeat(1,self.twp_second_linears.weight.T.size(0),1)

        batch_size = weighted_output.size(0)

        # if previous_level_projection is not None:   #其中vec.repeat（x，y）代表对vec对行复制x倍，对列复制y倍
        #     temp = [weighted_output,
        #             previous_level_projection.repeat(1, self.n_labels[label_level]).view(batch_size, self.n_labels[label_level], -1)]   #在pytorch中view和tensor使用reshape是相同的作用，用来改变形状
        #     weighted_output = torch.cat(temp, dim=2)
        #对应weight.mul来说，mul等于pytorch中的*乘法，也就是对对应位置进行相乘
        # weighted_output = self.third_linears[label_level].weight.mul(weighted_output).sum(dim=2).add(
        #     self.third_linears[label_level].bias)
        # # weighted_output = torch.dropout(weighted_output,0.2)
        # weighted_output = torch.sigmoid(self.projection_linears(weighted_output))

        return weighted_output, p_p_h

    # Using when use_regularisation = True
    @staticmethod
    def l2_matrix_norm(m):
        """
        Frobenius norm calculation
        :param m: {Variable} ||AAT - I||
        :return: regularized value
        """
        return torch.sum(torch.sum(torch.sum(m ** 2, 1), 1) ** 0.5)


def init_par_attention(arg,text_hidden):
    par_att = par_AttentionLayer(arg)
    x, y = par_att(text_hidden)
    return x,y