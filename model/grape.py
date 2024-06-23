import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv,GATConv

device = 'cuda'

CHAPTER = 1
THREE_CHARACTER = 2
FULL = 3
n_not_found = 0

def reformat(code,label2index):
    code = code.split(';')
    re_code = []
    for c in code:
        if c in label2index:
            re_code.append(c)
    return re_code

class GCN(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(GCN,self).__init__()
        # self.conv1 = GATConv(input_dim,hidden_dim,heads=8,concat=False,dropout=0.3)
        # self.conv2 = GATConv(hidden_dim*8,output_dim,heads=1,concat=False,dropout=0.3)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        # self.l = nn.Linear(output_dim,924)

    def forward(self,x,edge_index,edge_weight):
        x = F.leaky_relu(self.conv1(x,edge_index,edge_weight))
        # x = F.dropout(x,p=0.2)
        x = self.conv2(x,edge_index,edge_weight)
        # x = self.l(x)
        return x
        # return F.softmax(x,dim=1)



def create_grape(label2index,train_data,label_feature):
    all_label = list(label2index.values())
    num = len(all_label)
    graph = [[0 for col in range(num)] for row in range(num)]
    for df in train_data:
        label = reformat(df['LABELS'],label2index)
        src = label2index[label[0]]
        for trg_code in label:
            trg = label2index[trg_code]
            if trg == src:
                continue
            else:
                graph[src][trg] = graph[src][trg] + 1
                graph[trg][src] = graph[trg][src] + 1

    # all_weight = 0
    # for i in range(num):
    #     for j in range(i):
    #         all_weight += graph[i][j]
    # print(all_weight,all_weight//(num*num))

    for i in range(num):
        for j in range(num):
            if graph[i][j] > 60:
                # graph[i][j] = 1
                graph[i][j] = graph[i][j]
            else:
                graph[i][j] = 0

    # # 定义邻接矩阵
    # adj_matrix = np.array(graph)
    #
    # # 创建图对象
    # G = nx.from_numpy_matrix(adj_matrix)
    #
    # # 绘制图形
    # nx.draw(G, with_labels=True)
    #
    # # 显示图形
    # plt.show()

    def matrix_to_list(matrix):
        s = []
        d = []
        edge_weight = []
        for i in range(num):
            for j in range(i):
                if matrix[i][j] != 0:
                    edge_weight.append(graph[i][j])
                    s.append(i)
                    d.append(j)
        s = torch.LongTensor(s)
        d = torch.LongTensor(d)
        edge_index = torch.stack([s, d], dim=0)
        return edge_index, edge_weight

    # data = label_feature.reshape(label_feature.size(0)//8,-1)
    data = label_feature.reshape(label_feature.size(0) // 8, 8, -1)
    data = torch.max(data,1)[0]
    edge_index, edge_weight = matrix_to_list(graph)
    input_dim = 512
    hidden_dim = 128
    output_dim = 512
    GCN_model = GCN(input_dim, hidden_dim, output_dim).to(device)
    edge_weight = torch.Tensor(edge_weight).to(device)
    edge_index = edge_index.to(device)
    output = GCN_model(data, edge_index, edge_weight)
    # output = output.repeat(8,1)
    final = torch.zeros(output.repeat(8,1).size()).to(device)
    for i in range(output.size(0)):
        for j in range(8):
            final[i*8+j] = output[i]
            # print(i*8+j)

    return final