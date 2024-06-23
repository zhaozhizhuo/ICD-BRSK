import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import matplotlib as mpl
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_util import MimicFullDataset, my_collate_fn
import re
import matplotlib.colors as mcolors
import seaborn as sns
from tqdm import tqdm

folder_path = './output/tensor/'
file_names = os.listdir(folder_path)
sorted_file_names = sorted(file_names, key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))

# test_dataset = MimicFullDataset("mimic3-50", "test", "./embedding/word2vec_sg0_100.model", 4000)
# text = test_dataset.df

# from nltk.tokenize import RegexpTokenizer
# tokenizer = RegexpTokenizer(r'\w+')
#
# def split(text):
#     sp = re.sub(r'\n\n+|  +', '\t', text.strip()).replace("\n",
#                                                           " ").replace("!", "\t").replace("?", "\t").replace(".", "\t")
#     return [s.strip() for s in sp.split("\t") if s.strip()]
#
#
# def n_tokenize(text):
#     texts = split(text)
#     all_text = []
#     split_text = []
#     sentence_index = []
#     word_count = 0
#     for note in texts:
#         now_text = [w.lower() for w in tokenizer.tokenize(
#             note) if not w.isnumeric()]
#         if now_text:
#             all_text.extend(now_text)
#             split_text.append(now_text)
#             word_count += len(now_text)
#             sentence_index.append(word_count)
#     return all_text, sentence_index, split_text

folder_path_text = './output/text/'
file_names_text = os.listdir(folder_path_text)
sorted_file_names_text = sorted(file_names_text, key=lambda x: os.path.getmtime(os.path.join(folder_path_text, x)))

num = 0
for name in tqdm(sorted_file_names):
    # n = int(name.split('.')[0])
    # text_folder = sorted_file_names_text[n - 1]
    # with open(folder_path_text + text_folder, 'r', encoding='utf-8') as textfile:
    #     text = json.load(textfile)['text']
    # att_weight = np.load(folder_path + name)
    #
    # # 归一化权重到[0, 1]范围
    # norm = mcolors.Normalize(vmin=att_weight.min(), vmax=att_weight.max())
    # normalized_weights = norm(att_weight)
    #
    # plt.figure(figsize=(200, 4))
    # sns.heatmap(normalized_weights, xticklabels=text)
    # plt.savefig('./output/att/attention_heatmap{}.png'.format(num), dpi=300, bbox_inches='tight')
    # plt.show()
    # num += 1
    if num == 1:
        n = int(name.split('.')[0])
        text_folder = sorted_file_names_text[n-1]
        with open(folder_path_text+text_folder,'r',encoding='utf-8') as textfile:
            text = json.load(textfile)['text']
            # text = ' '.join(json.load(textfile)['text'])
        att_weight = np.load(folder_path + name)

        # normalized_weights = att_weight

        # 归一化权重到[0, 1]范围
        norm = mcolors.Normalize(vmin=att_weight.min(), vmax=att_weight.max())
        normalized_weights = norm(att_weight)
        att_list = normalized_weights.tolist()[0]
        text_location = []
        for i,i_num in enumerate(att_list):
            if i_num>=0.05:
                text_location.append(i)
        att_text = [' '.join(text[i-5:i+5]) for i in text_location]

        A = text[2160:2180]

        plt.figure(figsize=(200, 7))
        sns.heatmap(normalized_weights, xticklabels=text)
        plt.savefig('./output/att/attention_heatmap{}.png'.format(num), dpi=300, bbox_inches='tight')
        plt.show()
        num += 1
    else:
        print(num)
        num += 1
        continue

    # variables = text
    # labels = ['ID_0']
    #
    # mpl.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # 设置中文字体为黑体
    # mpl.rcParams['axes.unicode_minus'] = False
    #
    # df = pd.DataFrame(normalized_weights, columns=variables, index=labels)  # 其中d为4*4的矩阵
    # fig = plt.figure(figsize=(15, 15))  # 设置图片大小
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(df, interpolation='nearest', cmap='hot_r')
    # fig.colorbar(cax)
    #
    # tick_spacing = 1
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    #
    # ax.set_xticklabels([''] + list(df.columns))
    # ax.set_yticklabels([''] + list(df.index))
    #
    # plt.show()

    # plt.figure(figsize=(30, 20))
    # sns.heatmap(attention, vamx=100, vmin=0)
    # plt.savefig('./log/attention_matrix.png')

    # normalized_data = F.normalize(torch.from_numpy(att_weight), dim=1).cpu().numpy()

    # # 创建一个颜色映射，根据权重来选择颜色
    # colormap = plt.cm.get_cmap('viridis') # 这里使用 'viridis' 色谱，你可以根据需要选择不同的颜色映射
    #
    # # 创建一个颜色映射函数，根据权重来选择颜色
    # def custom_color_map(weight):
    #     # 这里使用简单的线性映射，你可以根据权重设置不同的颜色映射函数
    #     r = np.minimum(2 * weight, 1)
    #     g = np.minimum(2 * (1 - weight), 1)
    #     b = np.zeros_like(weight)
    #     return np.column_stack([r, g, b])
    #
    # # 将文本和对应的权重映射到颜色
    # colors = custom_color_map(normalized_weights)
    #
    # # 绘制文本
    # fig, ax = plt.subplots()
    # ax.text(0.1, 0.5, text, fontsize=12, color='black', backgroundcolor='white',
    #         bbox=dict(facecolor='white', edgecolor='white'))
    # for i, (char, color) in enumerate(zip(text, colors)):
    #     ax.text(0.1 + i * 0.015, 0.5, char, fontsize=12, color=color[:3])  # 只取RGB部分
    #
    # # 移除坐标轴
    # ax.axis('off')
    #
    # plt.show()

    # # 绘制灰度图
    # plt.figure(figsize=(6, 4))
    # plt.imshow(normalized_data, cmap='gray', interpolation='nearest')
    # plt.title('Attention Weight (b Dimension)')
    # plt.colorbar()  # 添加颜色条，表示灰度值对应的权重值
    # # 保存热图为图片
    # # plt.savefig('./attention_heatmap{}.png'.format(i), dpi=300, bbox_inches='tight')
    # plt.show()
