# -*- coding: utf-8 -*-

"""
    This is to create the embedding layer with the support of character-based word embeddings
    @author: Thanh Vu <thanh.vu@csiro.au>
    @date created: 01/04/2019
    @date last modified: 19/08/2020
"""

import torch.nn as nn
import torch
import copy


class EmbeddingLayer(nn.Module):
    def __init__(self,
                 pretrained_word_embeddings: torch.Tensor,
                 vocab_size: int,
                 embedding_size: int):
        """
        Init function
        :param embedding_mode: it can be "rand", "static", "non_static" or "multichannel".
            With "rand" mode, the embeddings are initialised randomly and fine tuned with the model training
            With "static" mode, the embeddings are initialised with pretrained embeddings and fixed
            With "non_static" mode, the embeddings are initialised with pretrained embeddings and fine tuned with the model training
            With "multichannel" mode, there are two versions of embeddings ("static" and "non_static")
        :param pretrained_word_embeddings: Pretrained word embeddings
        :param vocab_size: The size of the word vocab
        :param embedding_size: The embedding size
        """

        if pretrained_word_embeddings is not None:
            embedding_size = pretrained_word_embeddings.size(-1)

        super(EmbeddingLayer, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.embeddings.weight = nn.Parameter(copy.deepcopy(pretrained_word_embeddings), requires_grad=True)
        self.output_size = embedding_size

    def forward(self,
                batch_data: torch.LongTensor):
        """
        :param batch_data: [batch_size x max_len]
        :return: [batch_size x max_len x embedding_size]
            embedding_size = word_embedding_size + char_embedding_size if using char level word embeddings
        """

        embeds = self.embeddings(batch_data)  # [batch_size x max_seq_size x embedding_size]

        return embeds


