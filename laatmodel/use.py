# -*- coding: utf-8 -*-
"""
    Word-based RNN model for text classification
    @author: Thanh Vu <thanh.vu@csiro.au>
    @date created: 07/03/2019
    @date last modified: 19/08/2020
"""
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from laatmodel.models.attentions.util import *
from laatmodel.models.embeddings.util import *
import pickle
from torch import nn
from math import floor


class RNN(nn.Module):
    def __init__(self, vocab,
                 args):
        """

        :param vocab: Vocab
            The vocabulary normally built on the training data
        :param args:
            mode: rand/static/non-static/multichannel the mode of initialising embeddings
            hidden_size: (int) The size of the hidden layer
            n_layers: (int) The number of hidden layers
            bidirectional: (bool) Whether or not using bidirectional connection
            dropout: (float) The dropout parameter for RNN (GRU or LSTM)
        """

        super(RNN, self).__init__()
        self.vocab_size = vocab.n_words()
        self.vocab = vocab
        self.args = args
        self.use_last_hidden_state = args.rnn_dim
        self.n_layers = 1
        self.hidden_size = args.rnn_dim // 2
        self.bidirectional = bool(1)
        self.n_directions = int(self.bidirectional) + 1
        # self.attention_mode = args.attention_mode
        self.output_size = self.hidden_size
        # self.rnn_model = args.rnn_model
        self.level_projection_size = 128
        self.out_channels = 100
        self.kernel_size = 5

        self.embedding = init_embedding_layer(args, vocab)

        self.rnn = nn.LSTM(self.embedding.output_size, self.hidden_size, num_layers=self.n_layers,
                               bidirectional=self.bidirectional, dropout=self.dropout if self.n_layers > 1 else 0)
        self.conv = nn.Conv1d(in_channels=self.embedding.output_size, out_channels=self.out_channels,
                              kernel_size=self.kernel_size, padding=int(floor(self.kernel_size / 2)))

        self.use_dropout = True
        self.dropout = nn.Dropout(0.3)
        init_attention_layer(self)

    def init_hidden(self,
                    batch_size: int = 1) -> Variable:
        """
        Initialise the hidden layer
        :param batch_size: int
            The batch size
        :return: Variable
            The initialised hidden layer
        """
        # [(n_layers x n_directions) x batch_size x hidden_size]
        h = Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)).to('cuda')
        c = Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)).to('cuda')
        return h, c

    def forward(self,
                batch_data,
                lengths) -> tuple:
        """

        :param batch_data: torch.LongTensor
            [batch_size x max_seq_len]
        :param lengths: torch.LongTensor
            [batch_size x 1]
        :return: output [batch_size x n_classes]
            attention_weights
        """

        batch_size = batch_data.size()[0]
        hidden = self.init_hidden(batch_size)

        # embeds = self.embedding(batch_data)
        embeds = self.dropout(batch_data)

        self.rnn.flatten_parameters()

        embeds = pack_padded_sequence(embeds, lengths.to('cpu'), batch_first=True)

        rnn_output, hidden = self.rnn(embeds)

        hidden = hidden[0]

        rnn_output = pad_packed_sequence(rnn_output)[0]

        rnn_output = rnn_output.permute(1, 0, 2)

        weighted_outputs, attention_weights = perform_attention(self, rnn_output,
                                                                self.get_last_hidden_output(hidden)
                                                                )
        return weighted_outputs

    def get_last_hidden_output(self, hidden):
        if self.bidirectional:
            hidden_forward = hidden[-1]
            hidden_backward = hidden[0]
            if len(hidden_backward.shape) > 2:
                hidden_forward = hidden_forward.squeeze(0)
                hidden_backward = hidden_backward.squeeze(0)
            last_rnn_output = torch.cat((hidden_forward, hidden_backward), 1)
        else:

            last_rnn_output = hidden[-1]
            if len(hidden.shape) > 2:
                last_rnn_output = last_rnn_output.squeeze(0)

        return last_rnn_output

def laatrnn(args,text):
    with open('/home/ubuntu/zzz/cross_laat/LAAT-master/src/cached_data/mimic-iii_single_50/0064f75b373eb15ba5a4dce49aa2e06f.pkl', 'rb') as f:
        data = pickle.load(f)
        f.close()
    vocab = data["vocab"]

    model = RNN(vocab=vocab,args=args).to('cuda')
    init_state_dict = '/home/ubuntu/zzz/LAAT-master/src/checkpoints/mimic-iii_single_50/RNN_LSTM_1_256.static.label.0.001.0.2_01bceb6c684d7c7c457dc703995285d0/best_model.pkl'
    if init_state_dict is not None:
        f = torch.load(init_state_dict)
        model.load_state_dict(f['state_dict'])
    batch_size = text.size(0)
    lengths = torch.tensor([4000] * batch_size)
    with torch.no_grad():
        weighted_outputs = model(text,lengths)
    weighted_outputs = torch.sigmoid(weighted_outputs[0])

    return weighted_outputs

