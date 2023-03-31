import pdb
import copy
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
#from modules import BiLSTMLayer, TemporalConv

class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2, use_bn=False, num_classes=-1):
        super(TemporalConv, self).__init__()
        self.use_bn = use_bn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.conv_type = conv_type

        # also try different conv type
        if self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)

        self.fc = nn.Linear(self.hidden_size, self.num_classes)


    def update_lgt(self, lgt):
        # this function is to get the value of the image size after the Kernal or Pooling is applied
        feat_len = copy.deepcopy(lgt)
        for ks in self.kernel_size:
            if ks[0] == 'P':
                feat_len //= 2   # since we are doing pooling with 2x2 we are dividing it by 2
            else:
                feat_len -= int(ks[1]) - 1   # the formula learnt in the youtube video pytorch
        return feat_len

    def forward(self, x):
        # TODO: pass x through the self.temporal_conv
        return x


class BiLSTMLayer(nn.Module):
    def __init__(self, input_size, debug=False, hidden_size=512, num_layers=1, dropout=0.3,
                 bidirectional=True, rnn_type='LSTM', num_classes=-1):
        super(BiLSTMLayer, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = int(hidden_size / self.num_directions)
        self.rnn_type = rnn_type
        self.debug = debug
        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional)

    def forward(self, src_feats):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(src_feats)

        # concatenate the final hidden states from both directions
        h_n_concat = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)

        out = self.fc(h_n_concat)
        return out





class SLRModel(nn.Module):
    def __init__(self, num_classes, c2d_type, conv_type, use_bn=False, tm_type='BiLSTM',
                 hidden_size=1024, gloss_dict=None, loss_weights=None):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.conv2d = getattr(models, c2d_type)(pretrained=True)
        self.conv2d.fc = nn.Identity()
        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)

        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                              num_layers=2, bidirectional=True)
        # TODO: implemnt the decoder
        # self.decoder = utils.Decode(gloss_dict, num_classes)
        self.classifier = nn.Linear(hidden_size, self.num_classes)


        def forward(self, x, len_x):
            # videos
            # TODO: do necessary transformations to data before feeding
            framewise = self.conv2d(x)
            conv1d_outputs = self.conv1d(framewise)
            tm_outputs = self.temporal_model(x)

            # TODO: Implement and use deocder here
            return {
                #return the generated stuff here
            }


        def criterion_calculation(self, ret_dict, label, label_lgt):
            loss = 0
            for k, weight in self.loss_weights.items():
                if k == 'ConvCTC':
                    loss += weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                          label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                          label_lgt.cpu().int()).mean()
                elif k == 'SeqCTC':
                    loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                          label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                          label_lgt.cpu().int()).mean()
            return loss



