import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

from basicfunc import easyprint

lr = 0.001
epochs = 10

class LSTMVideoClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        out1 = self.fc(h_n[-1])
        notes_out = output[:, -1, :]
        return output, out1, notes_out



class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2, use_bn=False, num_classes=-1):
        super(TemporalConv, self).__init__()
        self.use_bn = use_bn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
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

        if self.num_classes != -1:
            self.fc = nn.Linear(self.hidden_size, self.num_classes)


    def forward(self, frame_feat):
        visual_feat = self.temporal_conv(frame_feat)
        return visual_feat



class SingleConv(nn.Module):
    '''
    defining a 1d convolution layers with K5, P2, K5, P2
    change it here if required
    '''
    def __init__(self, input_channel, output_channel):
        super(SingleConv, self).__init__()
        self.in_channel = input_channel
        self.out_channel = output_channel

        layer1 = nn.Conv1d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=5, stride=1, padding=0)
        layer2 = nn.BatchNorm1d(self.out_channel)
        layer3 = nn.ReLU(inplace=True)
        layer4 = nn.MaxPool1d(kernel_size= 2, ceil_mode=False)

        # now the input to the conv1d is of size 1024(hidden layer)
        layer5 = nn.Conv1d(in_channels= self.out_channel, out_channels=self.out_channel, kernel_size=5, stride=1, padding=0)
        layer6 = nn.BatchNorm1d(self.out_channel)
        layer7 = nn.ReLU(inplace=True)
        layer8 = nn.MaxPool1d(kernel_size=2, ceil_mode=False)

        # self.fc = nn.Linear(self.hidden_size, self.num_classes)

        layers = [layer1, layer2 ,layer3, layer4, layer5, layer6, layer7, layer8]
        self.single_conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.single_conv(x)
        return out




class Signmodel(nn.Module):

    def __init__(self, num_classes):
        super(Signmodel, self).__init__()
        self.conv2d = models.resnet18(weights="IMAGENET1K_V1")
        self.conv2d.fc = nn.Linear(in_features=512, out_features=512)
        self.single_conv = SingleConv(input_channel=512, output_channel=1024)
        self.temp_conv = TemporalConv(input_size=512, hidden_size=1024 )
        self.temporal_lstm = LSTMVideoClassifier(input_size=1024, hidden_size=1024, output_size=num_classes)

    def forward(self, x):
        batch, temp, channel, height, width = x.shape

        print('input to model: ', x.shape)
        x = x.reshape(-1, 3, 224, 224)
        print('after reshaping going to feed to 2d:', x.shape)
        print('input to conv2d : ', x.shape)
        out = self.conv2d(x)
        print('after conv2d : ', out.shape)
        out = out.reshape(batch, temp, -1).transpose(1, 2)
        print('after reshape from 2d output : ', out.shape)
        print('input to single_conv 1d : ', out.shape)
        out = self.single_conv(out)
        print('after 1d covolution shape : ', out.shape)
        out = out.permute(0, 2, 1)
        print('after permute and feed to temporal lstm : ', out.shape)
        out = self.temporal_lstm(out)
        print('shape without fc : ', out[0].shape)
        print('shape with fc : ', out[1].shape)
        print('shape of note_way : ', out[2].shape)


        return out

    # def __init__(self, no_classes):
    #     super(Signmodel, self).__init__()
    #     self.conv2d = models.resnet18(pretrained=True)
    #     out_of_resnet = self.conv2d.fc.in_features
    #     self.conv2d.fc = nn.Linear(in_features=out_of_resnet, out_features=no_classes)  # [1, 0, 0, 0, 1....] until number of classes
    #
    # def forward(self, x):
    #     print(x.shape)
    #     if len(x.shape) == 5:
    #         batch, temp, channel, height, width = x.shape
    #         inputs = x.reshape(batch * temp, channel, height, width)
    #     out = self.conv2d(inputs)
    #     # TODO : out.permute(2,0,1)
    #     return out


    def criterion_calculation(self):
        return self.ctc_loss_func

