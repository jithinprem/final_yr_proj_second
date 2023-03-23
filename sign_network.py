import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import CTCDecoder

from basicfunc import easyprint

lr = 0.001
epochs = 30

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

        layers = [layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8]
        self.single_conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.single_conv(x)
        return out

class decode:
    def __init__(self):
        pass


class Signmodel(nn.Module):


    def __init__(self, num_classes):
        super(Signmodel, self).__init__()
        self.conv2d = models.resnet18(weights="IMAGENET1K_V1")
        self.conv2d.fc = nn.Linear(in_features=512, out_features=512)
        self.single_conv = SingleConv(input_channel=512, output_channel=1024)
        self.temporal_lstm = LSTMVideoClassifier(input_size=1024, hidden_size=1024, output_size=num_classes)
        self.classifier = nn.Linear(in_features= 1024, out_features=num_classes+1) # 224 is (no_classes +1 )

        # Our loss func : ctc loss
        self.myloss = torch.nn.CTCLoss(reduction='none', zero_infinity=False)

    def forward(self, x):
        batch, temp, channel, height, width = x.shape

        print('input to model: ', x.shape)
        # input to model : [1, 176, 3, 224, 224]   176 depends on no. of frames in the sign
        x = x.reshape(-1, 3, 224, 224)
        # after reshape : [176 , 3 , 224, 224]
        out = self.conv2d(x)
        # out.shape = [176, 512] output of fc is set to 512
        print('after conv2d : ', out.shape)
        out = out.reshape(batch, temp, -1).transpose(1, 2)
        # out after reshape : [1, 512, 176]
        out = self.single_conv(out)
        print('after 1d covolution shape : ', out.shape)
        # shape out = [1, 1024, 41]
        out = out.permute(0, 2, 1)
        print('after permute and feed to temporal lstm : ', out.shape)
        out = self.temporal_lstm(out)
        print('shape of simply output is : ')
        print('shape with fc : ', out[1].shape)
        print('shape of note_way : ', out[2].shape)
        out = self.classifier(out[0])
        print('dimension after classifer : ', out.shape)
        out = out.permute(1, 0, 2)
        print('dimension after permute to feed to loss func : ', out.shape)



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


    def criterion_calculation(self, ret_T_N_C1, true_label, inp_len, lab_len):
        # ctc loss, feat_len : feature_length
        weight = 1
        loss = 0

        inp_len = torch.tensor(inp_len)
        inp_len = inp_len.to(torch.int)

        lab_len = torch.tensor(lab_len)
        lab_len = lab_len.to(torch.int)

        loss = weight * self.myloss(ret_T_N_C1.log_softmax(-1),
                                              true_label, inp_len,
                                              lab_len).mean()
        return loss

