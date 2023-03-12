import torch
import torch.nn as nn
import torchvision

from basicfunc import easyprint


class LSTMVideoClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        x = x.view(batch_size * timesteps, C, H, W)
        x = torchvision.models.resnet18(pretrained=True)(x)
        x = x.view(batch_size, timesteps, -1)
        easyprint('the new size before lstm is : ', x.shape)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out

