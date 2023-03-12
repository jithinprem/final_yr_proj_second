import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn

from basicfunc import easyprint

# vid_label = [123, 3, 53, 453]
# one_hot = [0, 0, 1, 0, 0.......]

def criterion_loss():
    # TODO: rahul h complete ctc loss function here !
    '''performing mse loss func for err cal'''
    return nn.CrossEntropyLoss()

def seq_train(loader, model, optimizer, no_classes):
    model.train()
    loss_value = []
    loss_func = criterion_loss()
    for batch_idx, data in enumerate(tqdm(loader)):
        # TODO: assign the video and labels to gpu if exist..
        vid = data[0]
        words = data[1]
        label = [0 for i in range(no_classes)]
        for ind in words:
            label[ind] = 1

        label = torch.tensor(label, dtype=torch.float32)

        y_pred = model(vid)

        # y_pred = y_pred.squeeze(0)
        # # easyprint('the predicted value is ', [y_pred, len(y_pred), len(y_pred[0])])
        # easyprint('the size of y_pred vs label : ', [y_pred.shape, label.shape])
        # loss = loss_func(y_pred, label)
        # easyprint('error occoured : ', loss)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # loss_value.append(loss.item())
    # optimizer.scheduler.step()
    return loss_value