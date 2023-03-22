import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn

from basicfunc import easyprint

# vid_label = [123, 3, 53, 453]
# one_hot = [0, 0, 1, 0, 0.......]



def seq_train(loader, model, optimizer, no_classes):
    model.train()
    loss_value = []
    for batch_idx, data in enumerate(tqdm(loader)):
        # TODO: assign the video and labels to gpu if exist..
        vid = data[0]
        words = data[1]


        y_pred = model(vid)
        # loss = model.criterion_calculation(y_pred, words, (y_pred), len(words))
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # loss_value.append(loss.item())
    # optimizer.scheduler.step()
    return loss_value