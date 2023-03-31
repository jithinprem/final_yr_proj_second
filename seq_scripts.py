import numpy as np
import torch
from fast_ctc_decode import viterbi_search
from tqdm import tqdm
import torch.nn as nn
import worderrorrate

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
        loss = model.criterion_calculation(y_pred, torch.tensor(words), int(y_pred.shape[0]), len(words))
        print('loss : ', loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value.append(loss.item())
    # optimizer.scheduler.step()
    return loss_value

def seq_test(loader, model, gloss_dict):
    model.eval()
    loss_value = []
    for batch_idx, data in enumerate(tqdm(loader)):
        # TODO: assign the video and labels to gpu if exist..
        vid = data[0]
        words = data[1]
        lab_pred = model(vid)

        # use decoder to decode sentence
        seq, path = viterbi_search(lab_pred, list(gloss_dict.keys()) + ['_'])

        # calculate loss through word err rate
        w = worderrorrate.WER(words, seq)
        print(w)
        loss_value.append(w)
    return loss_value




