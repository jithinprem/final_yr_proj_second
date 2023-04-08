import numpy as np
import torch
from fast_ctc_decode import viterbi_search,beam_search
from tqdm import tqdm
import torch.nn as nn
import worderrorrate
from jiwer import wer
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
    # model.eval()
    loss_value = []
    for batch_idx, data in enumerate(tqdm(loader)):
        # TODO: assign the video and labels to gpu if exist..
        vid = data[0]
        words = data[1]
        lab_pred = model(vid)
        print(lab_pred.shape)
        lab_pred = lab_pred.reshape(-1, 224)
        # lab_pred = lab_pred.permute(1,0)
        lab_pred = lab_pred.detach().numpy()
        # use decoder to decode sentence
        print(lab_pred.shape)
        print(len(gloss_dict))
        print(gloss_dict.keys())
        vocab = [chr(x) for x in range(0, len(gloss_dict)+2)]
        seq, path = beam_search(lab_pred, list(gloss_dict.keys())+['_']+['-'],beam_size=5,beam_cut_threshold=0.0001)
        seq, path = beam_search(lab_pred, vocab,beam_size=5)
        op = []
        values = list(gloss_dict.keys())+['-','_']
        for alpha in seq:
            op.append(values[ord(alpha)])
        print(op)
        # org = list(gloss_dict.keys())
        org = []
        for seq in words:
            org.append((values[seq]))
        print(org)
        print('seq : ',seq)
        print('path : ', path)
        print('og_label : ', words)

        # calculate loss through word err rate
        w = worderrorrate.WER(words, seq)
        # syms = []
        # for id in words:
        #     id = int(id.item())
        #     syms += gloss_dict[id]
        # w = wer(syms,seq)
        print('word err :', w)
        print('wer value: ',w.wer())
        loss_value.append(w)
    return loss_value




