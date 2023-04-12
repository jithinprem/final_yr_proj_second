import glob

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from basicfunc import *
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class SignDataset(Dataset):
    def __init__(self, mode):
        if mode == 'test':
            mode = 'train'
        self.inputs_list = np.load(f"preprocess/phoenix2014/{mode}_info.npy", allow_pickle=True).item()
        easyprint('saved dictionary of all details', self.inputs_list)
        self.dict = np.load(f"preprocess/phoenix2014/gloss_dict.npy", allow_pickle=True).item()
        easyprint('gloss dict', self.dict)

    def __getitem__(self, index):
        fi = self.inputs_list[index] # input_list is the dict // its printed.. see
        path = 'dataset/phoenix-2014-multisigner/features/fullFrame-256x256px/' + fi['folder']
        images_path_list = sorted(glob.glob(path))
        label_list = []
        for phase in fi['label'].split(" "):
            if phase == '':
                continue
            if phase in self.dict.keys():
                label_list.append(self.dict[phase][0])


        sign_images = torch.stack([(transform(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))) for img_path in images_path_list])

        return sign_images, label_list

    @staticmethod
    def collate_fn(batch):   # this function is used to pad the original video to approriate length
        batch = [item for item in sorted(batch, key=lambda x : len(x[0]), reverse=True)]
        video, label = list(zip(*batch))

        # maximum length video is determined
        max_len = len(video[0])

        # setting up padding
        left_pad = 6
        right_pad = int(np.ceil(max_len/4.0))*4 - max_len+6     # here we are adding 6 to the rightpad and some additional to make it multiple of 4

        max_len = max_len + right_pad + left_pad
        padded_video = [torch.cat(
            (
                vid[0][None].expand(left_pad, -1, -1, -1),
                vid,
                vid[-1][None].expand(max_len-len(vid)-left_pad, -1, -1, -1),
            )
            , dim=0)
            for vid in video
        ]
        padded_video = torch.stack(padded_video)
        padded_label = []
        for lab in label:
            padded_label.extend(lab)
        return padded_video, padded_label

    def __len__(self):
        return len(self.inputs_list) - 1   # -1 is to avoid the first prefix entry
        # TODO : check if it would be different for testing data

    def test_output(self, index):
        '''
        this function is just implemented to test ie. debugging

        :param index:
        :return:
        '''

        fi = self.inputs_list[index]  # input_list is the dict // its printed.. see
        path = 'dataset/phoenix-2014-multisigner/features/fullFrame-256x256px/' + fi['folder']
        images_path_list = sorted(glob.glob(path))
        label_list = []
        for phase in fi['label'].split(" "):
            if phase == '':
                continue
            if phase in self.dict.keys():
                label_list.append(self.dict[phase][0])

        sign_images = torch.stack([(transform(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))) for img_path in images_path_list])
        return sign_images, label_list

