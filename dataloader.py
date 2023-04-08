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



        # print('shape before -- ', sign_images.shape)
        # '''
        #     1) the shape before was [104, 256, 256, 3]
        #     2) now i need to permute to bring 3 in front rearrange it to [3, 104, 256, 256]
        #     3) i need to merge the images as a single dimensional array ie [3, 104, 256*256]
        # '''
        # sign_images = sign_images.permute(3, 0, 1, 2)
        # print('after permutation -- ', sign_images.shape)
        # per_shape = sign_images.shape
        # sign_images = sign_images.view(per_shape[0], per_shape[1], -1)
        # print('after merging the dimension -- ', sign_images.shape)
        #
        # # sign_images = sign_images.squeeze(0)  # Remove the batch dimension
        # # sign_images = sign_images.permute(3, 0, 1, 2)  # Move the channel dimension to the front
        # # print('shape afterwards', sign_images.shape)  # Should print [3, 214, 256, 256]
        #
        # print('from __getitem___ : ', type(sign_images))
        # # normalized_img = [sign_pic/255.0 for sign_pic in sign_images]  # this is list of all images for that sign
        return sign_images, label_list


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

