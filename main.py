import os
import torch.optim as optim
from torch.utils.data import DataLoader

from basicfunc import easyprint
from seq_scripts import seq_train
from sign_network import Signmodel
from utility.parameters import get_parser
from utility.device import GpuDataParallel
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import yaml
import faulthandler
import numpy as np
from slr_network import SLRModel
faulthandler.enable()
from dataloader import SignDataset
import matplotlib.pyplot as plt
from sign_network import *

class Processor:
    def __init__(self, arg):
        self.arg = arg
        self.device = GpuDataParallel()
        self.dataset = {}
        self.data_loader = {}

        # gloss dict we get after preprocessing in which each word has its associated integer_index and occurance, its a list [uniq_index, occourance]
        # dataset_info is specifically being loaded from main method
        self.gloss_dict = np.load(self.arg.dataset_info['dict_path'], allow_pickle=True).item()

        self.arg.model_args['num_classes'] = len(self.gloss_dict) + 1
        easyprint('number of classes', arg.model_args['num_classes'])

        self.model, self.optimizer = self.loading()


    def loading(self):
        self.device.set_device(self.arg.device)
        print("Loading model")

        # TODO: model set here
        model = Signmodel(self.arg.model_args['num_classes'])
        # model = LSTMVideoClassifier(output_size= self.arg.model_args['num_classes'])
        # easyprint('the output has no_of features -', self.arg.model_args['num_classes'])

        optimizer = optim.Adam(
            model.parameters(),   # training all weights, try without including the resnet18 weights
            lr=self.arg.optimizer_args['base_lr'],  # TODO: difference between weight_decay and lr
            weight_decay=self.arg.optimizer_args['weight_decay']
        )
        print('returning model/ opitmizer')
        return model, optimizer


    def start(self):
        '''
            if train then:
                import the SRL model
                run seqTrain method : parameters passed (model, gloss_dict, data_loader, )
            if test:
                get the model created and set the eval mode on
                run seqEval method : parameters passed (model, gloss_dict, dataloader, )

            train model
        :return:
        '''

        choice = input('Do you wish to 1)Train or 2)Eval : ')
        if choice == '1':
            self.train_model()
        else:
            self.eval_model()


        # TODO : load data in required format
        # TODO : implement train function complete
        # TODO : implement eval func

    def train_model(self):
        print('training model...')
        self.model.train()
        dataset_train = SignDataset('train')
        dataloader = DataLoader(dataset_train, shuffle=False, batch_size=1)
        for epoch in range(self.arg.num_epoch):
            incured_loss = seq_train(dataloader, self.model, self.optimizer, self.arg.model_args['num_classes'])
            x = [i for i in range(len(incured_loss))]
            plt.plot(x, incured_loss)
            plt.xlabel('each_batch')
            plt.ylabel('loss currently')
            plt.title('Refrence Liss func')
            plt.show()

    def eval_model(self):
        print('evaluating model...')
        self.model.eval()


if __name__ == '__main__':
    sparser = get_parser()
    p = sparser.parse_args()   # returns a argparse.ArgumentParser class
    p.config = "configs\\baseline.yaml"
    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)

        sparser.set_defaults(**default_arg)

    args = sparser.parse_args()

    args_dict = vars(args)

    print('-------------------------------- printing the set arg values ----------------------------------------')
    for key, value in args_dict.items():
        print(f"{key}: {value}")
    print('-----------------------------------------------------------------------------------------------------')

    with open(f"./configs/{args.dataset}.yaml", 'r') as f:
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)

    processor = Processor(args)
    processor.start()
    print("All finished")