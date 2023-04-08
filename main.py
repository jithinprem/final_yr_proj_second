import os
import torch.optim as optim
from torch.utils.data import DataLoader

from basicfunc import easyprint
from seq_scripts import seq_train, seq_test
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
from sign_network import *
from fast_ctc_decode import beam_search, viterbi_search

# from torch.utils import tensorboard

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
        easyprint('gloss dict ', list(self.gloss_dict.keys()))
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
        # optimizer.sheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)
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

        choice = input('Do you wish to 1)Train or 2)single test or 3)evaluate : ')
        if choice == '1':
            all_epoch_loss = self.train_model()
            # plot the loss in graph
        if choice == '2':
            print("choice 2")
            # test model, give any row number from the csv file
            modelno = input('enter model no ')
            PATH = f'model'+ modelno +'.pt'
            trained_model = torch.load(PATH)['model_state_dict']
            ind_no = input('enter the index_no : ')
            # fetch data according to index and run on the model
            dataset_train = SignDataset('train')
            vid, lab = dataset_train.test_output(ind_no)
            lab_pred = trained_model(vid)
            # the predicted is a series of probablity
            seq, path = viterbi_search(lab_pred, list(self.gloss_dict.keys()) + ['_'])
            print(seq)
        if choice == '3':
            self.eval_model()


        # TODO : load data in required format
        # TODO : implement train function complete
        # TODO : implement eval func

    def train_model(self):
        print('training model...')
        epoch = 0

        p_use = input('you wanna use the pretrained model ? ')
        if p_use == '1' or p_use =='y' or p_use =='Y':
            checkpoint = torch.load('model20.pt')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']

        self.model.train()
        dataset_train = SignDataset('train')
        dataloader = DataLoader(dataset_train, shuffle=False, batch_size=2)
        # writer = SummaryWriter()
        all_epoch_loss = []
        for epoch in range(epoch, self.arg.num_epoch+1):
            incured_loss = seq_train(dataloader, self.model, self.optimizer, self.arg.model_args['num_classes'])
            if epoch % 5 == 0:
                path = "model"+str(epoch)+".pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    # 'scheduler_state_dict': self.optimizer.scheduler.state_dict(),
                },path)

            all_epoch_loss.append(incured_loss)
        return all_epoch_loss

    def pad_vid(self, batch):
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



    def eval_model(self):
        print('evaulating model')
        self.model.eval()

        '''
            using a pretrained model for evaulation
        '''
        model_no = input('enter model number for evaulation : ')
        # Load the saved model from disk
        checkpoint = torch.load(f'model{model_no}.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        # Load the optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


        # setting dataset for eval
        dataset_test = SignDataset('test')
        dataloader = DataLoader(dataset_test, shuffle=False, batch_size=2, collate_fn=self.pad_vid)

        # testing
        incured_loss = seq_test(dataloader, self.model, self.gloss_dict)

        # loss
        print(incured_loss)



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