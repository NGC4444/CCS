
from distutils.command.config import config
from typing import Sequence
import torch
from torch import nn
import numpy as np
from pycox.models import utils
import pandas as pd
import torchtuples as tt
import pdb
import argparse
import torch.nn.functional as F
from .modeling_bert import BaseModel, BertEmbeddings, BertEncoder, BertCLS, BertCLSMulti
from .utils import pad_col
from .config import STConfig
from .counterfactual_gan import CounterfactualTimeGAN
from .dataloader import *
from sklearn.preprocessing import OneHotEncoder
from torch.optim import Adam
import pandas as pd
from .dataloader import load_data, get_dataloaders
from .classifier import LSTMClassifier, BidirectionalLSTMClassifier
from .utils_cf import test_classifier, save_model
from .train_classifier import train as cls_train 
from lifelines import CoxPHFitter
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
def split_target_and_input(X, y, target_class):
    """Split input data and labels into queries and targets.

    Args:
        X: Data
        y: Labels
        target_class (int): Desired target class for counterfactuals
    """
    X_target_samples = X[np.argmax(y, axis=1) == target_class]
    y_target_samples = y[np.argmax(y, axis=1) == target_class]

    X_generator_input = X[np.argmax(y, axis=1) != target_class]
    y_generator_input = y[np.argmax(y, axis=1) != target_class]

    return X_target_samples, y_target_samples, X_generator_input, y_generator_input

class Counterfactual():
    def __init__(self) -> None:
        self.counterfactorGAN = CounterfactualTimeGAN()
    def init_argparse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--num_epochs", type=int, default=30) 
        parser.add_argument("--target_class", type=int, default=1)
        parser.add_argument("--max_batches", type=int, default=500)
        parser.add_argument("--dataset", type=str, default="simulate", help="motionsense, simulated")
        parser.add_argument("--batchsize", type=int, default=8)
        parser.add_argument("--dislr", type=float, default=0.001) #dislr
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--save_indicator", type=bool, default=False, help="False or True")
        parser.add_argument("--lambda1", type=float, default=1.0, help="Weight of adversarial loss")
        parser.add_argument("--lambda2", type=float, default=1.0, help="Weight of classification loss")
        parser.add_argument("--lambda3", type=float, default=4.0, help="Weight of similarity loss")
        parser.add_argument("--lambda4", type=float, default=2.0, help="Weight of sparsity loss")
        parser.add_argument("--lambda5", type=float, default=1.0, help="Weight of jerk loss")
        '''
        parser.add_argument("--save_indicator", type=bool, default=False, help="False or True")
        parser.add_argument("--lambda1", type=float, default=1.0, help="Weight of adversarial loss")
        parser.add_argument("--lambda2", type=float, default=1.0, help="Weight of classification loss")
        parser.add_argument("--lambda3", type=float, default=4.0, help="Weight of similarity loss")
        parser.add_argument("--lambda4", type=float, default=2.0, help="Weight of sparsity loss")
        parser.add_argument("--lambda5", type=float, default=1.0, help="Weight of jerk loss")
        '''
        parser.add_argument("--freeze_features", type=list, default=[])
        parser.add_argument("--seed", type=int, default=123, help='random seed for splitting data')
        parser.add_argument("--num_reps", type=int, default=1, help='number of repetitions of experiments')
        parser.add_argument("--max_iter", type=int, default=10)
        parser.add_argument("--init_lambda", type=float, default=1.0)
        parser.add_argument("--approach", type=str, default="sparce")
        parser.add_argument("--save", type=bool, default=True, help="save experiment file, originals and cfs")
        parser.add_argument("--max_lambda_steps", type=int, default=5)
        parser.add_argument("--lambda_increase", type=float, default=0.001)

        return parser
    def tranin_cf_model(self):
        parser = self.init_argparse()
        args = parser.parse_args()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seeds = 2022
        testdoc = []
        home = os.path.expanduser('~')
        pathToProject = os.getcwd()
        #数据集
        dataset = 'pbc2'
        pathToData = os.path.join(pathToProject, 'data', dataset) 
        fileFormat = '.npy'
        batchsize = 32
        device = "cpu"

        replicate_labels_indicator = False # use for many-to-many classification
        bidirectional_indicator = False

        # print(f"Dataset: {dataset}")
        #导入数据
        (X_train, y_train,t_train), (X_val, y_val,t_val), (X_test, y_test,t_test) = load_data(pathToData, fileFormat, replicate_labels_indicator=replicate_labels_indicator)
        dic = np.concatenate([X_train.reshape(-1,25), y_train.reshape(-1,1), t_train.reshape(-1,1)],1)
        data_df = pd.DataFrame(dic)
        data_df.columns = [str(i) for i in range(25)] + ['E',"T"]
        data_df=data_df.dropna(axis=0)
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(data_df,duration_col='T',event_col='E',show_progress=True)
        print(cph.print_summary())
        
        # cph = CoxPHSurvivalAnalysis()
        # df = pd.read_csv('dsm\datasets\pbc2_single.csv',encoding='gbk')
        # cph.fit(df=df[['drug', 'sex', 'ascites', 'hepatomegaly',
        #           'spiders', 'edema', 'histologic','serBilir', 'serChol', 'albumin', 'alkaline',
        #           'SGOT', 'platelets', 'prothrombin','years','label']],duration_col='years',event_col='label',show_progress=True)
        # cph.print_summary()
        X_train = np.nan_to_num(np.concatenate([X_train,X_test],axis=0))
        X_val = np.nan_to_num(X_val)
        X_test = np.nan_to_num(X_test)
        
        y_train = np.concatenate([y_train,y_test],axis=0)
        y_train = y_train[:,0,:]
        y_val = y_val[:,0,:]
        y_test = y_test[:,0,:]
        enc = OneHotEncoder()
        y_train=enc.fit_transform(y_train)
        y_train=y_train.toarray()
        y_val=enc.fit_transform(y_val)
        y_val=y_val.toarray()
        y_test=enc.fit_transform(y_test)
        y_test=y_test.toarray()

        train_dl, val_dl, test_dl = get_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batchsize)
        X_target_samples, y_target_samples, X_generator_input, y_generator_input = split_target_and_input(X_train,y_train,0)
        generator_inp = get_dataloader(X_generator_input,y_generator_input,batchsize)   
        target_inp = get_dataloader(X_target_samples,y_target_samples,batchsize,shuffle=True)
        print()
        for rep in range(args.num_reps):

            args.seed = 2022
            # print(f'----------------------------- Repetition {rep} / {args.num_reps}, Seed: {args.seed} -----------------------------')


            # X_train_generator_input, train_dl_real_samples, train_dl_generator_input, test_dl_real_samples, test_dl_generator_input, train_max_samples, train_max_batches, test_max_samples, test_max_batches = prepare_counterfactual_data(args)

            # model and training parameters
            num_features = X_train.shape[2]

            # model = CounterfactualTimeGAN()

            #训练分类器
            self.train_classifier(X_train, y_train, X_val, y_val, X_test, y_test)
            print('--- build gan model ----')
            self.counterfactorGAN.build_model(args=args, device=device, num_features=num_features, bidirectional=True, \
                hidden_dim_generator=256, layer_dim_generator=2, hidden_dim_discriminator=128, layer_dim_discriminator=1,\
                    classifier_model_name=self.classifier)
            #训练GAN模型--生成器
            self.counterfactorGAN.train(train_dl=generator_inp,target_input=target_inp,survival_model=cph)
            #测试
            testdoc = self.counterfactorGAN.test(test_dl=generator_inp, tgt_data=target_inp, testdoc=testdoc)
    def train_classifier(self,X_train, y_train, X_val, y_val, X_test, y_test):
        # load data
        home = os.path.expanduser('~')
        pathToProject = os.getcwd()
        dataset = 'simulate'
        pathToData = os.path.join(pathToProject, 'data', dataset) 
        fileFormat = '.npy'
        batchsize = 32
        device = "cpu"

        replicate_labels_indicator = False # use for many-to-many classification
        bidirectional_indicator = False

        # # print(f"Dataset: {dataset}")
        # (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(pathToData, fileFormat, replicate_labels_indicator=replicate_labels_indicator)
        # X_train = np.concatenate([X_train,X_test],axis=0)
        # y_train = np.concatenate([y_train,y_test],axis=0)
        # y_train = y_train[:,0,:]
        # y_val = y_val[:,0,:]
        # y_test = y_test[:,0,:]
        # enc = OneHotEncoder()
        # y_train=enc.fit_transform(y_train)
        # y_train=y_train.toarray()
        # y_val=enc.fit_transform(y_val)
        # y_val=y_val.toarray()
        # y_test=enc.fit_transform(y_test)
        # y_test=y_test.toarray()


        # # reshape y
        # y_train = y_train[:,0,:]
        # y_val = y_val[:,0,:]
        # y_test = y_test[:,0,:]
        # print(X_train.shape)
        # print(y_train.shape)

        train_dl, val_dl, test_dl = get_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batchsize)

        num_features = X_train.shape[2]
        num_timesteps = X_train.shape[1]
        if replicate_labels_indicator:
            output_dim = y_train.shape[2]
        else:
            output_dim = y_train.shape[1] # number of classes, 3 or 6

        hidden_dim = 32
        layer_dim = 2

        # instantiate model
        if bidirectional_indicator:
            self.classifier = BidirectionalLSTMClassifier(num_features, hidden_dim, layer_dim, output_dim, replicate_labels_indicator)
        else:
            self.classifier = LSTMClassifier(num_features, hidden_dim, layer_dim, output_dim, replicate_labels_indicator)

        self.classifier = self.classifier.to(device)
        # print(self.classifier)

        # define loss function and optimizer
        num_epochs = 30
        INIT_LR = 1e-2
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(self.classifier.parameters(), lr=INIT_LR)
        cls_train(num_epochs, train_dl, val_dl, self.classifier, loss_fn, optimizer)
