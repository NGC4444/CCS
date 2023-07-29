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

from .dataloader import load_data, get_dataloaders
from .classifier import LSTMClassifier, BidirectionalLSTMClassifier
from .utils_cf import test_classifier, save_model
from .train_classifier import train as cls_train 
import sys
# sys.path.append("..")
# import train


class SurvTraceMulti(BaseModel):
    '''SurvTRACE model for competing events survival analysis.
    '''
    def __init__(self,  config: STConfig):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.cls = BertCLSMulti(config)
        self.config = config
        self.init_weights()
        # self.duration_index = config['duration_index']
        self.use_gpu = True
        # self.counterfactorGAN = cf_model
        # self.tranin_cf_model()
        

    @property
    def duration_index(self):
        """
        Array of durations that defines the discrete times. This is used to set the index
        of the DataFrame in `predict_surv_df`.
        
        Returns:
            np.array -- Duration index.
        """
        return self._duration_index

    @duration_index.setter
    def duration_index(self, val):
        self._duration_index = val

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
    
    # def init_argparse(self):
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument("--num_epochs", type=int, default=20) 
    #     parser.add_argument("--target_class", type=int, default=1)
    #     parser.add_argument("--max_batches", type=int, default=500)
    #     parser.add_argument("--dataset", type=str, default="simulate", help="motionsense, simulated")
    #     parser.add_argument("--batchsize", type=int, default=32)
    #     parser.add_argument("--dislr", type=float, default=0.001) #dislr
    #     parser.add_argument("--lr", type=float, default=0.001)
    #     parser.add_argument("--save_indicator", type=bool, default=False, help="False or True")
    #     parser.add_argument("--lambda1", type=float, default=1.0, help="Weight of adversarial loss")
    #     parser.add_argument("--lambda2", type=float, default=2.0, help="Weight of classification loss")
    #     parser.add_argument("--lambda3", type=float, default=1.0, help="Weight of similarity loss")
    #     parser.add_argument("--lambda4", type=float, default=1.0, help="Weight of sparsity loss")
    #     parser.add_argument("--lambda5", type=float, default=1.0, help="Weight of jerk loss")
    #     parser.add_argument("--freeze_features", type=list, default=[])
    #     parser.add_argument("--seed", type=int, default=123, help='random seed for splitting data')
    #     parser.add_argument("--num_reps", type=int, default=1, help='number of repetitions of experiments')
    #     parser.add_argument("--max_iter", type=int, default=10)
    #     parser.add_argument("--init_lambda", type=float, default=1.0)
    #     parser.add_argument("--approach", type=str, default="sparce")
    #     parser.add_argument("--save", type=bool, default=True, help="save experiment file, originals and cfs")
    #     parser.add_argument("--max_lambda_steps", type=int, default=5)
    #     parser.add_argument("--lambda_increase", type=float, default=0.001)

    #     return parser
    # def tranin_cf_model(self):
    #     parser = self.init_argparse()
    #     args = parser.parse_args()
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     seeds = 2022
    #     testdoc = []
    #     home = os.path.expanduser('~')
    #     pathToProject = os.getcwd()
    #     dataset = 'simulate'
    #     pathToData = os.path.join(pathToProject, 'data', dataset) 
    #     fileFormat = '.npy'
    #     batchsize = 32
    #     device = "cpu"

    #     replicate_labels_indicator = False # use for many-to-many classification
    #     bidirectional_indicator = False

    #     # print(f"Dataset: {dataset}")
    #     (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(pathToData, fileFormat, replicate_labels_indicator=replicate_labels_indicator)
    #     X_train = np.concatenate([X_train,X_test],axis=0)
    #     y_train = np.concatenate([y_train,y_test],axis=0)
    #     y_train = y_train[:,0,:]
    #     y_val = y_val[:,0,:]
    #     y_test = y_test[:,0,:]
    #     enc = OneHotEncoder()
    #     y_train=enc.fit_transform(y_train)
    #     y_train=y_train.toarray()
    #     y_val=enc.fit_transform(y_val)
    #     y_val=y_val.toarray()
    #     y_test=enc.fit_transform(y_test)
    #     y_test=y_test.toarray()
    #     train_dl, val_dl, test_dl = get_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batchsize)
    #     for rep in range(args.num_reps):

    #         args.seed = 2022
    #         # print(f'----------------------------- Repetition {rep} / {args.num_reps}, Seed: {args.seed} -----------------------------')


    #         # X_train_generator_input, train_dl_real_samples, train_dl_generator_input, test_dl_real_samples, test_dl_generator_input, train_max_samples, train_max_batches, test_max_samples, test_max_batches = prepare_counterfactual_data(args)

    #         # model and training parameters
    #         num_features = X_train.shape[2]

    #         # model = CounterfactualTimeGAN()
    #         self.train_classifier(X_train, y_train, X_val, y_val, X_test, y_test)
    #         self.counterfactorGAN.build_model(args=args, device=device, num_features=num_features, bidirectional=True, \
    #             hidden_dim_generator=256, layer_dim_generator=2, hidden_dim_discriminator=128, layer_dim_discriminator=1,\
    #                  classifier_model_name=self.classifier)
    #         self.counterfactorGAN.train(train_dl=train_dl)
    #         testdoc = self.counterfactorGAN.test(test_dl=train_dl, testdoc=testdoc)
    # def train_classifier(self,X_train, y_train, X_val, y_val, X_test, y_test):
    #     # load data
    #     home = os.path.expanduser('~')
    #     pathToProject = os.getcwd()
    #     dataset = 'simulate'
    #     pathToData = os.path.join(pathToProject, 'data', dataset) 
    #     fileFormat = '.npy'
    #     batchsize = 32
    #     device = "cpu"

    #     replicate_labels_indicator = False # use for many-to-many classification
    #     bidirectional_indicator = False

    #     # # print(f"Dataset: {dataset}")
    #     # (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(pathToData, fileFormat, replicate_labels_indicator=replicate_labels_indicator)
    #     # X_train = np.concatenate([X_train,X_test],axis=0)
    #     # y_train = np.concatenate([y_train,y_test],axis=0)
    #     # y_train = y_train[:,0,:]
    #     # y_val = y_val[:,0,:]
    #     # y_test = y_test[:,0,:]
    #     # enc = OneHotEncoder()
    #     # y_train=enc.fit_transform(y_train)
    #     # y_train=y_train.toarray()
    #     # y_val=enc.fit_transform(y_val)
    #     # y_val=y_val.toarray()
    #     # y_test=enc.fit_transform(y_test)
    #     # y_test=y_test.toarray()


    #     # # reshape y
    #     # y_train = y_train[:,0,:]
    #     # y_val = y_val[:,0,:]
    #     # y_test = y_test[:,0,:]
    #     # print(X_train.shape)
    #     # print(y_train.shape)

    #     train_dl, val_dl, test_dl = get_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batchsize)

    #     num_features = X_train.shape[2]
    #     num_timesteps = X_train.shape[1]
    #     if replicate_labels_indicator:
    #         output_dim = y_train.shape[2]
    #     else:
    #         output_dim = y_train.shape[1] # number of classes, 3 or 6

    #     hidden_dim = 32
    #     layer_dim = 2

    #     # instantiate model
    #     if bidirectional_indicator:
    #         self.classifier = BidirectionalLSTMClassifier(num_features, hidden_dim, layer_dim, output_dim, replicate_labels_indicator)
    #     else:
    #         self.classifier = LSTMClassifier(num_features, hidden_dim, layer_dim, output_dim, replicate_labels_indicator)

    #     self.classifier = self.classifier.to(device)
    #     # print(self.classifier)

    #     # define loss function and optimizer
    #     num_epochs = 30
    #     INIT_LR = 1e-2
    #     loss_fn = nn.CrossEntropyLoss()
    #     optimizer = Adam(self.classifier.parameters(), lr=INIT_LR)
    #     cls_train(num_epochs, train_dl, val_dl, self.classifier, loss_fn, optimizer)

    def forward(
        self,
        input_ids=None,
        input_nums=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        event=0, # output the prediction for different competing events
        epoch=0,
        cf_model_global=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            if len(input_shape) == 2:
                batch_size, seq_length = input_shape
            else:
                batch_size, seq_length , feature_num = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        #这部分Attention要改，先横向编码，再纵向
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        # if self.config.hidden_size == 25:
        #     embedding_output = input_ids
        # else:
        input_fc = input_ids.clone().float()
        # cf_model_global.generator
        Counterfactual_input = cf_model_global.generator(input_ids)

        embedding_output, cf_embbedding_output = self.embeddings(
            input_ids=input_ids,
            cf_input_ids = Counterfactual_input,
            input_x_num=input_nums,
            inputs_embeds=inputs_embeds,
        )

        # embedding_output = input_ids # torch.cat([input_ids, input_nums], axis=1)
        # out = (batch, 16,16, 16):batch-size,队列数， 特征数， 隐藏向量长度
        # embedding_output = embedding_output.permute(1, 0, 2, 3)

        ##### 0
        encoder_outputs,cf_encoder_output = self.encoder(embedding_output, cf_embbedding_output,epoch=epoch)
        ##### 1
        # encoder_outputs = torch.unsqueeze(self.encoder(embedding_output[0])[0],0)
        # for i in range(1,16):
        #     tmp = self.encoder(embedding_output[i])
        #     encoder_outputs = torch.cat([encoder_outputs, torch.unsqueeze(tmp[0], 0)] , dim=0)

        # encoder_outputs = self.encoder(embedding_output)
        # sequence_output = encoder_outputs.permute(1, 0, 2, 3)
        sequence_output = encoder_outputs[0]
        fake_sequence_output = cf_encoder_output[0]
        predict_logits = self.cls(sequence_output, event=event)
        fake_predict_logits = self.cls(fake_sequence_output,event=event)
        return sequence_output, predict_logits, fake_sequence_output, fake_predict_logits

    def predict(self, x_input, batch_size=None, event=0):
        if not isinstance(x_input, torch.Tensor):
            x_input_cat = x_input[:,:, :self.config.num_categorical_feature]
            x_input_num = x_input[:,:, self.config.num_categorical_feature:]
            x_num = torch.tensor(x_input_num).float()
            x_cat = torch.tensor(x_input_cat).float()
        else:
            x_cat = x_input[:,:, :self.config.num_categorical_feature].float()
            x_num = x_input[:,:, self.config.num_categorical_feature:].float()
        
        if self.use_gpu:
            x_num = x_num.cuda()
            x_cat = x_cat.cuda()

        num_sample = len(x_num)
        self.eval()
        with torch.no_grad():
            if batch_size is None:
                    preds = self.forward(x_cat, x_num, event=event)[1]
            else:
                preds = []
                num_batch = int(np.ceil(num_sample / batch_size))
                for idx in range(num_batch):
                    batch_x_num = x_num[idx*batch_size:(idx+1)*batch_size]
                    batch_x_cat = x_cat[idx*batch_size:(idx+1)*batch_size]
                    batch_pred = self.forward(batch_x_cat, batch_x_num,event=event)
                    preds.append(batch_pred[1])
                preds = torch.cat(preds)
        return preds

    def predict_hazard(self, input_ids, batch_size=None, event=0):
        preds = self.predict(input_ids, batch_size, event=event)
        hazard = F.softplus(preds)
        hazard = pad_col(hazard, where="start")
        return hazard
    
    def predict_risk(self, input_ids, batch_size=None, event=0):
        surv = self.predict_surv(input_ids, batch_size, event=event)
        return 1- surv

    def predict_surv(self, input_ids, batch_size=None, event=0):
        hazard = self.predict_hazard(input_ids, batch_size, event=event)
        surv = hazard.cumsum(1).mul(-1).exp()
        return surv
    
    def predict_surv_df(self, input_ids, batch_size=None, event=0):
        surv = self.predict_surv(input_ids, batch_size, event=event)
        return pd.DataFrame(surv.to("cpu").numpy().T, self.duration_index)


class SurvTraceSingle(BaseModel):
    '''survtrace used for single event survival analysis
    '''
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.cls = BertCLS(config)
        self.config = config
        self.init_weights()
        self.duration_index = config['duration_index']
        self.use_gpu = True

    @property
    def duration_index(self):
        """
        Array of durations that defines the discrete times. This is used to set the index
        of the DataFrame in `predict_surv_df`.
        
        Returns:
            np.array -- Duration index.
        """
        return self._duration_index

    @duration_index.setter
    def duration_index(self, val):
        self._duration_index = val

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids=None,
        input_nums=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            input_x_num=input_nums,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(embedding_output)
        sequence_output = encoder_outputs[1]
        
        # do pooling
        # sequence_output = (encoder_outputs[1][-2] + encoder_outputs[1][-1]).mean(dim=1)

        predict_logits = self.cls(encoder_outputs[0])

        return sequence_output, predict_logits

    def predict(self, x_input, batch_size=None):
        if not isinstance(x_input, torch.Tensor):
            x_input_cat = x_input.iloc[:, :self.config.num_categorical_feature]
            x_input_num = x_input.iloc[:, self.config.num_categorical_feature:]
            x_num = torch.tensor(x_input_num.values).float()
            x_cat = torch.tensor(x_input_cat.values).float()
        else:
            x_cat = x_input[:, :self.config.num_categorical_feature].float()
            x_num = x_input[:, self.config.num_categorical_feature:].float()
        
        if self.use_gpu:
            x_num = x_num.cuda()
            x_cat = x_cat.cuda()

        num_sample = len(x_num)
        self.eval()
        with torch.no_grad():
            if batch_size is None:
                    preds = self.forward(x_cat, x_num)[1]
            else:
                preds = []
                num_batch = int(np.ceil(num_sample / batch_size))
                for idx in range(num_batch):
                    batch_x_num = x_num[idx*batch_size:(idx+1)*batch_size]
                    batch_x_cat = x_cat[idx*batch_size:(idx+1)*batch_size]
                    batch_pred = self.forward(batch_x_cat,batch_x_num)
                    preds.append(batch_pred[1])
                preds = torch.cat(preds)
        return preds

    def predict_hazard(self, input_ids, batch_size=None):
        preds = self.predict(input_ids, batch_size)
        hazard = F.softplus(preds)
        hazard = pad_col(hazard, where="start")
        return hazard
    
    def predict_risk(self, input_ids, batch_size=None):
        surv = self.predict_surv(input_ids, batch_size)
        return 1- surv

    def predict_surv(self, input_ids, batch_size=None, epsilon=1e-7):
        hazard = self.predict_hazard(input_ids, batch_size)
        # surv = (1 - hazard).add(epsilon).log().cumsum(1).exp()
        surv = hazard.cumsum(1).mul(-1).exp()
        return surv
    
    def predict_surv_df(self, input_ids, batch_size=None):
        surv = self.predict_surv(input_ids, batch_size)
        return pd.DataFrame(surv.to("cpu").numpy().T, self.duration_index)