import torch
import numpy as np
import pandas as pd
import os
import math
import warnings
import itertools
import numbers
import torch.utils.data as utils
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
# import data


class GRUHM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, x_mean=0,\
                 bias=True, batch_first=False, bidirectional=False, dropout_type='mloss', dropout=0):
        super(GRUHM, self).__init__()

        self.rnn = torch.nn.LSTM(input_size, 100, 2)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.zeros = torch.autograd.Variable(torch.zeros(input_size)).cuda()
        self.x_mean = torch.autograd.Variable(torch.tensor(x_mean)).cuda()
        self.bias = bias
        self.batch_first = batch_first
        self.dropout_type = dropout_type
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
#        self.mode = mode
        self.mode = 'GRU'

        self.linear1 = torch.nn.Linear(100,1)
        self.linear2 = torch.nn.Linear(input_size,input_size)
        self.linear3 = torch.nn.Linear(self.hidden_size,1)   #183
        self.act2 = torch.nn.LeakyReLU()


        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))
        
        ################################
        gate_size = 1 # not used
        ################################
        
        self._all_weights = []

        '''
        w_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
        w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
        b_ih = Parameter(torch.Tensor(gate_size))
        b_hh = Parameter(torch.Tensor(gate_size))
        layer_params = (w_ih, w_hh, b_ih, b_hh)
        '''
        # decay rates gamma
        w_dg_x = torch.nn.Parameter(torch.Tensor(input_size))
        w_dg_h = torch.nn.Parameter(torch.Tensor(hidden_size))
        
        w_beita = torch.nn.Parameter(torch.Tensor(hidden_size))
        # z
        w_xz = torch.nn.Parameter(torch.Tensor(input_size))
        w_hz = torch.nn.Parameter(torch.Tensor(hidden_size))
        w_mz = torch.nn.Parameter(torch.Tensor(input_size))
        p_z = torch.nn.Parameter(torch.Tensor(input_size))
#        w_xz = torch.Tensor(input_size)
#        w_hz = torch.Tensor(hidden_size)
#        w_mz = torch.Tensor(input_size)

        # r
        w_xr = torch.nn.Parameter(torch.Tensor(input_size))
        w_hr = torch.nn.Parameter(torch.Tensor(hidden_size))
        w_mr = torch.nn.Parameter(torch.Tensor(input_size))
        p_r = torch.nn.Parameter(torch.Tensor(input_size))

        # h_tilde
        w_xh = torch.nn.Parameter(torch.Tensor(input_size))
        w_hh = torch.nn.Parameter(torch.Tensor(hidden_size))
        w_mh = torch.nn.Parameter(torch.Tensor(input_size))
        p_h = torch.nn.Parameter(torch.Tensor(input_size))
        
        # y (output)
        # w_hy = torch.nn.Parameter(torch.Tensor(output_size, hidden_size))

        # y_output_lstm
        w_lstm_y = torch.nn.Parameter(torch.Tensor(output_size, num_layers))


        # bias
        b_dg_x = torch.nn.Parameter(torch.Tensor(hidden_size))
        b_dg_h = torch.nn.Parameter(torch.Tensor(hidden_size))
        b_z = torch.nn.Parameter(torch.Tensor(hidden_size))
        b_r = torch.nn.Parameter(torch.Tensor(hidden_size))
        b_h = torch.nn.Parameter(torch.Tensor(hidden_size))
        # b_y = torch.nn.Parameter(torch.Tensor(output_size))
        b_lstm = torch.nn.Parameter(torch.Tensor(output_size))
        b_beita = torch.nn.Parameter(torch.Tensor(hidden_size))
        

        layer_params = (w_dg_x, w_dg_h,\
                        w_xz, w_hz, w_mz,\
                        w_xr, w_hr, w_mr,\
                        w_xh, w_hh, w_mh,\
                        p_z, p_r, p_h,\
                        # w_hy,\
                            w_lstm_y,w_beita,\
                        b_dg_x, b_dg_h, b_z, b_r, b_h, b_beita,\
                            # b_y,\
                            b_lstm)

        param_names = ['weight_dg_x', 'weight_dg_h',\
                       'weight_xz', 'weight_hz','weight_mz',\
                       'weight_xr', 'weight_hr','weight_mr',\
                       'weight_xh', 'weight_hh','weight_mh',\
                       'weight_pz', 'weight_pr','weight_ph',\
                    #    'weight_hy',\
                        'w_lstm_y' , 'weight_beita'  ]
        if bias:
            param_names += ['bias_dg_x', 'bias_dg_h',\
                            'bias_z',\
                            'bias_r',\
                            'bias_h',\
                            'bias_beita',\
                            # 'bias_y',\
                            'b_lstm']
        
        for name, param in zip(param_names, layer_params):
            setattr(self, name, param)
        self._all_weights.append(param_names)

        self.flatten_parameters()
        self.reset_parameters()
        
    def flatten_parameters(self):
        """
        Resets parameter data pointer so that they can use faster code paths.
        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        any_param = next(self.parameters()).data
        if not any_param.is_cuda or not torch.backends.cudnn.is_acceptable(any_param):
            return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        all_weights = self._flat_weights
        unique_data_ptrs = set(p.data_ptr() for p in all_weights)
        if len(unique_data_ptrs) != len(all_weights):
            return

        with torch.cuda.device_of(any_param):
            import torch.backends.cudnn.rnn as rnn

            # NB: This is a temporary hack while we still don't have Tensor
            # bindings for ATen functions
            with torch.no_grad():
                # NB: this is an INPLACE function on all_weights, that's why the
                # no_grad() is necessary.
                torch._cudnn_rnn_flatten_weight(
                    all_weights, (4 if self.bias else 2),
                    self.input_size, rnn.get_cudnn_mode(self.mode), self.hidden_size, self.num_layers,
                    self.batch_first, bool(self.bidirectional))

    def _apply(self, fn):
        ret = super(GRUHM, self)._apply(fn)
        # self.flatten_parameters()
        return ret

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def check_forward_args(self, input, hidden, batch_sizes):
        is_input_packed = batch_sizes is not None
        expected_input_dim = 2 if is_input_packed else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

        if is_input_packed:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)

        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)
        
        def check_hidden_size(hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
            if tuple(hx.size()) != expected_hidden_size:
                raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

        if self.mode == 'LSTM':
            check_hidden_size(hidden[0], expected_hidden_size,
                              'Expected hidden[0] size {}, got {}')
            check_hidden_size(hidden[1], expected_hidden_size,
                              'Expected hidden[1] size {}, got {}')
        else:
            check_hidden_size(hidden, expected_hidden_size)
    
    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)
    
    
    def __setstate__(self, d):
        super(GRUHM, self).__setstate__(d)
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        self._all_weights = []

        weights = ['weight_dg_x', 'weight_dg_h',\
                   'weight_xz', 'weight_hz','weight_mz',\
                   'weight_xr', 'weight_hr','weight_mr',\
                   'weight_xh', 'weight_hh','weight_mh',\
                   'weight_pz', 'weight_pr','weight_ph',\
                #    'weight_hy',\
                    'w_lstm_y','weight_beta',\
                   'bias_dg_x', 'bias_dg_h',\
                   'bias_z', 'bias_r', 'bias_h','bias_y','bias_beita', 
                #    'w_lstm_y',
                   'b_lstm']

        if self.bias:
            self._all_weights += [weights]
        else:
            self._all_weights += [weights[:2]]

    @property
    def _flat_weights(self):
        return list(self._parameters.values())

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]
    
    def forward(self, input):
        # input.size = (3, 44 ,858) : num_input or num_hidden, num_layer or step

        # 1,44,238 ---> lstm ---> y
        out_all = torch.zeros((input.shape[0],1))
        for i in range(input.shape[0]):
            input = input.cuda()
            X = torch.squeeze(input[i][0]) # .size = (44 ,238)
            Mask = torch.squeeze(input[i][1]) # .size = (44 ,238)
            Delta = torch.squeeze(input[i][2]) # .size = (44 ,238)
            
            Hidden_State = torch.autograd.Variable(torch.zeros(self.input_size)).cuda()
            
            step_size = X.size(1) # 49
            #print('step size : ', step_size)

            hidden_seq = []

            output = None
            h = Hidden_State

            # decay rates gamma
            w_dg_x = getattr(self, 'weight_dg_x')
            w_dg_h = getattr(self, 'weight_dg_h')
            w_beita = getattr(self, 'weight_beita')

            #z
            w_xz = getattr(self, 'weight_xz')
            w_hz = getattr(self, 'weight_hz')
            w_mz = getattr(self, 'weight_mz')
            p_z = getattr(self, 'weight_pz')

            # r
            w_xr = getattr(self, 'weight_xr')
            w_hr = getattr(self, 'weight_hr')
            w_mr = getattr(self, 'weight_mr')
            p_r = getattr(self, 'weight_pr')

            # h_tilde
            w_xh = getattr(self, 'weight_xh')
            w_hh = getattr(self, 'weight_hh')
            w_mh = getattr(self, 'weight_mh')
            p_h = getattr(self, 'weight_ph')

            # bias
            b_dg_x = getattr(self, 'bias_dg_x')
            b_dg_h = getattr(self, 'bias_dg_h')
            b_z = getattr(self, 'bias_z')
            b_r = getattr(self, 'bias_r')
            b_h = getattr(self, 'bias_h')
            b_beita = getattr(self, 'bias_beita')

            # w_hy = getattr(self, 'weight_hy')
            # b_y = getattr(self, 'bias_y')

            #  w_lstm_y,b_lstm
            w_lstm_y = getattr(self, 'w_lstm_y')
            b_lstm = getattr(self, 'b_lstm')
            
            miu = torch.sum(Mask,axis=1)
            
            for layer in range(self.num_layers):
                
                x = torch.squeeze(X[:,layer:layer+1])
                m = torch.squeeze(Mask[:,layer:layer+1])
                d = torch.squeeze(Delta[:,layer:layer+1])

                beita = torch.exp(-torch.max(self.zeros, (w_beita * miu + b_beita)))
                #(4)
#                gamma_x = torch.exp(-torch.max(self.zeros, (w_dg_x * d + b_dg_x)))
#                gamma_h = torch.exp(-torch.max(self.zeros, (w_dg_h * d + b_dg_h)))
                gamma_x = torch.exp(-torch.max(self.zeros, (w_dg_x * d + b_dg_x)))
                gamma_h = torch.exp(-torch.max(self.zeros, (w_dg_h * d + b_dg_h)))

                #(5)
#                x = m * x + (1 - m) * (gamma_x * x + (1 - gamma_x) * self.x_mean.reshape(178,)) #183 （1，）---> (238,1) * 44
#                x = m * x + (1 - m) * (gamma_x * x + (1 - gamma_x) * self.x_mean.reshape(self.x_mean.shape[0],)) #183 （1，）---> (238,1) * 44
                x = beita * x + (1 - beita) * (gamma_x * x + (1 - gamma_x) * self.x_mean.reshape(self.x_mean.shape[0],)) #183 （1，）---> (238,1) * 44

                #(6)
                if self.dropout == 0:
                    h = gamma_h * h

#                    z = torch.sigmoid((w_xz*x + w_hz*h + w_mz*m + b_z))
#                    r = torch.sigmoid((w_xr*x + w_hr*h + w_mr*m + b_r))
#                    z = torch.relu((w_xz*x + w_hz*h + b_z))
#                    r = torch.relu((w_xr*x + w_hr*h + b_r))
#                    z = torch.sigmoid((w_xz@x + w_hz@h + w_mz@m + p_z * beita + b_z))
#                    r = torch.sigmoid((w_xr@x + w_hr@h + w_mr@m + p_r * beita + b_r))
#                    z = torch.selu((w_xz*x + w_hz*h + w_mz*m + b_z))
#                    r = torch.selu((w_xr*x + w_hr*h + w_mr*m + b_r))
#                    z = torch.elu((w_xz*x + w_hz*h + w_mz*m + b_z),alpha=1.1)
#                    r = torch.elu((w_xr*x + w_hr*h + w_mr*m + b_r),alpha=1.1)
#                    z = torch.sigmoid((w_xz*x + w_hz*h + w_mz*m + p_z * beita + b_z))
#                    r = torch.sigmoid((w_xr*x + w_hr*h + w_mr*m + p_r * beita + b_r))
#                    h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + w_mh*m + p_h * beita+ b_h))
                    z = torch.sigmoid((w_xz*x + w_hz*h + p_z * beita + b_z))
                    r = torch.sigmoid((w_xr*x + w_hr*h + p_r * beita + b_r))
                    h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + p_h * beita+ b_h))
#                    h_tilde = torch.tanh((w_xh@x + w_hh@(r * h) + w_mh@m + p_h * beita+ b_h))
#                    z = torch.sigmoid((w_xz*x + w_hz*h + w_mz*m + b_z))
#                    r = torch.sigmoid((w_xr*x + w_hr*h + w_mr*m + b_r))
#                    h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + w_mh*m + b_h))
                    
#                    print('w_xz in device :',w_xz.device)
#                    print('x in device :',x.device)
#                    z = torch.sigmoid((np.multiply(w_xz,x) + np.multiply(w_hz,h) + np.multiply(w_mz,m) + b_z))
#                    r = torch.sigmoid((np.multiply(w_xr,x) + np.multiply(w_hr,h) + np.multiply(w_mr,m) + b_r))
#                    h_tilde = torch.tanh((np.multiply(w_xh,x) + np.multiply(w_hh,(r * h)) + np.multiply(w_mh,m) + b_h))

                    h = (1 - z) * h + z * h_tilde

                elif self.dropout_type == 'Moon':
                    '''
                    RNNDROP: a novel dropout for rnn in asr(2015)
                    '''
                    h = gamma_h * h

#                    z = torch.sigmoid((w_xz*x + w_hz*h + w_mz*m + b_z))
#                    r = torch.sigmoid((w_xr*x + w_hr*h + w_mr*m + b_r))
#                    z = torch.relu((w_xz*x + w_hz*h + b_z))
#                    r = torch.relu((w_xr*x + w_hr*h + b_r))
#                    z = torch.selu((w_xz*x + w_hz*h + w_mz*m + b_z))
#                    r = torch.selu((w_xr*x + w_hr*h + w_mr*m + b_r))
#                    z = torch.elu((w_xz*x + w_hz*h + w_mz*m + b_z),alpha=1.1)
#                    r = torch.elu((w_xr*x + w_hr*h + w_mr*m + b_r),alpha=1.1)

#                    z = torch.sigmoid((w_xz@x + w_hz@h + w_mz@m + p_z * beita + b_z))
#                    r = torch.sigmoid((w_xr@x + w_hr@h + w_mr@m + p_r * beita + b_r))
#                    h_tilde = torch.tanh((w_xh@x + w_hh@(r * h) + w_mh@m + p_h * beita + b_h))
#                    z = torch.sigmoid((w_xz*x + w_hz*h + w_mz*m + p_z * beita + b_z))
#                    r = torch.sigmoid((w_xr*x + w_hr*h + w_mr*m + p_r * beita + b_r))
#                    h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + w_mh*m + p_h * beita+ b_h))
                    z = torch.sigmoid((w_xz*x + w_hz*h + p_z * beita + b_z))
                    r = torch.sigmoid((w_xr*x + w_hr*h + p_r * beita + b_r))
                    h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + p_h * beita+ b_h))
#                    h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + b_h))
#                    z = torch.sigmoid((w_xz*x + w_hz*h + w_mz*m + b_z))
#                    r = torch.sigmoid((w_xr*x + w_hr*h + w_mr*m + b_r))
#                    h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + w_mh*m + b_h))

#                    z = torch.sigmoid((np.multiply(w_xz,x) + np.multiply(w_hz,h) + np.multiply(w_mz,m) + b_z))
#                    r = torch.sigmoid((np.multiply(w_xr,x) + np.multiply(w_hr,h) + np.multiply(w_mr,m) + b_r))
#                    h_tilde = torch.tanh((np.multiply(w_xh,x) + np.multiply(w_hh,(r * h)) + np.multiply(w_mh,m) + b_h))

                    h = (1 - z) * h + z * h_tilde
                    dropout = torch.nn.Dropout(p=self.dropout)
                    h = dropout(h)

                elif self.dropout_type == 'Gal':
                    '''
                    A Theoretically grounded application of dropout in recurrent neural networks(2015)
                    '''
                    dropout = torch.nn.Dropout(p=self.dropout)
                    h = dropout(h)

                    h = gamma_h * h

#                    z = torch.sigmoid((w_xz*x + w_hz*h + w_mz*m + b_z))
#                    r = torch.sigmoid((w_xr*x + w_hr*h + w_mr*m + b_r))
#                    z = torch.relu((w_xz*x + w_hz*h + b_z))
#                    r = torch.relu((w_xr*x + w_hr*h + b_r))
#                    z = torch.selu((w_xz*x + w_hz*h + w_mz*m + b_z))
#                    r = torch.selu((w_xr*x + w_hr*h + w_mr*m + b_r))
#                    z = torch.elu((w_xz*x + w_hz*h + w_mz*m + b_z),alpha=1.1)
#                    r = torch.elu((w_xr*x + w_hr*h + w_mr*m + b_r),alpha=1.1)
#                    z = torch.sigmoid((w_xz@x + w_hz@h + w_mz@m + p_z * beita + b_z))
#                    r = torch.sigmoid((w_xr@x + w_hr@h + w_mr@m + p_r * beita + b_r))
#                    h_tilde = torch.tanh((w_xh@x + w_hh@(r * h) + w_mh@m + p_h * beita + b_h))
#                    z = torch.sigmoid((w_xz*x + w_hz*h + w_mz*m + p_z * beita + b_z))
#                    r = torch.sigmoid((w_xr*x + w_hr*h + w_mr*m + p_r * beita + b_r))
#                    h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + w_mh*m + p_h * beita+ b_h))
                    z = torch.sigmoid((w_xz*x + w_hz*h + p_z * beita + b_z))
                    r = torch.sigmoid((w_xr*x + w_hr*h + p_r * beita + b_r))
                    h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + p_h * beita+ b_h))
#                    h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + b_h))
#                    z = torch.sigmoid((w_xz*x + w_hz*h + w_mz*m + b_z))
#                    r = torch.sigmoid((w_xr*x + w_hr*h + w_mr*m + b_r))
#                    h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + w_mh*m + b_h))
                    
#                    z = torch.sigmoid((np.multiply(w_xz,x) + np.multiply(w_hz,h) + np.multiply(w_mz,m) + b_z))
#                    r = torch.sigmoid((np.multiply(w_xr,x) + np.multiply(w_hr,h) + np.multiply(w_mr,m) + b_r))
#                    h_tilde = torch.tanh((np.multiply(w_xh,x) + np.multiply(w_hh,(r * h)) + np.multiply(w_mh,m) + b_h))

                    h = (1 - z) * h + z * h_tilde

                elif self.dropout_type == 'mloss':
                    '''
                    recurrent dropout without memory loss arXiv 1603.05118
                    g = h_tilde, p = the probability to not drop a neuron
                    '''

                    h = gamma_h * h

#                    z = torch.sigmoid((w_xz*x + w_hz*h + w_mz*m + b_z))
#                    r = torch.sigmoid((w_xr*x + w_hr*h + w_mr*m + b_r))
#                    z = torch.relu((w_xz*x + w_hz*h + b_z))
#                    r = torch.relu((w_xr*x + w_hr*h + b_r))
#                    z = torch.selu((w_xz*x + w_hz*h + w_mz*m + b_z))
#                    r = torch.selu((w_xr*x + w_hr*h + w_mr*m + b_r))
#                    z = torch.elu((w_xz*x + w_hz*h + w_mz*m + b_z),alpha=1.1)
#                    r = torch.elu((w_xr*x + w_hr*h + w_mr*m + b_r),alpha=1.1)
#                    z = torch.sigmoid((w_xz@x + w_hz@h + w_mz@m + p_z * beita + b_z))
#                    r = torch.sigmoid((w_xr@x + w_hr@h + w_mr@m + p_r * beita + b_r))
#                    h_tilde = torch.tanh((w_xh@x + w_hh@(r * h) + w_mh@m + p_h * beita + b_h))
#                    z = torch.sigmoid((w_xz*x + w_hz*h + w_mz*m + p_z * beita + b_z))
#                    r = torch.sigmoid((w_xr*x + w_hr*h + w_mr*m + p_r * beita + b_r))
#                    h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + w_mh*m + p_h * beita+ b_h))
                    z = torch.sigmoid((w_xz*x + w_hz*h + p_z * beita + b_z))
                    r = torch.sigmoid((w_xr*x + w_hr*h + p_r * beita + b_r))
                    h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + p_h * beita+ b_h))
#                    h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + w_mh*m + b_h))
#                    h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + b_h))
#                    z = torch.sigmoid((w_xz*x + w_hz*h + w_mz*m + b_z))
#                    r = torch.sigmoid((w_xr*x + w_hr*h + w_mr*m + b_r))
#                    h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + w_mh*m + b_h))
#                    
#                    z = torch.sigmoid((np.multiply(w_xz,x) + np.multiply(w_hz,h) + np.multiply(w_mz,m) + b_z))
#                    r = torch.sigmoid((np.multiply(w_xr,x) + np.multiply(w_hr,h) + np.multiply(w_mr,m) + b_r))
#                    h_tilde = torch.tanh((np.multiply(w_xh,x) + np.multiply(w_hh,(r * h)) + np.multiply(w_mh,m) + b_h))


                    dropout = torch.nn.Dropout(p=self.dropout)
                    h_tilde = dropout(h_tilde)

                    h = (1 - z)* h + z*h_tilde

                else:
                    h = gamma_h * h

#                    z = torch.sigmoid((w_xz*x + w_hz*h + w_mz*m + b_z))
#                    r = torch.sigmoid((w_xr*x + w_hr*h + w_mr*m + b_r))
#                    z = torch.relu((w_xz*x + w_hz*h + b_z))
#                    r = torch.relu((w_xr*x + w_hr*h + b_r))
#                    z = torch.selu((w_xz*x + w_hz*h + w_mz*m + b_z))
#                    r = torch.selu((w_xr*x + w_hr*h + w_mr*m + b_r))
#                    z = torch.elu((w_xz*x + w_hz*h + w_mz*m + b_z),alpha=1.1)
#                    r = torch.elu((w_xr*x + w_hr*h + w_mr*m + b_r),alpha=1.1)
#                    z = torch.sigmoid((w_xz@x + w_hz@h + w_mz@m + p_z * beita + b_z))
#                    r = torch.sigmoid((w_xr@x + w_hr@h + w_mr@m + p_r * beita + b_r))
#                    h_tilde = torch.tanh((w_xh@x + w_hh@(r * h) + w_mh@m + p_h * beita + b_h))
#                    z = torch.sigmoid((w_xz*x + w_hz*h + w_mz*m + p_z * beita + b_z))
#                    r = torch.sigmoid((w_xr*x + w_hr*h + w_mr*m + p_r * beita + b_r))
#                    h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + w_mh*m + p_h * beita+ b_h))
                    z = torch.sigmoid((w_xz*x + w_hz*h + p_z * beita + b_z))
                    r = torch.sigmoid((w_xr*x + w_hr*h + p_r * beita + b_r))
                    h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + p_h * beita+ b_h))
#                    h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + w_mh*m + b_h))
#                    h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + b_h))
#                    z = torch.sigmoid((w_xz*x + w_hz*h + w_mz*m + b_z))
#                    r = torch.sigmoid((w_xr*x + w_hr*h + w_mr*m + b_r))
#                    h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + w_mh*m + b_h))
                    
#                    z = torch.sigmoid((np.multiply(w_xz,x) + np.multiply(w_hz,h) + np.multiply(w_mz,m) + b_z))
#                    r = torch.sigmoid((np.multiply(w_xr,x) + np.multiply(w_hr,h) + np.multiply(w_mr,m) + b_r))
#                    h_tilde = torch.tanh((np.multiply(w_xh,x) + np.multiply(w_hh,(r * h)) + np.multiply(w_mh,m) + b_h))

                    h = (1 - z) * h + z * h_tilde
                
                # output = torch.matmul(w_hy, h) + b_y
                # output = self.linear2(h.flatten())
                # output = torch.sigmoid(output)
                # hidden_seq.append(h.unsqueeze(0))

            # hidden_seq = torch.cat(hidden_seq, dim=0)  # (44,185)
            # hidden_seq = hidden_seq.reshape(1,self.num_layers,-1)

                
            # w_hy = getattr(self, 'weight_hy')
            # b_y = getattr(self, 'bias_y')

            # output = torch.matmul(w_hy, h) + b_y
            # output = torch.sigmoid(output)
            # output_lstm, (h,c) = self.rnn(hidden_seq)  # (1,44,100)

            # b_lstm
            # out = torch.matmul(w_lstm_y,output_lstm.reshape(44,100)) + b_lstm

            # 1
            # out = self.linear3(h)
            # out = self.act2(out)

            # out_all[i] = out
            #2
            out = h




        return out