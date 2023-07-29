from dsm import datasets
import numpy as np
from sklearn.model_selection import ParameterGrid
from dsm import DeepRecurrentSurvivalMachines
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc
from data.simulate.utilities import _padding_for_npy
from survtrace import train_cf as cf
import random
import torch
import os
import matplotlib.pyplot as plt
from sklearn import metrics

def get_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__=='__main__':
    get_random_seed(2022)
    # random.seed = 2022
    x, t, e = datasets.load_dataset('PBC2', sequential = True)
    horizons = [0.25, 0.5, 0.75,0.99]
    times = np.quantile([t_[-1] for t_, e_ in zip(t, e) if e_[-1] == 1], horizons).tolist()
    n = len(x)

    tr_size = int(n*0.70)
    vl_size = int(n*0.10)
    te_size = int(n*0.20)

    x_train, x_test, x_val = np.array(x[:tr_size], dtype = object), np.array(x[-te_size:], dtype = object), np.array(x[tr_size:tr_size+vl_size], dtype = object)
    t_train, t_test, t_val = np.array(t[:tr_size], dtype = object), np.array(t[-te_size:], dtype = object), np.array(t[tr_size:tr_size+vl_size], dtype = object)
    e_train, e_test, e_val = np.array(e[:tr_size], dtype = object), np.array(e[-te_size:], dtype = object), np.array(e[tr_size:tr_size+vl_size], dtype = object)


    # param_grid = {'k' : [3, 4, 6],
    #           'distribution' : ['LogNormal', 'Weibull'],
    #           'learning_rate' : [ 1e-5, 1e-4, 1e-3],
    #           'hidden': [25, 50, 75,  100],
    #           'layers': [3, 2, 1],
    #           'typ':['TRANSFORMER'],
    #          }

    param_grid = {'k' : [18],
            'distribution' : ['LogNormal', 'Weibull'],
            'learning_rate' : [  1e-4, 1e-3,1e-2],
            'hidden': [100,25],
            'layers': [3],
            'typ':['TRANSFORMER'],
            }
             #['TRANSFORMER','LSTM', 'GRU', 'RNN']
    params = ParameterGrid(param_grid)
    ##反事实模型
    cf_model = cf.Counterfactual()
    #训练
    cf_model.tranin_cf_model()
    # cf_model.eval()
    cf_model_global = cf_model.counterfactorGAN
    models = []
    for param in params:
        # if(param['hidden'] > 25):
        #     print()
        model = DeepRecurrentSurvivalMachines(k = param['k'],
                                    distribution = param['distribution'],
                                    hidden = param['hidden'],  
                                    typ = param['typ'],
                                    layers = param['layers'])
        # The fit method is called to train the model

        model.fit(x_train, t_train, e_train, iters = 30, learning_rate = param['learning_rate'],cf_model_global=cf_model_global)
        models.append([[model.compute_nll(x_val, t_val, e_val,cf_model_global=cf_model_global), model]])

    best_model = min(models)
    
    model = best_model[0][1]
    t_test = _padding_for_npy(t_test, t_test.shape[0], 16)
    out_risk = model.predict_risk(x_test, t_test, times,cf_model_global=cf_model_global)
    out_survival = model.predict_survival(x_test, t_test, times,cf_model_global=cf_model_global)




    # inference

cis = []
brs = []

et_train = np.array([(e_train[i][j], t_train[i][j]) for i in range(len(e_train)) for j in range(len(e_train[i]))],
                 dtype = [('e', bool), ('t', float)])
et_test = np.array([(e_test[i][j], t_test[i][j]) for i in range(len(e_test)) for j in range(len(e_test[i]))],
                 dtype = [('e', bool), ('t', float)])
et_val = np.array([(e_val[i][j], t_val[i][j]) for i in range(len(e_val)) for j in range(len(e_val[i]))],
                 dtype = [('e', bool), ('t', float)])

for i, _ in enumerate(times):
    cis.append(concordance_index_ipcw(et_train, et_test, out_risk[:, i], times[i])[0])
brs.append(brier_score(et_train, et_test, out_survival, times)[1])
roc_auc = []
for i, _ in enumerate(times):
    roc_auc.append(cumulative_dynamic_auc(et_train, et_test, out_risk[:, i], times[i])[0])
for horizon in enumerate(horizons):
    print(f"For {horizon[1]} quantile,")
    print("TD Concordance Index:", cis[horizon[0]])
    print("Brier Score:", brs[0][horizon[0]])
    print("ROC AUC ", roc_auc[horizon[0]][0], "\n")