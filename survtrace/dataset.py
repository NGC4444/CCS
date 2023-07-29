from random import random
from pycox.datasets import metabric, nwtco, support, gbsg, flchain
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, StandardScaler
import numpy as np
import pandas as pd
import pdb

from .utils import LabelTransform, pata_discre
from sklearn.model_selection import train_test_split

##### USER-DEFINED FUNCTIONS
def f_get_Normalization(X, norm_mode):    
    num_Patient, num_Feature = np.shape(X)
    
    if norm_mode == 'standard': #zero mean unit variance
        for j in range(num_Feature):
            if np.nanstd(X[:,j]) != 0:
                X[:,j] = (X[:,j] - np.nanmean(X[:, j]))/np.nanstd(X[:,j])
            else:
                X[:,j] = (X[:,j] - np.nanmean(X[:, j]))
    elif norm_mode == 'normal': #min-max normalization
        for j in range(num_Feature):
            X[:,j] = (X[:,j] - np.nanmin(X[:,j]))/(np.nanmax(X[:,j]) - np.nanmin(X[:,j]))
    else:
        print("INPUT MODE ERROR!")
    
    return X


def f_get_fc_mask1(meas_time, num_Event, num_Category):
    '''

        mask3 is required to get the contional probability (to calculate the denominator part)
        mask3 size is [N, num_Event, num_Category]. 1's until the last measurement time
    '''
    
    mask = np.zeros([np.shape(meas_time)[0], num_Event, num_Category]) # for denominator
    for i in range(np.shape(meas_time)[0]):
        mask[i, :, :int(meas_time[i, 0]+1)] = 1 # last measurement time

    return mask


def f_get_fc_mask2(time, label, num_Event, num_Category):
    
    '''
        mask4 is required to get the log-likelihood loss 
        mask4 size is [N, num_Event, num_Category]
            if not censored : one element = 1 (0 elsewhere)
            if censored     : fill elements with 1 after the censoring time (for all events)
    '''

    mask = np.zeros([np.shape(time)[0], num_Event, num_Category]) # for the first loss function
    for i in range(np.shape(time)[0]):
        if label[i,0] != 0:  #not censored
            mask[i,int(label[i,0]-1),int(time[i,0])] = 1
        else: #label[i,2]==0: censored  
            mask[i,:,int(time[i,0]+1):] =  1 #fill 1 until from the censoring time (to get 1 - \sum F)
    return mask


def f_get_fc_mask3(time, meas_time, num_Category):
    '''
        mask5 is required calculate the ranking loss (for pair-wise comparision)
        mask5 size is [N, num_Category]. 
        - For longitudinal measurements:
             1's from the last measurement to the event time (exclusive and inclusive, respectively)
             denom is not needed since comparing is done over the same denom
        - For single measurement:
             1's from start to the event time(inclusive)
    '''
    mask = np.zeros([np.shape(time)[0], num_Category]) # for the first loss function
    if np.shape(meas_time):  #lonogitudinal measurements 
        for i in range(np.shape(time)[0]):
            t1 = int(meas_time[i, 0]) # last measurement time
            t2 = int(time[i, 0]) # censoring/event time
            mask[i,(t1+1):(t2+1)] = 1  #this excludes the last measurement time and includes the event time
    else:                    #single measurement
        for i in range(np.shape(time)[0]):
            t = int(time[i, 0]) # censoring/event time
            mask[i,:(t+1)] = 1  #this excludes the last measurement time and includes the event time
    return mask
def f_construct_dataset(df, feat_list):
    '''
        id   : patient indicator
        tte  : time-to-event or time-to-censoring
            - must be synchronized based on the reference time
        times: time at which observations are measured
            - must be synchronized based on the reference time (i.e., times start from 0)
        label: event/censoring information
            - 0: censoring
            - 1: event type 1
            - 2: event type 2
            ...
    '''

    grouped  = df.groupby(['id'])
    id_list  = pd.unique(df['id'])
    max_meas = np.max(grouped.count())[0]

    data     = np.zeros([len(id_list), max_meas, len(feat_list)+1])
    pat_info = np.zeros([len(id_list), 5])

    for i, tmp_id in enumerate(id_list):
        tmp = grouped.get_group(tmp_id).reset_index(drop=True)

        pat_info[i,4] = tmp.shape[0]                                   #number of measurement
        pat_info[i,3] = np.max(tmp['times'])     #last measurement time
        pat_info[i,2] = tmp['label'][0]      #cause
        pat_info[i,1] = tmp['tte'][0]         #time_to_event
        pat_info[i,0] = tmp['id'][0]      

        data[i, :int(pat_info[i, 4]), 1:]  = tmp[feat_list]
        data[i, :int(pat_info[i, 4]-1), 0] = np.diff(tmp['times'])
    
    return pat_info, data
def load_data(config):
    '''load data, return updated configuration.
    '''
    data = config['data']
    horizons = config['horizons']
    assert data in ["metabric", "nwtco", "support", "gbsg", "flchain", "seer", "pbc2"], "Data Not Found!"
    get_target = lambda df: (df['duration'].values, df['event'].values)

    if data == "metabric":
        # data processing, transform all continuous data to discrete
        df = metabric.read_df()

        # evaluate the performance at the 25th, 50th and 75th event time quantile
        times = np.quantile(df["duration"][df["event"]==1.0], horizons).tolist()

        cols_categorical = ["x4", "x5", "x6", "x7"]
        cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']

        df_feat = df.drop(["duration","event"],axis=1)
        df_feat_standardize = df_feat[cols_standardize] 
        df_feat_standardize_disc = StandardScaler().fit_transform(df_feat_standardize)
        df_feat_standardize_disc = pd.DataFrame(df_feat_standardize_disc, columns=cols_standardize)

        # must be categorical feature ahead of numerical features!
        df_feat = pd.concat([df_feat[cols_categorical], df_feat_standardize_disc], axis=1)
        
        vocab_size = 0
        for _,feat in enumerate(cols_categorical):
            df_feat[feat] = LabelEncoder().fit_transform(df_feat[feat]).astype(float) + vocab_size
            vocab_size = df_feat[feat].max() + 1
                
        # get the largest duraiton time
        max_duration_idx = df["duration"].argmax()
        df_test = df_feat.drop(max_duration_idx).sample(frac=0.3)
        df_train = df_feat.drop(df_test.index)
        df_val = df_train.drop(max_duration_idx).sample(frac=0.1)
        df_train = df_train.drop(df_val.index)

        # assign cuts
        labtrans = LabelTransform(cuts=np.array([df["duration"].min()]+times+[df["duration"].max()]))
        labtrans.fit(*get_target(df.loc[df_train.index]))
        y = labtrans.transform(*get_target(df)) # y = (discrete duration, event indicator)
        df_y_train = pd.DataFrame({"duration": y[0][df_train.index], "event": y[1][df_train.index], "proportion": y[2][df_train.index]}, index=df_train.index)
        df_y_val = pd.DataFrame({"duration": y[0][df_val.index], "event": y[1][df_val.index],  "proportion": y[2][df_val.index]}, index=df_val.index)
        # df_y_test = pd.DataFrame({"duration": y[0][df_test.index], "event": y[1][df_test.index],  "proportion": y[2][df_test.index]}, index=df_test.index)
        df_y_test = pd.DataFrame({"duration": df['duration'].loc[df_test.index], "event": df['event'].loc[df_test.index]})
    
    elif data == "support":
        df = support.read_df()
        times = np.quantile(df["duration"][df["event"]==1.0], horizons).tolist()
        cols_categorical = ["x1", "x2", "x3", "x4", "x5", "x6"]
        cols_standardize = ['x0', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']

        df_feat = df.drop(["duration","event"],axis=1)
        df_feat_standardize = df_feat[cols_standardize]        
        df_feat_standardize_disc = StandardScaler().fit_transform(df_feat_standardize)
        df_feat_standardize_disc = pd.DataFrame(df_feat_standardize_disc, columns=cols_standardize)

        df_feat = pd.concat([df_feat[cols_categorical], df_feat_standardize_disc], axis=1)
        
        vocab_size = 0
        for i,feat in enumerate(cols_categorical):
            df_feat[feat] = LabelEncoder().fit_transform(df_feat[feat]).astype(float) + vocab_size
            vocab_size = df_feat[feat].max() + 1

        # get the largest duraiton time
        max_duration_idx = df["duration"].argmax()
        df_test = df_feat.drop(max_duration_idx).sample(frac=0.3)
        df_train = df_feat.drop(df_test.index)
        df_val = df_train.drop(max_duration_idx).sample(frac=0.1)
        df_train = df_train.drop(df_val.index)

        # assign cuts
        # labtrans = LabTransDiscreteTime(cuts=np.array([0]+times+[df["duration"].max()]))
        labtrans = LabelTransform(cuts=np.array([0]+times+[df["duration"].max()]))

        labtrans.fit(*get_target(df.loc[df_train.index]))
        # y = labtrans.fit_transform(*get_target(df)) # y = (discrete duration, event indicator)
        y = labtrans.transform(*get_target(df)) # y = (discrete duration, event indicator)
        df_y_train = pd.DataFrame({"duration": y[0][df_train.index], "event": y[1][df_train.index], "proportion":y[2][df_train.index]}, index=df_train.index)
        df_y_val = pd.DataFrame({"duration": y[0][df_val.index], "event": y[1][df_val.index], "proportion":y[2][df_val.index]}, index=df_val.index)
        # df_y_test = pd.DataFrame({"duration": y[0][df_test.index], "event": y[1][df_test.index], "proportion":y[2][df_test.index]}, index=df_test.index)
        df_y_test = pd.DataFrame({"duration": df['duration'].loc[df_test.index], "event": df['event'].loc[df_test.index]})


    elif data == "seer":
        PATH_DATA = "./data/seer_processed.csv"
        df = pd.read_csv(PATH_DATA)
        times = np.quantile(df["duration"][df["event_breast"]==1.0], horizons).tolist()

        event_list = ["event_breast", "event_heart"]

        cols_categorical = ["Sex", "Year of diagnosis", "Race recode (W, B, AI, API)", "Histologic Type ICD-O-3",
                    "Laterality", "Sequence number", "ER Status Recode Breast Cancer (1990+)",
                    "PR Status Recode Breast Cancer (1990+)", "Summary stage 2000 (1998-2017)",
                    "RX Summ--Surg Prim Site (1998+)", "Reason no cancer-directed surgery", "First malignant primary indicator",
                    "Diagnostic Confirmation", "Median household income inflation adj to 2019"]
        cols_standardize = ["Regional nodes examined (1988+)", "CS tumor size (2004-2015)", "Total number of benign/borderline tumors for patient",
                    "Total number of in situ/malignant tumors for patient",]

        df_feat = df.drop(["duration","event_breast", "event_heart"],axis=1)

        df_feat_standardize = df_feat[cols_standardize]        
        df_feat_standardize_disc = StandardScaler().fit_transform(df_feat_standardize)
        df_feat_standardize_disc = pd.DataFrame(df_feat_standardize_disc, columns=cols_standardize)
        df_feat = pd.concat([df_feat[cols_categorical], df_feat_standardize_disc], axis=1)

        vocab_size = 0
        for i,feat in enumerate(cols_categorical):
            df_feat[feat] = LabelEncoder().fit_transform(df_feat[feat]).astype(float) + vocab_size
            vocab_size = df_feat[feat].max() + 1
        
        # get the largest duraiton time
        max_duration_idx = df["duration"].argmax()
        df_test = df_feat.drop(max_duration_idx).sample(frac=0.3)
        df_train = df_feat.drop(df_test.index)
        df_val = df_train.drop(max_duration_idx).sample(frac=0.1)
        df_train = df_train.drop(df_val.index)

        # assign cuts
        labtrans = LabelTransform(cuts=np.array([0]+times+[df["duration"].max()]))
        get_target = lambda df,event: (df['duration'].values, df[event].values)

        # this datasets have two competing events!
        df_y_train = pd.DataFrame({"duration":df["duration"][df_train.index]})
        df_y_test = pd.DataFrame({"duration":df["duration"][df_test.index]})
        df_y_val = pd.DataFrame({"duration":df["duration"][df_val.index]})

        for i,event in enumerate(event_list):
            labtrans.fit(*get_target(df.loc[df_train.index], event))
            y = labtrans.transform(*get_target(df, event)) # y = (discrete duration, event indicator)

            event_name = "event_{}".format(i)
            df[event_name] = y[1]
            df_y_train[event_name] = df[event_name][df_train.index]
            df_y_val[event_name] = df[event_name][df_val.index]
            df_y_test[event_name] = df[event_name][df_test.index]

        # discretized duration
        df["duration_disc"] = y[0]

        # proportion is the same for all events
        df["proportion"] = y[2]

        df_y_train["proportion"] = df["proportion"][df_train.index]
        df_y_val["proportion"] = df["proportion"][df_val.index]
        df_y_test["proportion"] = df["proportion"][df_test.index]

        df_y_train["duration"] = df["duration_disc"][df_train.index]
        df_y_val["duration"] = df["duration_disc"][df_val.index]
        df_y_test["duration"] = df["duration"][df_test.index]

        # set number of events
        config['num_event'] = 2
    elif data == 'pbc2':
        seed = config['seed']
        norm_mode = 'standard'
        PATH_DATA = "./data/pbc2_cleaned.csv"
        df_ = pd.read_csv(PATH_DATA) # times队列时间,tte终点时间
        bin_list           = ['drug', 'sex', 'ascites', 'hepatomegaly', 'spiders']
        cont_list          = ['age', 'edema', 'serBilir', 'serChol', 'albumin', 'alkaline', 'SGOT', 'platelets', 'prothrombin', 'histologic']
        feat_list          = cont_list + bin_list
        df_                = df_[['id', 'tte', 'times', 'label']+feat_list]
        df_org_            = df_.copy(deep=True)

        df_[cont_list]     = f_get_Normalization(np.asarray(df_[cont_list]).astype(float), norm_mode)

        pat_info, data_     = f_construct_dataset(df_, feat_list)
        _, data_org        = f_construct_dataset(df_org_, feat_list)

        data_mi                  = np.zeros(np.shape(data_))
        data_mi[np.isnan(data_)]  = 1
        data_org[np.isnan(data_)] = 0
        data_[np.isnan(data_)]     = 0 

        x_dim           = np.shape(data_)[2] # 1 + x_dim_cont + x_dim_bin (including delta)
        x_dim_cont      = len(cont_list)
        x_dim_bin       = len(bin_list) 

        last_meas       = pat_info[:,[3]]  #pat_info[:, 3] contains age at the last measurement
        label           = pat_info[:,[2]]  #two competing risks
        time            = pat_info[:,[1]]  #age when event occurred

        num_Category    = int(np.max(pat_info[:, 1]) * 1.2) #随访时间分类or specifically define larger than the max tte
        num_Event       = len(np.unique(label)) - 1

        if num_Event == 1:
            label[np.where(label!=0)] = 1 #make single risk

        mask1           = f_get_fc_mask1(last_meas, num_Event, num_Category)
        mask2           = f_get_fc_mask2(time, label, num_Event, num_Category)
        mask3           = f_get_fc_mask3(time, -1, num_Category)

        DIM             = (x_dim, x_dim_cont, x_dim_bin)
        DATA            = (data_, time, label)
        # DATA            = (data, data_org, time, label)
        MASK            = (mask1, mask2, mask3)

        ## discreat time
        times = np.quantile(df_["tte"], horizons).tolist()
        cuts=np.array([0]+times+[df_["tte"].max()])
        y = pata_discre(pat_info, cuts)
        df_train, df_test, df_y_train, df_y_test,info_train, info_test = train_test_split(data_, y,pat_info, test_size = 0.2, random_state = seed)
        df_train, df_val, df_y_train, df_y_val, info_train, info_val = train_test_split(df_train, df_y_train, info_train, test_size = 0.2, random_state = seed)
        # get_target = lambda df,event: (df['tte'].values, df[event].values)
        config['num_event'] = 2



        df = data_
        # df_train = ''

          
    if data != 'pbc2':
        config['labtrans'] = labtrans
        config['num_numerical_feature'] = int(len(cols_standardize))
        config['num_categorical_feature'] = int(len(cols_categorical))
        config['num_feature'] = int(len(df_train.columns))
        config['vocab_size'] = int(vocab_size)
        config['duration_index'] = labtrans.cuts
        config['out_feature'] = int(labtrans.out_features)
    else:
        config['duration_index'] = cuts
        config['out_feature'] = 4
        config['num_numerical_feature'] = 11
        config['num_categorical_feature'] = 5
        config['num_feature'] = 16
        config['vocab_size'] = int(185)
    return df, df_train, df_y_train, df_test, df_y_test, df_val, df_y_val, info_train[:, 1:3], info_test[:, 1:3], info_val[:, 1:3]#y:时期, event1, event2, 比例