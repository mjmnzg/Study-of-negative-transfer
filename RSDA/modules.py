import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import scipy.io
from torch.utils.data import Dataset
import torch

def load_seed(path, session="all", feature="LDS", n_samples=250):
    """
    SEED I
    A total number of 15 subjects participated the experiment. For each participant,
    3 sessions are performed on different days, and each session contains 24 trials. 
    In one trial, the participant watch one of the film clips, while his(her) EEG 
    signals and eye movements are collected with the 62-channel ESI NeuroScan System 
    and SMI eye-tracking glasses.
    
    """
    
    
    session1 = [
        "dujingcheng_20131027",
        "jianglin_20140404", 
        "jingjing_20140603", 
        "liuqiujun_20140621", 
        "liuye_20140411", 
        "mahaiwei_20130712", 
        "penghuiling_20131027",
        "sunxiangyu_20140511",
        "wangkui_20140620",
        "weiwei_20131130", 
        "wusifan_20140618",
        "wuyangwei_20131127",
        "xiayulu_20140527", 
        "yansheng_20140601", 
        "zhujiayi_20130709"
        ]
        
    session2 = [
        "dujingcheng_20131030", 
        "jianglin_20140413", 
        "jingjing_20140611", 
        "liuqiujun_20140702",
        "liuye_20140418",  
        "mahaiwei_20131016", 
        "penghuiling_20131030", 
        "sunxiangyu_20140514", 
        "wangkui_20140627", 
        "weiwei_20131204",  
        "wusifan_20140625",
        "wuyangwei_20131201", 
        "xiayulu_20140603", 
        "yansheng_20140615",
        "zhujiayi_20131016",
        ]
        
    # SESSION 3
    
    session3 = [
        "dujingcheng_20131107",
        "jianglin_20140419",
        "jingjing_20140629",
        "liuqiujun_20140705",
        "liuye_20140506", 
        "mahaiwei_20131113",
        "penghuiling_20131106",
        "sunxiangyu_20140521",
        "wangkui_20140704",
        "weiwei_20131211",
        "wusifan_20140630",
        "wuyangwei_20131207",
        "xiayulu_20140610", 
        "yansheng_20140627",
        "zhujiayi_20131105"
        ]
        
        
    # LABELS
    labels = scipy.io.loadmat(path + "label.mat", mat_dtype=True)
    y_session = labels["label"][0]
    # relabel to neural networks [0,1,2]
    for i in range(len(y_session)):
        y_session[i] += 1
    print(y_session)
    
    # select session
    if session == 1:
        x_session = session1
    elif session == 2:
        x_session = session2
    elif session == 3:
        x_session = session3
    
    # Load samples
    samples_by_subject = 0
    X = []
    Y = []
    flag = False

    for subj in x_session:
        # load data .mat
        dataMat = scipy.io.loadmat(path + subj + ".mat", mat_dtype=True)
        print("Subject load:", subj)
        
        for i in range(15):
            
            # "Differential_entropy (DE)"
            #   62 channels
            #   42 epochs
            #   5 frequency band
            features = dataMat[feature+str(i+1)]
            
            # swap frequency bands with epochs
            features = np.swapaxes(features, 0, 1)
            
            # select last samples
            if (features.shape[0] - n_samples) > 0:
                pos = features.shape[0] - n_samples
                features = features[pos:]
            
            # set labels for each epoch
            labels = np.array([y_session[i]]*features.shape[0])
            
            # add to arrays
            if flag == 0:
                X = features
                Y = labels
                flag = True
            else:
                X = np.concatenate((X, features), axis=0)
                Y = np.concatenate((Y, labels), axis=0)
        
        if samples_by_subject == 0:
            samples_by_subject = len(X)

    # reorder data by subject
    X_subjects = {}
    Y_subjects = {}
    n = samples_by_subject
    r = 0
    for subj in range(len(x_session)):
        X_subjects[subj] = X[r:r+n]
        Y_subjects[subj] = Y[r:r+n]
        # increment range
        r += n
        print(X_subjects[subj].shape)
    
    return X_subjects, Y_subjects



def load_seed_iv(dir_name, session="all", n_samples=200):
    """
    SEED IV
    A total number of 15 subjects participated the experiment. For each participant,
    3 sessions are performed on different days, and each session contains 24 trials. 
    In one trial, the participant watch one of the film clips, while his(her) EEG 
    signals and eye movements are collected with the 62-channel ESI NeuroScan System 
    and SMI eye-tracking glasses.
    
    """
    
    # SESSION 1
    session1 = [
        "1_20160518", 
        "2_20150915", 
        "3_20150919", 
        "4_20151111",
        "5_20160406",  
        "6_20150507", 
        "7_20150715", 
        "8_20151103", 
        "9_20151028", 
        "10_20151014",  
        "11_20150916",
        "12_20150725", 
        "13_20151115", 
        "14_20151205",
        "15_20150508"
        ]
    # SESSION 2
    session2 = [
        "1_20161125", 
        "2_20150920", 
        "3_20151018", 
        "4_20151118",
        "5_20160413",  
        "6_20150511", 
        "7_20150717", 
        "8_20151110", 
        "9_20151119", 
        "10_20151021",  
        "11_20150921",
        "12_20150804", 
        "13_20151125", 
        "14_20151208",
        "15_20150514"
        ]
    # SESSION 3
    session3 = [
        "1_20161126",
        "2_20151012",
        "3_20151101",
        "4_20151123",
        "5_20160420",
        "6_20150512", 
        "7_20150721", 
        "8_20151117",
        "9_20151209", 
        "10_20151023",  
        "11_20151011",
        "12_20150807", 
        "13_20161130", 
        "14_20151215",
        "15_20150527"
        ]
    
    # select session
    if session == 1:
        x_session = session1
        y_session = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
    elif session == 2:
        x_session = session2
        y_session = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
    elif session == 3:
        x_session = session3
        y_session = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
    
    # Load samples
    samples_by_subject = 0
    X = []
    Y = []
    flag = False
        
    for subj in x_session:
        
        # load data .mat
        dataMat = scipy.io.loadmat(dir_name + "/"+ str(session) + "/" + subj + ".mat", mat_dtype=True)
        print("Subject load:", subj)
        
        for i in range(24):
            
            # "Differential_entropy (DE)"
            #   62 channels
            #   42 epochs
            #   5 frequency band
            
            features = dataMat["de_LDS"+str(i+1)]
            
            # swap frequency bands with epochs
            features = np.swapaxes(features, 0, 1)
            
            # select last samples
            if (features.shape[0] - n_samples) > 0:
                pos = features.shape[0] - n_samples
                features = features[pos:]
            
            # set labels for each epoch
            labels = np.array([y_session[i]]*features.shape[0])
            
            # add to arrays
            if flag == 0:
                X = features
                Y = labels
                flag = True
            else:
                X = np.concatenate((X, features), axis=0)
                Y = np.concatenate((Y, labels), axis=0)
        
        if samples_by_subject == 0:
            samples_by_subject = len(X)
    
    # reorder data by subject
    X_subjects = {}
    Y_subjects = {}
    n = samples_by_subject
    r = 0
    for subj in range(len(x_session)):
        X_subjects[subj] = X[r:r+n]
        Y_subjects[subj] = Y[r:r+n]
        # increment range
        r += n
        print(X_subjects[subj].shape)
    
    return X_subjects, Y_subjects


def z_score(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    z = (mean - X) / std
    return z, mean, std

def normalize(X, mean, std):
    z = (mean - X) / std
    return z

def one_hot(y, n_cls):
    y_new = []
    y = np.array(y, 'int32')
    for i in range(len(y)):
        target = [0] * n_cls
        target[y[i]] = 1
        y_new.append(target)
    return np.array(y_new, 'int32')

# Obtaining TRAIN and TEST from DATA
def split_data(X, Y, seed, test_size=0.3):

    s = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    for train_index, test_index in s.split(X, Y):
        X_tr, X_ts = X[train_index], X[test_index]
        Y_tr, Y_ts = Y[train_index], Y[test_index]

    return X_tr, Y_tr, X_ts, Y_ts



# dataset definition
class PseudoLabeledData(Dataset):
    # load the dataset
    def __init__(self, X, Y, W):
        self.X = torch.Tensor(X).float()
        self.Y = torch.Tensor(Y).long()
        # weights
        self.W = torch.Tensor(W).float()

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx], self.W[idx]]


def load_synth_data(path, name_file, seed, test_size = 0.3):
    # load source data
    X = np.loadtxt(path + 'x_' + name_file + '.csv', delimiter=',')
    Y = np.loadtxt(path + 'y_' + name_file + '.csv', delimiter=',')

    # split data
    Tr_x, Tr_y, Ts_x, Ts_y = split_data(X, Y, seed, test_size=test_size)

    # calculate standardization over training
    Tr_x, m, std = z_score(Tr_x)
    # standardize test data using mean and std from training
    Ts_x = normalize(Ts_x, m, std)

    return Tr_x, Tr_y, Ts_x, Ts_y


def load_stroke(dir_name):
    """
    SEED IV
    A total number of 15 subjects participated the experiment. For each participant,
    3 sessions are performed on different days, and each session contains 24 trials.
    In one trial, the participant watch one of the film clips, while his(her) EEG
    signals and eye movements are collected with the 62-channel ESI NeuroScan System
    and SMI eye-tracking glasses.

    """

    data = [
        "P1_v11_atribMOVPREFAE_etiqOrig",
        "P2_v11_atribMOVPREFAE_etiqOrig",
        "P4_v11_atribMOVPREFAE_etiqOrig",
        "P5_v11_atribMOVPREFAE_etiqOrig"
    ]
    nfeatures = 28
    num_labels = 4

    # Load samples
    samples_by_subject = 0
    X = {}
    Y = {}
    flag = False

    for subj in data:

        # load data file
        data_train = np.genfromtxt(dir_name + subj + ".txt", delimiter=',')


        x_train = np.concatenate((data_train[:, 8:nfeatures], data_train[:, :5]), axis=1)
        y_train = data_train[:, nfeatures:-1]

        X[subj] = []
        Y[subj] = []

        for i in range(len(y_train)):
            label = y_train[i].tolist()

            if label.count(1) > 1:
                pos = 0
                while True:

                    if label[pos] == 1 and (pos != 0 and pos != 2):
                        X[subj].append(x_train[i])
                        Y[subj].append(pos)

                    pos += 1
                    if pos >= num_labels:
                        break

            else:
                lbl = label.index(1)
                if lbl != 0 and lbl != 2:
                    X[subj].append(x_train[i])
                    Y[subj].append(lbl)

        X[subj] = np.array(X[subj])
        Y[subj] = np.array(Y[subj])

        # relabel
        Y[subj] = relabel(Y[subj])

    return X, Y


def relabel(labels):
    """Module to relabel samples due to the order of labels is wrong
    for example: 0, 1, 4, 6.
    """

    #labels = domains.tolist()
    D = labels.astype(int)
    Y = []
    a = set(D)
    d = {}

    # transforms classes from 1-15 to 0-12
    for i, v in enumerate(a):
        d[v] = i

    for i, s in enumerate(D):
        Y.append(d[s])

    Y = np.array(Y)

    return Y.astype(int)