import numpy as np
from codebase.args import args
from sklearn.model_selection import StratifiedShuffleSplit
from codebase.modules_jmnzg import z_score, normalize



def get_info(domain_id, domain):
    train, test = domain.train, domain.test
    Y_shape_train = None if train.labels is None else train.labels.shape
    Y_shape_test = None if test.labels is None else test.labels.shape

    print('{} info'.format(domain_id))
    print('Train X/Y shapes: {}, {}'.format(train.samples.shape, Y_shape_train))
    print('Train X min/max/cast: {}, {}, {}'.format(
        train.samples.min(),
        train.samples.max(),
        train.cast))

    print('Test shapes: {}, {}'.format(test.samples.shape, Y_shape_test))
    print('Test X min/max/cast: {}, {}, {}\n'.format(
        test.samples.min(),
        test.samples.max(),
        test.cast))

class Data(object):
    def __init__(self, samples, labels=None, labeler=None, domain_labels=None, cast=False):
        """Data object constructs mini-batches to be fed during training

        samples - (NHWC) data
        labels - (NK) one-hot data
        labeler - (tb.function) returns simplex value given an image
        cast - (bool) converts uint8 to [-1, 1] float
        """
        self.samples = samples
        self.labels = labels
        self.domain_labels = domain_labels
        self.labeler = labeler
        self.cast = False

    def next_batch(self, bs):
        """Constructs a mini-batch of size bs without replacement
        """
        idx = np.random.choice(len(self.samples), bs, replace=False)
        x = self.samples[idx]
        y = self.labeler(x) if self.labels is None else self.labels[idx]
        return x, y
    
    
    def next_batch_stratified(self, batch_size):
        
        samples = None
        labels = None
        bs = batch_size # // len(dom_lbls)
        flag = False
        
        # generate function batches
        for i in range(args.Y):
            
            # identify elements that belong to domain "i"
            indices = self.labels == i
            # get indices for identified elements
            indices = np.squeeze(np.nonzero(indices))
            
            x, y = self.get_batch(bs, indices)
            if not flag:
                samples = x
                labels = y
                flag = True
            else:
                samples = np.concatenate((samples, x), 0)
                labels = np.concatenate((labels, y), 0)
        return samples, labels
    
    
    def get_batch(self, bs, indices):
        """Constructs a mini-batch of size bs without replacement
        """
        idx = np.random.choice(len(self.samples[indices]), bs, replace=True)
        x = self.samples[idx]
        y = self.labeler(x) if self.labels is None else self.labels[idx]
        return x, y
    

class Synthetic(object):
    def __init__(self, name_data, seed):
        """Synthetic Dataset
        """
        print("Loading Synthetic Data")
        
        def reformat_target_logits(y, nb_classes):
            """Reformat Targets (labels) to be used 2 dimensions [1,2,3] ===> [[1,0,0][0,1,0][0,0,1]]"""
            y_new = []
            for i in range(len(y)):
                target = [0]*nb_classes
                target[y[i]] = 1
                y_new.append(target)

            return np.array(y_new,'int32')

        # load data based on artificial normal distribution
        X = np.loadtxt('/home/magdiel/Descargas/Datasets/Synthetic/x_' + name_data + '.csv',delimiter=',')
        Y = np.loadtxt('/home/magdiel/Descargas/Datasets/Synthetic/y_' + name_data + '.csv',delimiter=',')
        
        
        # reformat data
        Y = reformat_target_logits(Y.astype(int), args.Y)
        s = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
        
        for train_index, test_index in s.split(X,Y):
            trainx, testx = X[train_index], X[test_index]
            trainy, testy = Y[train_index], Y[test_index]

        # standardize training data
        trainx, mean, std = z_score(trainx)
        testx = normalize(testx, mean, std)

        self.train = Data(trainx, trainy)
        self.test = Data(testx, testy)


class Mdata(object):
    def __init__(self, trainx, trainy, testx, testy):
        """
        LOAD Dataset
        """
        
        def reformat_target_logits(y, nb_classes):
            """Reformat Targets (labels) to be used 2 dimensions [1,2,3] ===> [[1,0,0][0,1,0][0,0,1]]"""
            y_new = []
            for i in range(len(y)):
                target = [0]*nb_classes
                target[y[i]] = 1
                y_new.append(target)

            return np.array(y_new,'int32')

        # reshape labels
        trainy = reformat_target_logits(trainy.astype(int), args.Y)
        testy = reformat_target_logits(testy.astype(int), args.Y)
        
        self.train = Data(trainx, trainy)
        self.test = Data(testx, testy)
        

class PseudoData(object):
    def __init__(self, domain_id, domain, teacher):
        """Variable domain with psuedolabeler

        domain_id - (str) {Mnist,Mnistm,Svhn,etc}
        domain - (obj) {Mnist,Mnistm,Svhn,etc}
        teacher - (fn) Teacher model used for pseudolabeling
        """
        print("Constructing pseudodata")
        print("DATASET {} PSEUDOLABEL".format(domain_id))
        labeler = teacher

        self.train = Data(domain.train.samples, labeler=labeler, cast=False)
        self.test = Data(domain.test.samples, labeler=labeler, cast=False)


def get_data(id_data, name_data, seed=1234):
    """Returns Domain object based on domain_id
    """
    if id_data in ['synth_blobs','synth_moons']:
        return Synthetic(name_data, seed)
    else:
        raise Exception('dataset {:s} not recognized'.format("data"))
