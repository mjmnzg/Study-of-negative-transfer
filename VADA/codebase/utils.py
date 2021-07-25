import os
import numpy as np
import shutil
import tensorbayes as tb
import tensorflow as tf
from codebase.args import args
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

def u2t(x):
    """Convert uint8 to [-1, 1] float
    """
    return x.astype('float32') / 255 * 2 - 1

def s2t(x):
    """Convert [0, 1] float to [-1, 1] float
    """
    return x * 2 - 1

def delete_existing(path, run):
    """Delete directory if it exists

    Used for automatically rewrites existing log directories
    """
    
    if run < 999:
        assert not os.path.exists(path), "Cannot overwrite {:s}".format(path)

    else:
        if os.path.exists(path):
            shutil.rmtree(path)

def save_model(saver, M, model_dir, global_step):
    path = saver.save(M.sess, os.path.join(model_dir, 'model'),
                      global_step=global_step)
    print("Saving model to {}".format(path))

def save_value(fn_val, tag, data,
               train_writer=None, global_step=None, print_list=None,
               full=True):
    """Log fn_val evaluation to tf.summary.FileWriter

    fn_val       - (fn) Takes (x, y) as input and returns value
    tag          - (str) summary tag for FileWriter
    data         - (Data) data object with samples/labels attributes
    train_writer - (FileWriter)
    global_step  - (int) global step in file writer
    print_list   - (list) list of vals to print to stdout
    full         - (bool) use full dataset v. first 1000 samples
    """
    acc, summary = compute_value(fn_val, tag, data, full)
    train_writer.add_summary(summary, global_step)
    print_list += [os.path.basename(tag), acc]

def compute_value(fn_val, tag, data, full=True):
    """Compute value w.r.t. data

    fn_val - (fn) Takes (x, y) as input and returns value
    tag    - (str) summary tag for FileWriter
    data   - (Data) data object with samples/labels attributes
    full   - (bool) use full dataset v. first 1000 samples
    """
    with tb.nputils.FixedSeed(0):
        shuffle = np.random.permutation(len(data.samples))

    xs = data.samples[shuffle]
    ys = data.labels[shuffle] if data.labels is not None else None

    if not full:
        xs = xs[:1000]
        ys = ys[:1000] if ys is not None else None

    acc = 0.
    n = len(xs)
    bs = 200

    for i in range(0, n, bs):
        x = xs[i:i+bs]#data.preprocess()
        y = ys[i:i+bs] if ys is not None else data.labeler(x)
        acc += fn_val(x, y) / n * len(x)

    summary = tf.Summary.Value(tag=tag, simple_value=acc)
    summary = tf.Summary(value=[summary])
    return acc, summary

def reformat_target_digits(y):
    """Reformat Targets (labels) to be used 2 dimensions [[1,0,0][0,1,0][0,0,1]] to [1,2,3]"""
    y_new = []
    for i in range(len(y)):
        target = np.argmax(y[i])
        y_new.append(target)

    return np.array(y_new,'int32')

def plot_embedding_tsne(X, y, d, output_dir, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    
    plt.clf()
    # PLOT COLORED NUMBERS
    plt.figure(figsize=(10, 10)) 
    ax = plt.subplot(111)
    
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color = plt.cm.bwr(d[i] / 1.),
                 fontdict={'weight': 'bold', 'size': 12})
    xmean, xvar = np.mean(X, 0), np.var(X, 0)
    if title is not None:
        plt.title(title)
    
    plt.savefig(output_dir)



def loocv(domains):
    """
    Generate Leave-Subject-Out cross validation
    """
    fold_pairs = []
    for i in np.unique(domains):
        ts = domains == i       #return array with True where the index i is equal to indices in subjNumbers
        tr = np.squeeze(np.nonzero(np.bitwise_not(ts))) #first return array with Trues where the index i is equal to indices in subjNumbers but inverted
                                                        #after convert this array of numbers.
        ts = np.squeeze(np.nonzero(ts))                 #conver ts with trues to array with numbers
        np.random.shuffle(tr)       # Shuffle indices
        np.random.shuffle(ts)
        fold_pairs.append((tr, ts))
    
    return fold_pairs
