#!/usr/bin/python3


import os
import argparse
from codebase import args as codebase_args
from pprint import pprint
import tensorflow as tf
import numpy as np
import random
from codebase.modules_jmnzg import load_seed_i, load_seed_iv, load_stroke, split_data, z_score, normalize


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data',           type=str,   default='synthetic',   help="EEG, Synthetic")
parser.add_argument('--source_data',    type=str, default='dsb00',    help="name of source domain")
parser.add_argument('--target_data',    type=str, default='dsb01',    help="name of target domain")
parser.add_argument('--loss',           type=str, default='VADA',    help="type of loss function")
parser.add_argument('--nn',             type=str,   default='dnn',   help="Architecture")
parser.add_argument('--use_ema',    type=bool,   default=False,   help="Use EMA average")
parser.add_argument('--trim',   type=int,   default=2,         help="Trim")
parser.add_argument('--inorm',  type=int,   default=0,         help="Instance normalization flag")
parser.add_argument('--radius', type=float, default=3.5,       help="Perturbation 2-norm ball radius")
parser.add_argument('--dw',     type=float, default=0.01,      help="Domain weight")
parser.add_argument('--bw',     type=float, default=1e-2,      help="Beta (KL) weight")
parser.add_argument('--sw',     type=float, default=1e-2,         help="Src weight")
parser.add_argument('--tw',     type=float, default=1e-2,      help="Trg weight") # 0.5 para 
parser.add_argument('--lr',     type=float, default=1e-3,      help="Learning rate")
parser.add_argument('--dirt',   type=int,   default=0,         help="0 == VADA, >0 == DIRT-T interval")
parser.add_argument('--run',    type=int,   default=999,       help="Run index. >= 999 == debugging")
parser.add_argument('--datadir',type=str,   default="outputs",      help="folder to save weights")
parser.add_argument('--dir_resume',type=str,   default="outputs",      help="folder to save results")
parser.add_argument('--logdir', type=str,   default='log',     help="Log directory")
parser.add_argument('--seed',   type=int, default=1234,    help="random seed")
parser.add_argument('--batch_size',   type=int, default=96,    help="batch size")
parser.add_argument('--optimizer',   type=str, default="sgd",    help="optimizer algorithm gradient descent")
parser.add_argument('--num_epochs',   type=int, default=30,    help="number of epochs")
parser.add_argument('--session',   type=int, default=1,    help="session for SEED database")
parser.add_argument('--measure',   type=str, default="voi",    help="Options: 'voi' or 'ent'")
parser.add_argument('--partition',   type=int, default=1,    help="1-5, for imagined speech dataset")



codebase_args.args = args = parser.parse_args()


    

def loocv_emotions(X, Y, args):
    """
    Leave One Out Cross Validation (LOOCV) EMOTION DATA
    *******************************************************
    Params
    X: all dataset
    Y: labels of classes.
    subjects: labels of subject.
    model: architecture to be used.
    """
    # IMPORT LIBRARYS Here to use "args" as variables in other FILES
    from codebase.models.dirtt import dirtt
    from codebase.train import train
    from codebase.datasets import Mdata
    
    list_metrics_clsf = []
    
    # dictionary keys
    subjects = X.keys()
    
    foldNum = 0
    
    # Iterate over subject data
    for subj in subjects:
        
        print('Beginning fold ' + str(foldNum+1) + ' of ' + str(len(subjects)))

        # Create source domain
        Sx_train = Sy_train = Sx_valid = Sy_valid = None

        i = 0
        for s in subjects:
            if s != subj:
                Xs = np.array(X[s])
                Ys = np.array(Y[s])
                # split data
                tr_x, tr_y, va_x, va_y = split_data(Xs, Ys, args.seed, test_size=0.05)
                # Standardize
                tr_x, m, std = z_score(tr_x)
                va_x = normalize(va_x, mean=m, std=std)

                if i == 0:
                    Sx_train = tr_x
                    Sy_train = tr_y
                    Sx_valid = va_x
                    Sy_valid = va_y
                else:
                    Sx_train = np.concatenate((Sx_train, tr_x), axis=0)
                    Sy_train = np.concatenate((Sy_train, tr_y), axis=0)
                    Sx_valid = np.concatenate((Sx_valid, va_x), axis=0)
                    Sy_valid = np.concatenate((Sy_valid, va_y), axis=0)
                i+=1

        # Create Target domain
        X_target = np.array(X[subj])
        Y_target = np.array(Y[subj])
        # split data
        Tx_train, Ty_train, Tx_test, Ty_test = split_data(X_target, Y_target, args.seed, test_size=0.3)
        # Standardize Target domain
        Tx_train, m, std = z_score(Tx_train)
        Tx_test = normalize(Tx_test, mean=m, std=std)
        
        # labels
        y_classes = np.unique(Ty_train)
        
        
        print("Sx_train-shape:",Sx_train.shape, "Sx_valid-shape:",Sx_valid.shape)
        print("Sy_train-shape:",Sy_train.shape, "Sy_valid-shape:",Sy_valid.shape)
        print("Tx_train-shape:",Tx_train.shape, "Tx_test-shape:",Tx_test.shape)
        print("Ty_train-shape:",Ty_train.shape, "Ty_test-shape:",Ty_test.shape)
        print("y_classes:", y_classes)
        
        
        # Source and target data
        src = Mdata(Sx_train, Sy_train, Sx_valid, Sy_valid)
        trg = Mdata(Tx_train, Ty_train, Tx_test, Ty_test)
        
        # Make model name
        setup = [
            ('model={:s}',  'dirtt'),
            ('data={:s}',    args.data),
            ('name_data={:s}', args.target_data),
            ('nn={:s}',     args.nn),
            ('trim={:d}',   args.trim),
            ('dw={:.0e}',   args.dw),
            ('bw={:.0e}',   args.bw),
            ('sw={:.0e}',   args.sw),
            ('tw={:.0e}',   args.tw),
            ('dirt={:05d}', args.dirt),
            ('run={:04d}',  999)
        ]
        print
        model_name = '_'.join([t.format(v) for (t, v) in setup])
        model_name += "_subj="+str(foldNum+1)
        print("Model name:", model_name)
        
        
        # RESET GRAPH MODEL
        tf.reset_default_graph()
        # CREATE MODEL
        M = dirtt(input_shape_samples=(None, Sx_train.shape[1], Sx_train.shape[2]))
        # Inizialization variables
        M.sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        
        # if It uses DIRT-T where it suppose that Ps(x)=Pt(x)
        if args.dirt > 0:
            # RESTORE PARAMETERS OF VADA
            run = args.run if args.run < 999 else 0
            setup = [
                ('model={:s}',  'dirtt'),
                ('data={:s}',    args.data),
                ('name_data={:s}', args.target_data),
                ('nn={:s}',     args.nn),
                ('trim={:d}',   args.trim),
                ('dw={:.0e}',   args.dw),
                ('bw={:.0e}',   0),
                ('sw={:.0e}',   args.sw),
                ('tw={:.0e}',   args.tw),
                ('dirt={:05d}', 0),
                ('run={:04d}',  999)
            ]
            vada_name = '_'.join([t.format(v) for (t, v) in setup])
            vada_name += "_subj="+str(foldNum+1)
            path = os.path.join('checkpoints', vada_name)
            print(path)
            path = tf.train.latest_checkpoint(path)
            saver.restore(M.sess, path)
            print("Restored from {}".format(path))
            
        
        disc = True if args.loss == "vada" or args.loss == "dann" else False
        
        # WOW! TRAIN MODEL
        classification_metrics = train(M, src, trg,
              saver=saver,
              has_disc=disc,
              model_name=model_name,
              generate_classification_metrics=True)
        print("\n")
        
        foldNum += 1
        
        # add to list
        list_metrics_clsf.append(classification_metrics)
        print()
    
    # To np array
    list_metrics_clsf = np.array(list_metrics_clsf)
    
    print("CLASSIFICATION METRICS:")
    for i in range(len(list_metrics_clsf[0])):
        mean = list_metrics_clsf[:,i].mean()
        print("Metric [",(i+1),"] = ", list_metrics_clsf[:,i]," Mean:", mean)
    
    # Save Classification Metrics
    save_file = args.dir_resume+"/" + args.loss + "-" + args.data + "-metrics-classification.csv"
    f=open(save_file,'ab')
    np.savetxt(f, list_metrics_clsf, delimiter=",", fmt='%0.4f')
    f.close()




def loocv_stroke(X, Y, args):
    """
    LEAVE ONE-OUT CROSS VALIDATION ON STROKE DATA

    PARAMETERS
        X:      all dataset
        Y:      labels of classes
        model:  deep architecture and domain adaptation model
        output: output directory
        args:   input arguments
        draw_original_data_only: flag to draw original data, analysis of data.
    """
    list_metrics_clsf = []

    print("SEED:", args.seed)

    subjects = X.keys()

    foldNum = 0
    from codebase.models.dirtt import dirtt
    from codebase.train import train
    from codebase.datasets import Mdata

    # Iterate over fold_pairs
    for subj in subjects:

        print('Beginning fold ' + str(foldNum + 1) + ' of ' + str(len(subjects)))

        Sx_train = Sy_train = None
        i = 0

        for s in subjects:
            if s != subj:

                X_s = np.array(X[s])
                Y_s = np.array(Y[s])
                #standardize
                X_s, m, std = z_score(X_s)

                if i == 0:
                    Sx_train = X_s
                    Sy_train = Y_s
                else:
                    Sx_train = np.concatenate((Sx_train, X_s), axis=0)
                    Sy_train = np.concatenate((Sy_train, Y_s), axis=0)
                i += 1

        # Create Target domain
        X_target = np.array(X[subj])
        Y_target = np.array(Y[subj])

        # split data
        Tx_train, Ty_train, Tx_test, Ty_test = split_data(X_target, Y_target, args.seed, test_size=0.3)
        # standardize
        Tx_train, m, std = z_score(Tx_train)
        Tx_test = normalize(Tx_test, m, std)

        # labels
        y_classes = np.unique(Ty_train)

        print("Sx_train-shape:", Sx_train.shape, "Sy_train-shape:", Sy_train.shape)
        print("Tx_train-shape:", Tx_train.shape, "Ty_train-shape:", Ty_train.shape)
        print("Tx_test-shape:", Tx_test.shape, "Ty_test-shape:", Ty_test.shape)
        print("y_classes:", y_classes)


        # Source and target data
        src = Mdata(Sx_train, Sy_train, Sx_train, Sy_train)
        trg = Mdata(Tx_train, Ty_train, Tx_test, Ty_test)


        # Make model name
        setup = [
            ('model={:s}', 'dirtt'),
            ('data={:s}', args.data),
            ('name_data={:s}', args.target_data),
            ('nn={:s}', args.nn),
            ('trim={:d}', args.trim),
            ('dw={:.0e}', args.dw),
            ('bw={:.0e}', args.bw),
            ('sw={:.0e}', args.sw),
            ('tw={:.0e}', args.tw),
            ('dirt={:05d}', args.dirt),
            ('run={:04d}', 999)
        ]
        print
        model_name = '_'.join([t.format(v) for (t, v) in setup])
        model_name += "_subj=" + str(foldNum + 1)
        print("Model name:", model_name)

        # RESET GRAPH MODEL
        tf.reset_default_graph()
        # CREATE MODEL
        M = dirtt(input_shape_samples=(None, Sx_train.shape[1]))
        # Inizialization variables
        M.sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        # if It uses DIRT-T where it suppose that Ps(x)=Pt(x)
        if args.dirt > 0:
            # RESTORE PARAMETERS OF VADA
            run = args.run if args.run < 999 else 0
            setup = [
                ('model={:s}', 'dirtt'),
                ('data={:s}', args.data),
                ('name_data={:s}', args.target_data),
                ('nn={:s}', args.nn),
                ('trim={:d}', args.trim),
                ('dw={:.0e}', args.dw),
                ('bw={:.0e}', 0),
                ('sw={:.0e}', args.sw),
                ('tw={:.0e}', args.tw),
                ('dirt={:05d}', 0),
                ('run={:04d}', 999)
            ]
            vada_name = '_'.join([t.format(v) for (t, v) in setup])
            vada_name += "_subj=" + str(foldNum + 1)
            path = os.path.join('checkpoints', vada_name)
            print(path)
            path = tf.train.latest_checkpoint(path)
            saver.restore(M.sess, path)
            print("Restored from {}".format(path))

        disc = True if args.loss == "vada" or args.loss == "dann" else False

        # WOW! TRAIN MODEL
        classification_metrics = train(M, src, trg,
                                       saver=saver,
                                       has_disc=disc,
                                       model_name=model_name,
                                       generate_classification_metrics=True)
        print("\n")

        foldNum += 1

        # add to list
        list_metrics_clsf.append(classification_metrics)
        print()

        # To np array
    list_metrics_clsf = np.array(list_metrics_clsf)

    print("CLASSIFICATION METRICS:")
    for i in range(len(list_metrics_clsf[0])):
        mean = list_metrics_clsf[:, i].mean()
        print("Metric [", (i + 1), "] = ", list_metrics_clsf[:, i], " Mean:", mean)

    # Save Classification Metrics
    save_file = args.dir_resume + "/" + args.loss + "-stroke-metrics-classification.csv"
    f = open(save_file, 'ab')
    np.savetxt(f, list_metrics_clsf, delimiter=",", fmt='%0.4f')
    f.close()

        
def synthetic_classification(args):
    """
    DOMAIN ADAPTATION SYNTHETIC DATA
    *******************************************************
    Params
    args: arguments
    """
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.set_random_seed(args.seed)
    
    # IMPORT LIBRARYS Here to use "args" as variables in other FILES
    from codebase.models.dirtt import dirtt
    from codebase.train import train
    from codebase.datasets import get_data

    # Make model name
    setup = [
        ('model={:s}',  'dirtt'),
        ('data={:s}',    args.data),
        ('target_data={:s}', args.target_data),
        ('nn={:s}',     args.nn),
        ('trim={:d}',   args.trim),
        ('dw={:.0e}',   args.dw),
        ('bw={:.0e}',   args.bw),
        ('sw={:.0e}',   args.sw),
        ('tw={:.0e}',   args.tw),
        ('dirt={:05d}', args.dirt),
        ('run={:04d}',  args.run)
    ]
    model_name = '_'.join([t.format(v) for (t, v) in setup])
    print("Model name:", model_name)
    model = args.loss
    
    
    # RESET GRAPH MODEL
    tf.reset_default_graph()
    # CREATE MODEL
    M = dirtt(input_shape_samples=(None, 2))
    # Inizialization variables
    M.sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    # if It uses DIRT-T where it suppose that Ps(x)=Pt(x)
    if args.dirt > 0:
        # RESTORE PARAMETERS OF VADA
        args.run = args.run if args.run < 999 else 0
        setup = [
            ('model={:s}',  'dirtt'),
            ('data={:s}',    args.data),
            ('target_data={:s}', args.target_data),
            ('nn={:s}',     args.nn),
            ('trim={:d}',   args.trim),
            ('dw={:.0e}',   args.dw),
            ('bw={:.0e}',   0),
            ('sw={:.0e}',   args.sw),
            ('tw={:.0e}',   args.tw),
            ('dirt={:05d}', 0),
            ('run={:04d}',  999)
        ]
        vada_name = '_'.join([t.format(v) for (t, v) in setup])
        path = os.path.join('checkpoints', vada_name)
        path = tf.train.latest_checkpoint(path)

        saver.restore(M.sess, path)
        print("Restored from {}".format(path))
        
        model = "dirtt"
        
    # Obtains DATA
    src = get_data(args.data, args.source_data, args.seed)
    trg = get_data(args.data, args.target_data, args.seed)
    
    
    disc = True if args.loss == "vada" or args.loss == "dann" else False
    
    # WOW! TRAIN MODEL
    classification_metrics = train(M, src, trg,
          saver=saver,
          has_disc=disc,
          model_name=model_name,
          generate_classification_metrics=True)
    
    list_metrics_clsf = []
    # add to list
    list_metrics_clsf.append(classification_metrics)
    print()
    
    # To np array
    list_metrics_clsf = np.array(list_metrics_clsf)
    
    print("CLASSIFICATION METRICS:")
    for i in range(len(list_metrics_clsf[0])):
        mean = list_metrics_clsf[:,i].mean()
        print("Metric [",(i+1),"] = ", list_metrics_clsf[:,i]," Mean:", mean)
    
    np.savetxt(args.datadir+"/metrics-classification.csv", list_metrics_clsf, delimiter=",", fmt='%0.4f')
    
    # Save Classification Metrics
    save_file = args.dir_resume+"/"+model+"-"+args.target_data+"-metrics-classification.csv"
    f=open(save_file,'ab')
    np.savetxt(f, list_metrics_clsf, delimiter=",", fmt='%0.4f')
    f.close()
    
    

def main(args):
    # Argument overrides and additions
    # Number of classes
    src2Y = {'synth_moons': 2, 'synth_blobs': 3, 'seed': 3, 'seediv': 4, 'stroke':2}
    args.Y = src2Y[args.data]
    args.H = 32
    args.bw = args.bw if args.dirt > 0 else 0.  # mask bw when running VADA
    pprint(vars(args))
    
    # set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.set_random_seed(args.seed)
    
    if args.data in ["synth_blobs", "synth_moons"]:
        synthetic_classification(args)
        
    elif args.data == "seed":
        print("SESSION:", args.session)
        X, y = load_seed_i("/home/magdiel/Descargas/Datasets/SEED/", session=args.session, feature="de_LDS")
        loocv_emotions(X, y, args)
    
    elif args.data == "seediv":
        X, y = load_seed_iv("/home/magdiel/Descargas/Datasets/SEED-IV/", session=args.session, feature="de_LDS")
        loocv_emotions(X, y, args)

    elif args.data == "stroke":
        # It is not a good example
        X, y = load_stroke("/home/magdiel/Descargas/Datasets/Stroke/")
        loocv_stroke(X, y, args)
        
    else:
        raise Exception("Unknown dataset.")
    

if __name__ == '__main__':
  main(args)
