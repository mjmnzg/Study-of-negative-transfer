import tensorflow as tf
import tensorbayes as tb
from codebase.args import args
from codebase.datasets import PseudoData, get_info
from codebase.utils import delete_existing, save_value, save_model, reformat_target_digits
import os
from sklearn.metrics import f1_score,accuracy_score
import copy
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


def update_dict(M, feed_dict, src=None, trg=None, bs=100):
    """Update feed_dict with new mini-batch

    M         - (TensorDict) the model
    feed_dict - (dict) tensorflow feed dict
    src       - (obj) source domain. Contains train/test Data obj
    trg       - (obj) target domain. Contains train/test Data obj
    bs        - (int) batch size
    """
    if src:
        src_x, src_y = src.train.next_batch(bs)
        feed_dict.update({M.src_x: src_x, M.src_y: src_y})

    if trg:
        trg_x, trg_y = trg.train.next_batch(bs)
        feed_dict.update({M.trg_x: trg_x, M.trg_y: trg_y})
        

def train(M, src=None, trg=None, has_disc=True, 
          saver=None, model_name=None,
          generate_classification_metrics=False):
    """Main training function

    Creates log file, manages datasets, trains model

    M          - (TensorDict) the model
    src        - (obj) source domain. Contains train/test Data obj
    trg        - (obj) target domain. Contains train/test Data obj
    has_disc   - (bool) whether model requires a discriminator update
    saver      - (Saver) saves models during training
    model_name - (str) name of the model being run with relevant parms info
    generate_decision_regions_image - (bool) Flag to generate decision regions
    generate_distribution_metrics - (bool) Flag to generate measures from distributions
    """
    # Training settings
    bs = args.batch_size # cognitive data
    iterep = src.train.samples.shape[0]//(args.batch_size)
    itersave = 500
    n_epoch = args.num_epochs
    epoch = 0
    feed_dict = {}
    
    # Reformat labels Test DATa
    Ty_train = reformat_target_digits(trg.train.labels)
    Ty_test = reformat_target_digits(trg.test.labels)

    # Get number of parameters
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *=dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters

    print("Number of parameters:", total_parameters)

    # Create a save directory
    if saver:
        model_dir = os.path.join('checkpoints', model_name)
        delete_existing(model_dir, args.run)
        os.makedirs(model_dir)

    # Replace src domain with psuedolabeled trg
    if args.dirt > 0:
        print("Setting backup and updating backup model")
        # backup date source
        source_data = copy.copy(src)
        
        src = PseudoData(args.target_data, trg, M.teacher)
        M.sess.run(M.update_teacher)
        
        # Sanity check model
        print_list = []

        if trg:
            
            # CALCULATE CLASSIFICATION MEASURES ON TEST DATA
                y_preds_train = M.sess.run(M.fn_ema_p,feed_dict={M.test_x:trg.train.samples, M.prob:1.0, M.phase:False})
                y_preds_test = M.sess.run(M.fn_ema_p,feed_dict={M.test_x:trg.test.samples, M.prob:1.0, M.phase:False})

                acc_train_target = accuracy_score(Ty_train, y_preds_train)
                acc_test_target = accuracy_score(Ty_test, y_preds_test)
                
                print_list += ['trg_train_ema', acc_train_target]
                print_list += ['trg_test_ema', acc_test_target]
            

        print(print_list)

    if src: get_info(args.data, src)
    if trg: get_info(args.data, trg)
    print("Batch size:", bs)
    print("Iterep:", iterep)
    print("Total iterations:", n_epoch * iterep)
    #print("Log directory:", log_dir)
    
    
    for i in range(n_epoch * iterep):
        
        # Run DISCRIMINATOR optimizer [part II -DISCRIMINATOR MODEL]
        if has_disc:
            update_dict(M, feed_dict, src, trg, bs)
            feed_dict.update({M.prob: 0.5})
            summary, _ = M.sess.run(M.ops_disc, feed_dict)
        
        
        # Run MAIN optimizer [part I - All loss function named MAIN LOSS]
        
        update_dict(M, feed_dict, src, trg, bs)
        feed_dict.update({M.prob: 0.5, M.phase:True})
        
        summary, _ = M.sess.run(M.ops_main, feed_dict)
        
        # show bar progress
        end_epoch, epoch = tb.utils.progbar(i, iterep,
                                            message='{}/{}'.format(epoch, i),
                                            display=args.run >= 999)



        # Update pseudolabeler (ONLY IF WE ARE USING DIRT-T)
        if args.dirt and (i + 1) % args.dirt == 0:
            print("Updating teacher model")
            M.sess.run(M.update_teacher)

        # Log end-of-epoch values
        if end_epoch:
            
            # EVALUATE train and test sets of TARGET DATA 
            if trg:
                print_list = M.sess.run(M.ops_print, feed_dict)
                
                # CALCULATE CLASSIFICATION MEASURES ON TEST DATA
                y_preds_train = M.sess.run(M.fn_ema_p,feed_dict={M.test_x:trg.train.samples, M.prob:1.0, M.phase:False})
                y_preds_test = M.sess.run(M.fn_ema_p,feed_dict={M.test_x:trg.test.samples, M.prob:1.0, M.phase:False})

                acc_train_target = accuracy_score(Ty_train, y_preds_train)
                acc_test_target = accuracy_score(Ty_test, y_preds_test)
                
                print_list += ['trg_train_ema', acc_train_target]
                print_list += ['trg_test_ema', acc_test_target]
                
                
                # CALCULATE CLASSIFICATION MEASURES ON TEST DATA
                y_preds_train = M.sess.run(M.fn_p,feed_dict={M.test_x:trg.train.samples, M.prob:1.0, M.phase:False})
                y_preds_test = M.sess.run(M.fn_p,feed_dict={M.test_x:trg.test.samples, M.prob:1.0, M.phase:False})

                acc_train_target = accuracy_score(Ty_train, y_preds_train)
                acc_test_target = accuracy_score(Ty_test, y_preds_test)
                
                print_list += ['trg_train', acc_train_target]
                print_list += ['trg_test', acc_test_target]
            
            print_list += ['epoch', epoch]

            #if (epoch) % 15 == 0:
                #M.lr = M.sess.run(M.learning.assign(M.lr * 0.9))
                #print("New learning rate", M.lr)
                #input("SO DAR")
            
            print(print_list)
    
    
    # restore source data
    if args.dirt > 0:
        src = source_data
    
    list_metrics_classification = []
    
    if generate_classification_metrics:
        # ==================================
        # GET measurements of CLASSIFICATION
        # ==================================
        # classifier
        print("EMA:",args.use_ema)
        if args.use_ema:
            classifier = M.fn_ema_p
            print("USING EMA...")
        else:
            classifier = M.fn_p
        
        
        # Reformat labels Test DATa
        Ty_train = reformat_target_digits(trg.train.labels)
        Ty_test = reformat_target_digits(trg.test.labels)
        
        # CALCULATE CLASSIFICATION MEASURES ON TEST DATA
        y_preds_train = M.sess.run(classifier,feed_dict={M.test_x:trg.train.samples, M.prob:1.0, M.phase:False})
        y_preds_test = M.sess.run(classifier,feed_dict={M.test_x:trg.test.samples, M.prob:1.0, M.phase:False})
        
        acc_train_target = accuracy_score(Ty_train, y_preds_train)
        acc_test_target = accuracy_score(Ty_test, y_preds_test)
        print("accuracy_train_target:",acc_train_target, "    accuracy_test_target:",acc_test_target)
        
        # F1 SCORE
        f1_train_target = f1_score(Ty_train, y_preds_train, average="weighted")
        f1_test_target = f1_score(Ty_test, y_preds_test, average="weighted")
        print("f1_train_target:",f1_train_target, "    f1_test_target:",f1_test_target)
        
        
        y1 = label_binarize(Ty_train, classes=range(args.Y))
        y2 = label_binarize(y_preds_train, classes=range(args.Y))
        auc_train_target = roc_auc_score(y1, y2, average='weighted')
        
        y1 = label_binarize(Ty_test, classes=range(args.Y))
        y2 = label_binarize(y_preds_test, classes=range(args.Y))
        auc_test_target = roc_auc_score(y1, y2, average='weighted')
        print("auc_train_target:",auc_train_target, "    auc_test_target:",auc_test_target)
        
        # metrics to classification
        list_metrics_classification.append(acc_train_target)
        list_metrics_classification.append(acc_test_target)
        list_metrics_classification.append(f1_train_target)
        list_metrics_classification.append(f1_test_target)
        list_metrics_classification.append(auc_train_target)
        list_metrics_classification.append(auc_test_target)
    
    # Saving final model
    if saver:
        save_model(saver, M, model_dir, i + 1)
    
    return list_metrics_classification
