import torch
import network
from dataloader import *
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import lr_schedule
import copy
import os
import utils
import torch.nn.functional as F
from modules import load_synth_data, PseudoLabeledData, load_seed, load_seed_iv, split_data,\
    z_score, normalize, load_stroke
import numpy as np



def classification(loader, model):
    start_test = True
    with torch.no_grad():
        # get iterate data
        iter_test = iter(loader["test"])
        for i in range(len(loader['test'])):
            # get sample and label
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            # load in gpu
            inputs = inputs.type(torch.FloatTensor).cuda()
            labels = labels
            # obtain predictions
            _, outputs = model(inputs)
            # concatenate predictions
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    # obtain labels
    _, predict = torch.max(all_output, 1)
    # calculate accuracy for all examples
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    return accuracy


def train_init(args):
    """
    Pretraining using DANN or MSTN.
    Parameters:
        @args: arguments
    """
    # --------------------------
    # Prepare data
    # --------------------------

    dset_loaders = {}
    # Synthetic data
    if args.dataset in ["synthetic"]:
        # load source data
        Sx, Sy, _, _ = load_synth_data(args.file_path, args.source, args.seed, test_size=0.3)
        # load target data
        Tx, Ty, Vx, Vy = load_synth_data(args.file_path, args.target, args.seed, test_size=0.3)

        # Convert to tensor
        Sx_tensor = torch.Tensor(Sx)
        Sy_tensor = torch.Tensor(Sy)
        Tx_tensor = torch.Tensor(Tx)
        Ty_tensor = torch.Tensor(Ty)
        Vx_tensor = torch.Tensor(Vx)
        Vy_tensor = torch.Tensor(Vy)

        # Load in TensorDataset
        source_tr = TensorDataset(Sx_tensor, Sy_tensor)
        # Load in TensorDataset
        target_tr = TensorDataset(Tx_tensor, Ty_tensor)
        target_ts = TensorDataset(Vx_tensor, Vy_tensor)

        # Load in DataLoader
        dset_loaders["source"] = DataLoader(source_tr, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        dset_loaders["target"] = DataLoader(target_tr, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        dset_loaders["test"] = DataLoader(target_ts, batch_size=2 * args.batch_size, shuffle=False, num_workers=4)

        print("Synthetic data were succesfully loaded")

    # Load SEED and SEED-IV data
    elif args.dataset in ["seed", "seed-iv"]:
        print("DATA:", args.dataset, " SESSION:", args.session)
        # Load imagined speech data
        if args.dataset == "seed":
            X, Y = load_seed(args.file_path, session=args.session, feature="de_LDS")
        else:
            X, Y = load_seed_iv(args.file_path, session=args.session)
        # get dictionary keys
        subjects = X.keys()

        # SOURCE DATASET
        Sx = Sy = None
        i = 1
        flag = False
        selected_subject = int(args.target)
        trg_subj = -1

        for s in subjects:
            # if subject is not the selected for target
            if i != selected_subject:
                # obtain data from subject 's'
                x_tr = np.array(X[s])
                y_tr = np.array(Y[s])

                # Standardize training data
                x_tr, m, std = z_score(x_tr)

                if not flag:
                    # initiliaze data
                    Sx = x_tr
                    Sy = y_tr
                    flag = True
                else:
                    # concatenate data to array
                    Sx = np.concatenate((Sx, x_tr), axis=0)
                    Sy = np.concatenate((Sy, y_tr), axis=0)
            else:
                # store ID
                trg_subj = s

            i += 1

        print("Target subject:", trg_subj)

        # Target data
        Tx = np.array(X[trg_subj])
        Ty = np.array(Y[trg_subj])

        # Split target data for testing
        Tx, Ty, Vx, Vy = split_data(Tx, Ty, args.seed, test_size=0.3)

        # Standardize target data
        Tx, m, std = z_score(Tx)
        Vx = normalize(Vx, mean=m, std=std)

        print("Sx_train:", Sx.shape, "Sy_train:", Sy.shape)
        print("Tx_train:", Tx.shape, "Ty_train:", Ty.shape)
        print("Tx_test:", Vx.shape, "Ty_test:", Vy.shape)

        # Convert to tensor
        Sx_tensor = torch.Tensor(Sx)
        Sy_tensor = torch.Tensor(Sy)
        Tx_tensor = torch.Tensor(Tx)
        Ty_tensor = torch.Tensor(Ty)
        Vx_tensor = torch.Tensor(Vx)
        Vy_tensor = torch.Tensor(Vy)

        # Store data in TensorDataset
        source_tr = TensorDataset(Sx_tensor, Sy_tensor)
        target_tr = TensorDataset(Tx_tensor, Ty_tensor)
        target_ts = TensorDataset(Vx_tensor, Vy_tensor)

        # Load data in DataLoader
        dset_loaders["source"] = DataLoader(source_tr, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        dset_loaders["target"] = DataLoader(target_tr, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        dset_loaders["test"] = DataLoader(target_ts, batch_size=2 * args.batch_size, shuffle=False, num_workers=4)

        print("Data were succesfully loaded")


    elif args.dataset in ["stroke"]:
        # Load Stroke data
        X, Y = load_stroke(args.file_path)
        # get dictionary keys
        subjects = X.keys()
        # Load source data
        Sx = Sy = None
        i = 1
        flag = False
        select_subject = int(args.target)
        trg_subj = -1

        for s in subjects:
            # if subject is not the selected for target
            if i != select_subject:
                # Obtain data from subject 's'
                tr_x = np.array(X[s])
                tr_y = np.array(Y[s])

                # Standardize training data
                tr_x, m, std = z_score(tr_x)

                # Append data
                if not flag:
                    Sx = tr_x
                    Sy = tr_y
                    flag = True
                else:
                    Sx = np.concatenate((Sx, tr_x), axis=0)
                    Sy = np.concatenate((Sy, tr_y), axis=0)
            else:
                # store ID
                trg_subj = s
            i += 1

        print("Target subject:", trg_subj)

        # Load target data
        Tx = np.array(X[trg_subj])
        Ty = np.array(Y[trg_subj])

        # split data in train and test sets
        Tx, Ty, Vx, Vy = split_data(Tx, Ty, args.seed, test_size=0.3)

        # Standardize target data
        Tx, m, std = z_score(Tx)
        Vx = normalize(Vx, mean=m, std=std)

        print("Sx_train:", Sx.shape, "Sy_train:", Sy.shape)
        print("Tx_train:", Tx.shape, "Ty_train:", Ty.shape)
        print("Tx_test:", Vx.shape, "Ty_test:", Vy.shape)

        # Convert to tensor
        Sx_tensor = torch.Tensor(Sx)
        Sy_tensor = torch.Tensor(Sy)
        Tx_tensor = torch.Tensor(Tx)
        Ty_tensor = torch.Tensor(Ty)
        Vx_tensor = torch.Tensor(Vx)
        Vy_tensor = torch.Tensor(Vy)

        # Store data in TensorDataset
        source_tr = TensorDataset(Sx_tensor, Sy_tensor)
        target_tr = TensorDataset(Tx_tensor, Ty_tensor)
        target_ts = TensorDataset(Vx_tensor, Vy_tensor)

        # Load data in DataLoader
        dset_loaders["source"] = DataLoader(source_tr, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        dset_loaders["target"] = DataLoader(target_tr, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        dset_loaders["test"] = DataLoader(target_ts, batch_size=2 * args.batch_size, shuffle=False, num_workers=4)

        print("Stroke data were succesfully loaded")

    else:
        print("This dataset does not exist.")
        exit(-1)


    # --------------------------
    # Create Deep Neural Network
    # --------------------------
    # For synthetic dataset
    if args.dataset in ["synthetic"]:
        # Define Neural Network
        model = network.ShallowNet(input_size=2, hidden_size=64, class_num=args.num_class, radius=args.radius).cuda()
        # Define Adversarial net
        adv_net = network.Discriminator(in_feature=model.output_num(), radius=args.radius, hidden_size=64).cuda()

    elif args.dataset in ["seed", "seed-iv"]:
        # Define Neural Network
        model = network.DFN(input_size=310, hidden_size=256, bottleneck_dim=args.bottleneck_dim, class_num=args.num_class, radius=args.radius).cuda()
        # Define Adversarial net
        adv_net = network.Discriminator(in_feature=model.output_num(), radius=args.radius, hidden_size=256).cuda()

    elif args.dataset in ["stroke"]:
        # Define Neural Network
        model = network.DFN(input_size=25, hidden_size=100, bottleneck_dim=args.bottleneck_dim, class_num=args.num_class, radius=args.radius).cuda()
        # Define Adversarial net
        adv_net = network.Discriminator(in_feature=model.output_num(), radius=args.radius, hidden_size=100).cuda()

    else:
        print("A neural network for this dataset has not been selected yet.")
        exit(-1)

    # Get network weights (parameters)
    parameter_list = model.get_parameters() + adv_net.get_parameters()

    # Define optimizer
    optimizer = torch.optim.SGD(parameter_list, lr=args.lr_a, momentum=0.9, weight_decay=0.005)

    # if gpus are availables
    gpus = args.gpu_id.split(',')
    if len(gpus) > 1:
        adv_net = nn.DataParallel(adv_net, device_ids=[int(i) for i in gpus])
        model = nn.DataParallel(model, device_ids=[int(i) for i in gpus])

    # ------------------------
    # Model training
    # ------------------------

    # length of the source and target domains
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])

    # copy the best model
    best_model = copy.deepcopy(model)

    # Number of centroids for semantic loss
    if args.dataset in ["synthetic"]:
        Cs_memory = torch.zeros(args.num_class, 64).cuda()
        Ct_memory = torch.zeros(args.num_class, 64).cuda()

    elif args.dataset in ["seed", "seed-iv"]:
        Cs_memory = torch.zeros(args.num_class, 256).cuda()
        Ct_memory = torch.zeros(args.num_class, 256).cuda()

    elif args.dataset in ["stroke"]:
        Cs_memory = torch.zeros(args.num_class, 100).cuda()
        Ct_memory = torch.zeros(args.num_class, 100).cuda()

    else:
        print("SETTING number of centroids: The dataset does not exist.")
        exit()


    # iterate until to achieve a number of maximum iterations
    for i in range(args.max_iter1):

        # Test model each certain iterations
        if i % args.test_interval == args.test_interval - 1:

            # set model to test
            model.train(False)
            # calculate accuracy performance
            best_acc = classification(dset_loaders, model)
            best_model = copy.deepcopy(model)

            # print accuracy
            log_str = "Iter: {:05d}, \t accuracy: {:.4f}".format(i, best_acc)
            args.log_file.write(log_str)
            args.log_file.flush()
            print(log_str)

        # save network weights
        if i % args.snapshot_interval == args.snapshot_interval -1:
            if not os.path.exists('snapshot'):
                os.mkdir('snapshot')
            if not os.path.exists('snapshot/save'):
                os.mkdir('snapshot/save')
            torch.save(best_model, 'snapshot/save/initial_model.pk')

        # Enable model to train
        model.train(True)
        adv_net.train(True)

        # schedule for learning rate
        optimizer = lr_schedule.inv_lr_scheduler(optimizer,i)

        # if we achieve the size of the dataset
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])

        # Get batch for source and target domains
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()

        # get features and predictions
        features_source, outputs_source = model(inputs_source)
        features_target, outputs_target = model(inputs_target)

        # concatenate features
        features = torch.cat((features_source, features_target), dim=0)

        # to Long
        labels_source = labels_source.type(torch.LongTensor).cuda()

        # set cross-entropy loss
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)

        # set adversarial loss
        adv_loss = utils.loss_adv(features, adv_net)

        # select weight factor
        if args.baseline == 'MSTN':
            lam = network.calc_coeff(i)
        elif args.baseline =='DANN':
            lam = 0.0

        # pseudo-labels
        pseu_labels_target = torch.argmax(outputs_target, dim=1)

        # calculate semantic loss
        loss_sm, Cs_memory, Ct_memory = utils.SM(features_source, features_target, labels_source, pseu_labels_target, Cs_memory, Ct_memory)

        # Total loss: classifier loss + adversarial loss + semantic loss
        total_loss = classifier_loss + adv_loss + lam * loss_sm

        # Compute gradients
        total_loss.backward()
        # Update weights
        optimizer.step()
        # Set gradients to zero in tensors
        optimizer.zero_grad()

        # print iteration
        #print('step:{: d},\t,class_loss:{:.4f},\t,adv_loss:{:.4f}'.format(i, classifier_loss.item(), adv_loss.item()))
        Cs_memory.detach_()
        Ct_memory.detach_()

    return best_acc, best_model




def train(args, samples, weighted_pseu_label, weights):

    # prepare data
    dset_loaders = {}

    if args.dataset in ["synthetic"]:
        # load source data
        Sx, Sy, Vsx, Vsy = load_synth_data(args.file_path, args.source, args.seed, test_size=0.3)
        # load target data
        Tx, Ty, Tsx, Tsy = load_synth_data(args.file_path, args.target, args.seed, test_size=0.3)

        # To tensor
        Sx_tensor = torch.tensor(Sx)
        Sy_tensor = torch.tensor(Sy)
        # create containers for source data
        source_tr = TensorDataset(Sx_tensor, Sy_tensor)


        # Create container for target data
        target_tr = PseudoLabeledData(samples, weighted_pseu_label, weights)

        Tsx_tensor = torch.tensor(Tsx)
        Tsy_tensor = torch.tensor(Tsy)

        # create container for test data
        target_ts = TensorDataset(Tsx_tensor, Tsy_tensor)

        # data loader
        dset_loaders["source"] = DataLoader(source_tr, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        dset_loaders["target"] = DataLoader(target_tr, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        dset_loaders["test"] = DataLoader(target_ts, batch_size=2 * args.batch_size, shuffle=False, num_workers=4)

        print("Synthetic data were succesfully loaded")

    elif args.dataset in ["seed", "seed-iv"]:

        print("DATA:", args.dataset, " SESSION:", args.session)
        # Load imagined speech data
        if args.dataset == "seed":
            X, Y = load_seed(args.file_path, session=args.session, feature="de_LDS")
        else:
            X, Y = load_seed_iv(args.file_path, session=args.session)

        # get dictionary keys
        subjects = X.keys()

        # build Source dataset
        Sx = Sy = None
        i = 1
        flag = False
        selected_subject = int(args.target)
        trg_subj = -1

        for s in subjects:
            # if subject is not the selected for target
            if i != selected_subject:

                tr_x = np.array(X[s])
                tr_y = np.array(Y[s])

                # Standardize training data
                tr_x, m, std = z_score(tr_x)

                if not flag:
                    Sx = tr_x
                    Sy = tr_y
                    flag = True
                else:
                    Sx = np.concatenate((Sx, tr_x), axis=0)
                    Sy = np.concatenate((Sy, tr_y), axis=0)
            else:
                # store ID
                trg_subj = s
            i += 1

        print("Target subject:", trg_subj)

        # Target dataset
        Tx = np.array(X[trg_subj])
        Ty = np.array(Y[trg_subj])
        # split data
        Tx, Ty, Tsx, Tsy = split_data(Tx, Ty, args.seed, test_size=0.3)
        # Standardize Target domain
        Tx, m, std = z_score(Tx)
        Tsx = normalize(Tsx, mean=m, std=std)

        print("Sx_train:", Sx.shape, "Sy_train:", Sy.shape)
        print("Tx_train:", Tx.shape, "Ty_train:", Ty.shape)
        print("Tx_test:", Tsx.shape, "Ty_test:", Tsy.shape)

        # to tensor
        Sx_tensor = torch.tensor(Sx)
        Sy_tensor = torch.tensor(Sy)

        # create containers for source data
        source_tr = TensorDataset(Sx_tensor, Sy_tensor)

        # create container for target data
        target_tr = PseudoLabeledData(samples, weighted_pseu_label, weights)

        # create container for test data
        Tsx_tensor = torch.tensor(Tsx)
        Tsy_tensor = torch.tensor(Tsy)
        target_ts = TensorDataset(Tsx_tensor, Tsy_tensor)

        # data loader
        dset_loaders["source"] = DataLoader(source_tr, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        dset_loaders["target"] = DataLoader(target_tr, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        dset_loaders["test"] = DataLoader(target_ts, batch_size=2 * args.batch_size, shuffle=False, num_workers=4)

        print("Data were succesfully loaded")



    elif args.dataset in ["stroke"]:
        # Load imagined speech data
        X, Y = load_stroke(args.file_path)

        # get dictionary keys
        subjects = X.keys()

        # build Source dataset
        Sx = Sy = None
        i = 1
        flag = False
        select_subject = int(args.target)
        trg_subj = -1

        for s in subjects:
            # if subject is not the selected for target
            if i != select_subject:

                x = np.array(X[s])
                y = np.array(Y[s])
                # split data
                tr_x, tr_y, va_x, va_y = split_data(x, y, args.seed, test_size=0.05)

                # Standardize training data
                tr_x, m, std = z_score(tr_x)

                if not flag:
                    Sx = tr_x
                    Sy = tr_y
                    flag = True
                else:
                    Sx = np.concatenate((Sx, tr_x), axis=0)
                    Sy = np.concatenate((Sy, tr_y), axis=0)
            else:
                # store ID
                trg_subj = s
            i += 1

        print("Target subject:", trg_subj)

        # Target dataset
        Tx = np.array(X[trg_subj])
        Ty = np.array(Y[trg_subj])

        # split data
        Tx, Ty, Tsx, Tsy = split_data(Tx, Ty, args.seed, test_size=0.3)

        # Standardize Target domain
        Tx, m, std = z_score(Tx)
        Tsx = normalize(Tsx, mean=m, std=std)

        print("Sx_train:", Sx.shape, "Sy_train:", Sy.shape)
        print("Tx_train:", Tx.shape, "Ty_train:", Ty.shape)
        print("Tx_test:", Tsx.shape, "Ty_test:", Tsy.shape)


        # create containers for source data
        # to tensor
        Sx_tensor = torch.tensor(Sx)
        Sy_tensor = torch.tensor(Sy)
        source_tr = TensorDataset(Sx_tensor, Sy_tensor)

        # create container for target data
        target_tr = PseudoLabeledData(samples, weighted_pseu_label, weights)

        # create container for test data
        Tsx_tensor = torch.tensor(Tsx)
        Tsy_tensor = torch.tensor(Tsy)
        target_ts = TensorDataset(Tsx_tensor, Tsy_tensor)

        # Load in DataLoader
        dset_loaders["source"] = DataLoader(source_tr, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        dset_loaders["target"] = DataLoader(target_tr, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        dset_loaders["test"] = DataLoader(target_ts, batch_size=2 * args.batch_size, shuffle=False, num_workers=4)

        print("Stroke data were succesfully loaded")

    else:
        print("This dataset does not exist.")
        exit()


    # Create model
    if args.dataset in ["synthetic"]:
        # setting Neural Network
        model = network.ShallowNet(input_size=2, hidden_size=64, bottleneck_dim=args.bottleneck_dim, class_num=args.num_class, radius=5.0).cuda()
        # setting Adversarial net
        adv_net = network.Discriminator(in_feature=model.output_num(), radius=5.0, hidden_size=64, max_iter=2000).cuda()

    elif args.dataset in ["seed", "seed-iv"]:

        # setting Neural Network
        model = network.DFN(input_size=310, hidden_size=args.bottleneck_dim, bottleneck_dim=256, class_num=args.num_class, radius=10.0).cuda()
        # setting Adversarial net
        adv_net = network.Discriminator(in_feature=model.output_num(), radius=10.0, hidden_size=256).cuda()

    elif args.dataset in ["stroke"]:
        # setting Neural Network
        model = network.DFN(input_size=25, hidden_size=args.bottleneck_dim, bottleneck_dim=100, class_num=args.num_class, radius=10.0).cuda()
        # setting Adversarial net
        adv_net = network.Discriminator(in_feature=model.output_num(), radius=10.0, hidden_size=100).cuda()

    else:
        print("A neural network for this dataset has not been selected yet.")
        exit(-1)

    # Ger trainable weights
    parameter_classifier = [model.get_parameters()[2]]
    parameter_feature = model.get_parameters()[0:2] + adv_net.get_parameters()

    # set optimizers
    #optimizer_classifier = torch.optim.SGD(parameter_classifier, lr=args.lr_b, momentum=0.9, weight_decay=0.005)
    #optimizer_feature = torch.optim.SGD(parameter_feature, lr=args.lr_b, momentum=0.9, weight_decay=0)

    optimizer_classifier = torch.optim.Adam(parameter_classifier, lr=args.lr_b)
    optimizer_feature = torch.optim.Adam(parameter_feature, lr=args.lr_b)

    # if number of GPUS is greater 1
    gpus = args.gpu_id.split(',')
    if len(gpus) > 1:
        adv_net = nn.DataParallel(adv_net, device_ids=[int(i) for i in gpus])
        model = nn.DataParallel(model, device_ids=[int(i) for i in gpus])

    ## Train MODEL

    # lenght of data
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])

    # auxiliar variables
    best_acc = 0.0
    best_model = copy.deepcopy(model)

    # centroids for each cluster
    if args.dataset in ["synthetic"]:
        Cs_memory = torch.zeros(args.num_class, 64).cuda()
        Ct_memory = torch.zeros(args.num_class, 64).cuda()

    elif args.dataset in ["seed", "seed-iv"]:
        Cs_memory = torch.zeros(args.num_class, 256).cuda()
        Ct_memory = torch.zeros(args.num_class, 256).cuda()

    elif args.dataset in ["stroke"]:
        Cs_memory = torch.zeros(args.num_class, 100).cuda()
        Ct_memory = torch.zeros(args.num_class, 100).cuda()
    else:
        print("The number of centroids for this dataset has not been selected yet.")
        exit()

    # iterate over
    for i in range(args.max_iter2):

        # Testing phase
        if i % args.test_interval == args.test_interval - 1:
            # set model training to False
            model.train(False)
            # calculate accuracy on test set
            best_acc = classification(dset_loaders, model)
            best_model = copy.deepcopy(model)
            # print accuracies
            log_str = "iter: {:05d}, \t accuracy: {:.4f}".format(i, best_acc)
            args.log_file.write(log_str)
            args.log_file.flush()
            print(log_str)

        # Save model
        if i % args.snapshot_interval == args.snapshot_interval -1:
            if not os.path.exists('snapshot'):
                os.mkdir('snapshot')
            if not os.path.exists('snapshot/save'):
                os.mkdir('snapshot/save')
            torch.save(best_model,'snapshot/save/best_model.pk')


        # Enable model for training
        model.train(True)
        adv_net.train(True)

        # obtain schedule for learning rate
        optimizer_classifier = lr_schedule.inv_lr_scheduler(optimizer_classifier, i)
        optimizer_feature = lr_schedule.inv_lr_scheduler(optimizer_feature, i)

        # get data
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])

        # Get batch for source and target domains
        inputs_source_, labels_source = iter_source.next()
        inputs_target, pseudo_labels_target, weights = iter_target.next()
        # Cast
        inputs_source_ = inputs_source_.type(torch.FloatTensor)
        labels_source = labels_source.type(torch.LongTensor)
        # to cuda
        inputs_source, labels_source = inputs_source_.cuda(),  labels_source.cuda()
        inputs_target, pseudo_labels_target = inputs_target.cuda(), pseudo_labels_target.cuda()
        weights = weights.type(torch.Tensor).cuda()

        # get features and labels for source and target domain
        features_source, outputs_source = model(inputs_source)
        features_target, outputs_target = model(inputs_target)

        # concatenate features
        features = torch.cat((features_source, features_target), dim=0)

        # cross-entropy loss
        source_class_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)

        # adversarial loss
        adv_loss = utils.loss_adv(features,adv_net)
        # entropy loss
        H = torch.mean(utils.Entropy(F.softmax(outputs_target, dim=1)))
        # function robust loss
        target_robust_loss = utils.robust_pseudo_loss(outputs_target,pseudo_labels_target,weights)

        # classifier loss
        classifier_loss = source_class_loss + target_robust_loss

        # reset gradients
        optimizer_classifier.zero_grad()
        # compute gradients
        classifier_loss.backward(retain_graph=True)


        if args.baseline == 'MSTN':
            lam = network.calc_coeff(i, max_iter=1000)
        elif args.baseline =='DANN':
            lam = 0.0

        # obtain pseudo labels
        pseu_labels_target = torch.argmax(outputs_target, dim=1)
        # semantic loss
        loss_sm, Cs_memory, Ct_memory = utils.SM(features_source, features_target, labels_source, pseu_labels_target, Cs_memory, Ct_memory)

        # TOTAL LOSS
        feature_loss = classifier_loss + adv_loss + lam * loss_sm + lam * H
        # reset gradients
        optimizer_feature.zero_grad()
        # compute gradients
        feature_loss.backward()

        # update parameters
        optimizer_classifier.step()
        # update weights
        optimizer_feature.step()


        #print('step:{: d},\t,source_class_loss:{:.4f},\t,target_robust_loss:{:.4f}'
        #      ''.format(i, source_class_loss.item(),target_robust_loss.item()))

        Cs_memory.detach_()
        Ct_memory.detach_()

    return best_acc, best_model









