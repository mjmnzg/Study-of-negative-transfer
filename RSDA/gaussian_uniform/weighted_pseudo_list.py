import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from dataloader import ImageList,image_test
from gaussian_uniform.EM_for_gaussian_uniform import gauss_unif
import os
from modules import load_synth_data, load_seed, load_seed_iv, split_data, z_score, normalize, load_stroke



def sample_weighting(features, labels, pseu_labels, num_class=31, bottleneck_dim=256):

    features = features.numpy()
    labels = labels.numpy()
    pseu_labels = pseu_labels.numpy()

    # get indices from array
    id = np.arange(len(features))
    # reorder using pseudo labels
    sort_index = np.argsort(pseu_labels)

    # reorder according to
    clust_features = features[sort_index]
    clust_pseu_labels = pseu_labels[sort_index]
    clust_id = id[sort_index]

    # arrays to store
    weighted_id = np.empty([0], dtype=int)
    weighted_pseu_label = np.empty([0], dtype=int)
    weights = np.empty([0])

    for i in range(num_class):

        class_feature = clust_features[clust_pseu_labels == i]
        class_id = clust_id[clust_pseu_labels == i]
        class_mean = np.mean(class_feature, axis=0)

        class_mean = class_mean / (np.linalg.norm(class_mean) + 1e-10)
        R = np.linalg.norm(class_feature,axis=1)[0]

        # class_dist=np.arccos(np.sum(class_feature / R * class_mean.reshape(-1, 256), axis=1))
        class_dist = 1 - np.sum(class_feature / R * class_mean.reshape(-1, bottleneck_dim), axis=1)
        class_dist = class_dist - np.min(class_dist)
        class_dist[2 * np.arange(len(class_dist) // 2)] = -1 * class_dist[2 * np.arange(len(class_dist) // 2)]
        weight = gauss_unif(class_dist.reshape(-1, 1))

        # concatenate weights, id and pseudo-label
        weights = np.hstack((weights, weight))
        weighted_id = np.hstack((weighted_id, class_id))
        weighted_pseu_label = np.hstack((weighted_pseu_label, np.ones_like(class_id, dtype=int) * i))

    return weighted_id, weighted_pseu_label, weights


def make_list(id, pseu_label, weights, list_path, save_path):
    lists = open(list_path).readlines()
    labeled_list = [lists[id[i]].split(' ')[0] + ' ' + str(pseu_label[i]) + ' '
                    + str(weights[i])  for i in range(len(id))]
    fw = open(save_path, 'w')
    for l in labeled_list:
        fw.write(l)
        fw.write('\n')

def make_weighted_pseudo_list(args, model):

    if not os.path.exists('data/{}/pseudo_list'.format(args.dataset)):
        os.mkdir('data/{}/pseudo_list'.format(args.dataset))
    save_path = 'data/{}/pseudo_list/{}_{}_list.txt'.format(args.dataset, args.source, args.target)

    # Load data
    if args.dataset in ["synthetic"]:
        # load synthetic target data
        Tx, Ty, _, _ = load_synth_data(args.file_path, args.target, args.seed, test_size=0.3)

        # to tensor
        Tx_tensor = torch.Tensor(Tx)
        Ty_tensor = torch.Tensor(Ty)
        # to TensorDataset
        target_tr = TensorDataset(Tx_tensor, Ty_tensor)
        # to DataLoader
        dloader = DataLoader(target_tr, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    elif args.dataset in ["seed", "seed-iv"]:

        print("DATA:", args.dataset, " SESSION:", args.session)
        # Load imagined speech data
        if args.dataset == "seed":
            X, Y = load_seed(args.file_path, session=args.session, feature="de_LDS")
        else:
            X, Y = load_seed_iv(args.file_path, session=args.session)

        # get dictionary keys
        subjects = X.keys()

        i = 1
        selected_subject = int(args.target)
        trg_subj = -1

        for s in subjects:
            # if subject is not the selected for target
            if i == selected_subject:
                # store ID
                trg_subj = s
                break
            i += 1

        print("Target subject:", trg_subj)

        # Target dataset
        Tx = np.array(X[trg_subj])
        Ty = np.array(Y[trg_subj])

        # split data
        Tx, Ty, _, _ = split_data(Tx, Ty, args.seed, test_size=0.3)
        # Standardize target data
        Tx, _, _ = z_score(Tx)

        print("Tx_train:", Tx.shape, "Ty_train:", Ty.shape)

        # To tensor
        Tx_tensor = torch.Tensor(Tx)
        Ty_tensor = torch.Tensor(Ty)
        # To TensorDataset
        target_tr = TensorDataset(Tx_tensor, Ty_tensor)
        # To DataLoader
        dloader = DataLoader(target_tr, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    elif args.dataset in ["stroke"]:

        # Load imagined speech data
        X, Y = load_stroke(args.file_path)

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
            if i == selected_subject:
                # store ID
                trg_subj = s
                break
            i += 1

        print("Target subject:", trg_subj)

        # Target dataset
        Tx = np.array(X[trg_subj])
        Ty = np.array(Y[trg_subj])

        # split data
        Tx, Ty, _, _ = split_data(Tx, Ty, args.seed, test_size=0.3)
        print("Tx_train:", Tx.shape, "Ty_train:", Ty.shape)

        # Standardize Target domain
        Tx, m, std = z_score(Tx)

        # to tensor
        Tx_tensor = torch.Tensor(Tx)
        Ty_tensor = torch.Tensor(Ty)

        # To TensorDataset
        target_tr = TensorDataset(Tx_tensor, Ty_tensor)
        # To DataLoader
        dloader = DataLoader(target_tr, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    else:
        print("This dataset cannot be loaded in memory")
        exit()

    samples = torch.Tensor([])
    features = torch.Tensor([])
    labels = torch.LongTensor([])
    pseu_labels = torch.LongTensor([])

    # make predictions
    with torch.no_grad():
        for data in dloader:

            sample = data[0]
            label = data[1]
            sample = sample.cuda()
            # obtain predictions
            feature, outputs = model(sample)

            samples = torch.cat([samples, sample.cpu()], dim=0)
            features = torch.cat([features, feature.cpu()], dim=0)
            labels = torch.cat([labels, label], dim=0)
            pseu_labels = torch.cat([pseu_labels, torch.argmax(outputs.cpu(), dim=1)], dim=0)

    # sample weighting
    weighted_id, weighted_pseu_label, weights = sample_weighting(features, labels, pseu_labels, num_class=args.num_class, bottleneck_dim=args.bottleneck_dim)

    samples = samples[weighted_id]

    #print("LEN SAMPLES:", len(samples))
    #print("LEN IDS:", len(weighted_id))

    # save samples in new order
    np.savetxt("./data/weighted_ids.csv", weighted_id, delimiter=",", fmt='%0.4f')
    # save pseudo-labels in new order
    np.savetxt("./data/pseudo-labels.csv", weighted_pseu_label, delimiter=",", fmt='%0.4f')
    # save weights in new order
    np.savetxt("./data/weights.csv", weights, delimiter=",", fmt='%0.4f')

    #make_list(weighted_id, weighted_pseu_label, weights, list_path, save_path)

    return samples, weighted_pseu_label, weights




