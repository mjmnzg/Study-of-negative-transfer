import argparse
from solvers import train_init, train
from gaussian_uniform.weighted_pseudo_list import make_weighted_pseudo_list
import copy
import torch
import os
import numpy as np
import random

def main(args):
    args.log_file.write('\n\n###########  initialization ############')
    
    # pre-training using DANN or MSTN
    acc, model = train_init(args)

    # store the best accuracy
    best_model = copy.deepcopy(model)

    # RSDA
    for stage in range(args.stages):
        
        print('\n\n########### stage : {:d}th ##############\n\n'.format(stage+1))
        args.log_file.write('\n\n########### stage : {:d}th    ##############'.format(stage+1))
        
        # updating parameters of gaussian-uniform mixture model with fixed network parameters，the updated pseudo labels and
        # posterior probability of correct labeling is listed in folder "./data/office(dataset name)/pseudo_list"
        samples, weighted_pseu_label, weights = make_weighted_pseudo_list(args, model)

        #updating network parameters with fixed gussian-uniform mixture model and pseudo labels
        acc, model = train(args, samples, weighted_pseu_label, weights)

        # copy model
        best_model = copy.deepcopy(model)
            
    torch.save(best_model, 'snapshot/save/final_best_model.pk')
    print('final accuracy:{:.4f}'.format(acc))

    list_metrics_classification = []
    list_metrics_classification.append(acc)

    list_metrics_clsf = []
    list_metrics_clsf.append(list_metrics_classification)
    list_metrics_clsf = np.array(list_metrics_clsf)

    if args.dataset in ["synthetic"]:
        save_file = "outputs/RSDA-" + args.target + "-metrics-classification.csv"
    elif args.dataset in ["seed", "seed-iv"]:
        save_file = "outputs/RSDA-" + args.dataset +"-session-" + str(args.session) + "-metrics-classification.csv"
    elif args.dataset in ["stroke"]:
        save_file = "outputs/RSDA-" + args.dataset +"-metrics-classification.csv"
    else:
        save_file = "outputs/RSDA-" + args.source + "-" + args.target + "-metrics-classification.csv"


    f = open(save_file, 'ab')
    np.savetxt(f, list_metrics_clsf, delimiter=",", fmt='%0.4f')
    f.close()

    return acc, best_model



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spherical Space Domain Adaptation with Pseudo-label Loss')
    parser.add_argument('--baseline', type=str, default='MSTN', choices=['MSTN', 'DANN'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--dataset',type=str,default='office')
    parser.add_argument('--source', type=str, default='amazon')
    parser.add_argument('--target',type=str,default='dslr')
    parser.add_argument('--source_list', type=str, default='data/office/amazon_list.txt', help="The source dataset path list")
    parser.add_argument('--target_list', type=str, default='data/office/dslr_list.txt', help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=50, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=1000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr_a', type=float, default=0.001, help="learning rate 1")
    parser.add_argument('--lr_b', type=float, default=0.001, help="learning rate 2")
    parser.add_argument('--radius', type=float, default=10, help="learning rate")
    parser.add_argument('--num_class',type=int,default=31,help='the number of classes')
    parser.add_argument('--stages', type=int, default=1, help='the number of alternative iteration stages')
    parser.add_argument('--max_iter1',type=int,default=2000)
    parser.add_argument('--max_iter2', type=int, default=1000)
    parser.add_argument('--batch_size',type=int,default=36)
    parser.add_argument('--seed', type=int, default=123, help="random seed number ")
    parser.add_argument('--bottleneck_dim', type=int, default=256, help="Bottleneck (features) dimensionality")
    parser.add_argument('--session', type=int, default=123, help="random seed number ")
    parser.add_argument('--file_path', type=str, default='/home/Descargas/', help="Path from the current dataset")
    parser.add_argument('--log_file')
    args = parser.parse_args()

    # Set random SEED
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # create directory snapshot
    if not os.path.exists('snapshot'):
        os.mkdir('snapshot')
    # create directory
    if not os.path.exists('snapshot/{}'.format(args.output_dir)):
        os.mkdir('snapshot/{}'.format(args.output_dir))
    # create file name for log.txt
    log_file = open('snapshot/{}/log.txt'.format(args.output_dir),'w')
    log_file.write('dataset:{}\tsource:{}\ttarget:{}\n\n'
                   ''.format(args.dataset,args.source,args.target))
    args.log_file = log_file

    # Assign file paths
    if args.dataset == "synthetic":
        args.file_path = "/home/magdiel/Descargas/Datasets/SYNTHETIC/"
    elif args.dataset == "seed":
        args.file_path = "/home/magdiel/Descargas/Datasets/SEED/"
    elif args.dataset == "seed-iv":
        args.file_path = "/home/magdiel/Descargas/Datasets/SEEDIV/"
    elif args.dataset == "stroke":
        args.file_path = "/home/magdiel/Descargas/Datasets/STROKE/"
    else:
        print("This dataset does not exist.")
        exit(-1)


    main(args)




