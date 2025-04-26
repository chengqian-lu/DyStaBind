import argparse
import os
import subprocess

import numpy as np
import pandas as pd
import torch
from torch import nn

from torch.utils.data.dataloader import DataLoader
from transformers import BertTokenizer, BertModel

from utils.dealWithData import read_fasta_files, split_dataset, myDataset
from utils.gen_bert_embedding import circRNABert
from utils.trainAndValidate import train, validate
from utils.utils import GradualWarmupScheduler, param_num, read_csv, save_result, save_roc_curves, save_middel_result, \
    getLabel, log_print, fix_seed
from Model.DyStaBind import DyStaBind



def main(args):
    fix_seed(args.seed)  # fix seed

    z_curve = {'name': [], 'feature': [], 'label': []}
    structure = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    file_name = args.data_file

    item = ""
    if args.structure:
        item = item + "_structure"
    if args.z_curve:
        item = item + "_z_curve"
    if args.icSHAPE:
        item = item + "_icSHAPE"
    if args.BERTEmbedding:
        item = item + "_BERTEmbedding"
    item = item + "_drop" + str(args.data_dropout)

    # ***********************train model***************************
    if args.train:
        # ****************load data***********************

        # loss
        criterion = NULL
        # optimizer
        optimizer = NULL
        scheduler = NULL
        # *******************train********************
        best_auc = 0
        best_acc = 0
        best_epoch = 0
        best_roc = None
        train_auc_list = []
        train_acc_list = []
        train_epoch_list = []
        test_auc_list = []
        test_acc_list = []
        test_epoch_list = []
        a = best_auc * 0.7 + best_acc * 0.3  # best_auc和best_acc的权重和

        model_save_path = args.model_save_path + "/temp"

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        early_stopping = args.early_stopping

        param_num(model)

        for epoch in range(1, 200):
            t_met = train(model, device, train_loader, criterion, optimizer, batch_size=32)
            v_met, _, _ = validate(model, device, test_loader, criterion)
            scheduler.step()
            lr = scheduler.get_lr()[0]
            color_best = 'green'
            train_auc_list.append(t_met.auc)
            train_acc_list.append(t_met.acc)
            train_epoch_list.append(epoch)
            test_auc_list.append(v_met.auc)
            test_acc_list.append(v_met.acc)
            test_epoch_list.append(epoch)
            b = v_met.auc * 0.7 + v_met.acc * 0.3  # 当前auc和acc的权重和
            # if best_auc < v_met.auc:
            if a < b:
                a = b
                best_roc = v_met.roc_curves[0][0]
                best_auc = v_met.auc
                best_acc = v_met.acc
                best_epoch = epoch
                color_best = 'red'
                path_name = os.path.join(model_save_path, file_name + item +'.pth')
                torch.save(model.state_dict(), path_name)
            if epoch - best_epoch > early_stopping:
                print("Early stop at %d, %s " % (epoch, 'MUNet_M'))
                break
            line = '{} \t Train Epoch: {}     avg.loss: {:.4f} Acc: {:.4f}, AUC: {:.4f} lr: {:.6f}'.format(
                file_name, epoch, t_met.other[0], t_met.acc, t_met.auc, lr)
            log_print(line, color='green', attrs=['bold'])

            line = '{} \t Test  Epoch: {}     avg.loss: {:.4f} Acc: {:.4f}, AUC: {:.4f} ({:.4f}) {}'.format(
                file_name, epoch, v_met.other[0], v_met.acc, v_met.auc, best_auc, best_epoch)
            log_print(line, color=color_best, attrs=['bold'])

        print("{} auc: {:.4f} acc: {:.4f}".format(file_name, best_auc, best_acc))

        # save results



    # *******************test model********************
    if args.validate:
        # comming soon
        pass


    # ********************************dynamic test*********************************
    if args.dynamic_validate:   # perform dynamic prediction between K562 cell and HepG2 cell
        # comming soon
        pass




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Welcome to DyStaBind!')
    parser.add_argument('--data_file', default='TIA1_Hela', type=str, help='RBP to train or validate')
    parser.add_argument('--data_path', default='./dataset', type=str, help='The data path')
    parser.add_argument('--model_save_path', default='./results/model', type=str, help='Save the trained model for dynamic prediction')
    parser.add_argument('--output_dir', default='./results/output', type=str,
                        help='Save output directory')
    parser.add_argument('--BERT_model_path', default='./BERT_Model', type=str,
                        help='BERT model path, in case you have another BERT')
    parser.add_argument('--middel_results_save_path', default='./results/middel_results', type=str,
                        help='Middel results(AUC, ACC, Epoch) for each epoch')

    parser.add_argument('--structure', default=False, action='store_true')
    parser.add_argument('--z_curve', default=False, action='store_true')
    parser.add_argument('--icSHAPE', default=False, action='store_true')
    parser.add_argument('--BERTEmbedding', default=False, action='store_true')
    parser.add_argument('--data_dropout', default=0.3, type=float)

    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--validate', default=False, action='store_true')
    parser.add_argument('--dynamic_validate', default=False, action='store_true')

    parser.add_argument('--seed', default=1024, type=int, help='The random seed')
    parser.add_argument('--early_stopping', type=int, default=20)

    args = parser.parse_args()
    main(args)