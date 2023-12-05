# -*- coding: utf-8 -*-
"""
@author: zifyloo
"""

import argparse
import torch
import logging
import os
from utils.read_write_data import makedir

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class options():
    def __init__(self):
        self._par = argparse.ArgumentParser(description='options for Deep Cross Modal')

        
        # self._par.add_argument('--model_name', type=str, help='experiment name', default="CUB_SSAN")
        # self._par.add_argument('--model_name', type=str, help='experiment name', default="SSAN_TIPCB3_bert_frozen_2space_4_6_3")
        # self._par.add_argument('--model_name', type=str, help='experiment name', default="SSAN_baseline")
        # self._par.add_argument('--model_name', type=str, help='experiment name', default="SSAN_TIPCB3_bert_frozen")
        

        
        self._par.add_argument('--rem_num', type=int, default=4, help='batch size')
        self._par.add_argument('--lr_dif', type=float, default=0.1, help='lr difference')
        self._par.add_argument('--mode', type=str, default='test', help='choose mode [train or test]')

        self._par.add_argument('--epoch', type=int, default=60, help='train epoch')
        self._par.add_argument('--epoch_decay', type=list, default=[20, 30, 50], help='decay epoch', nargs='*')
        self._par.add_argument('--epoch_begin', type=int, default=5, help='when calculate the auto margin')

        self._par.add_argument('--batch_size', type=int, default=2, help='batch size')
        self._par.add_argument('--adam_alpha', type=float, default=0.9, help='momentum term of adam')
        self._par.add_argument('--adam_beta', type=float, default=0.999, help='momentum term of adam')
        self._par.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
        self._par.add_argument('--margin', type=float, default=0.2, help='ranking loss margin')
        self._par.add_argument('--margin_intra', type=float, default=0.2, help='ranking loss margin')
        self._par.add_argument('--weight_intra', type=float, default=0.5, help='ranking loss margin')
        self._par.add_argument('--cr_beta', type=float, default=0.1, help='ranking loss margin')

        
        
        self._par.add_argument('--feature_length', type=int, default=1024, help='the length of feature')

        self._par.add_argument('--part', type=int, default=6, help='the num of image part')
        self._par.add_argument('--caption_length_max', type=int, default=100, help='the max length of caption')

        self._par.add_argument('--save_path', type=str, default='./checkpoints/test',
                               help='save the result during training')
        self._par.add_argument('--GPU_id', type=str, default='0', help='choose GPU ID [0 1]')
        self._par.add_argument('--device', type=str, default='', help='cuda devie')
        
        self._par.add_argument('--model_name', type=str, help='experiment name', default="debug")
        self._par.add_argument('--dataset', type=str, help='choose the dataset ', default='CUHK-PEDES')
        self._par.add_argument('--dataroot', type=str,  help='data root of the Data', default='./dataset/CUHK-PEDES/')
        self._par.add_argument('--vocab_size', type=int, default=5000, help='the size of vocab')
        self._par.add_argument('--class_num', type=int, default=11000,
                               help='num of class for StarGAN training on second dataset')
       
        # self._par.add_argument('--model_name', type=str, help='experiment name', default="CUB_SSAN_TIPCB3_bert_frozen_2space_4_6_3_224_new")
        # self._par.add_argument('--dataset', type=str, help='choose the dataset ', default='CUB_200_2011')
        # self._par.add_argument('--dataroot', type=str,  help='data root of the Data', default='/home/mayiwei/Project/ReID/SSAN/dataset/CUB_200_2011')
        # self._par.add_argument('--vocab_size', type=int, default=3300, help='the size of vocab')
        # self._par.add_argument('--class_num', type=int, default=3701,
        #                        help='num of class for StarGAN training on second dataset')


        # self._par.add_argument('--model_name', type=str, help='experiment name', default="Flower_SSAN_TIPCB3_bert_frozen_2space_4_6_3_224_new")
        # self._par.add_argument('--dataset', type=str, help='choose the dataset ', default='Flower')
        # self._par.add_argument('--dataroot', type=str,  help='data root of the Data', default='/home/mayiwei/Project/ReID/SSAN/dataset/Flower')
        # self._par.add_argument('--vocab_size', type=int, default=3000, help='the size of vocab')
        # self._par.add_argument('--class_num', type=int, default=103,
        #                        help='num of class for StarGAN training on second dataset')
        

        self.opt = self._par.parse_args()

        self.opt.device = torch.device('cuda:{}'.format(self.opt.GPU_id[0]))


def config(opt):

    log_config(opt)
    model_root = os.path.join(opt.save_path, 'model')
    if os.path.exists(model_root) is False:
        makedir(model_root)


def log_config(opt):
    logroot = os.path.join(opt.save_path, 'log')
    if os.path.exists(logroot) is False:
        makedir(logroot)
    filename = os.path.join(logroot, opt.mode + '.log')
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(handler)
    if opt.mode != 'test':
        logger.info(opt)



