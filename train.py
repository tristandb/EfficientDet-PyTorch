import time
import os
import copy
import argparse
import pdb
import collections
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

import model
from anchors import Anchors
import losses
from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import coco_eval
import csv_eval

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):

    parser = argparse.ArgumentParser(description='Training script for training a EfficientDet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--phi', help='EfficientNet scaling coefficient.', type=int, default=0)
    parser.add_argument('--batch-size', help='Batch size', type=int, default=8)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)

    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017', transform=transforms.Compose([Normalizer(), Augmenter(), Resizer(img_size=512)]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer(img_size=512)]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO')


        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    efficientdet = model.efficientdet(num_classes=dataset_train.num_classes(), pretrained=True, phi=parser.phi)      

    use_gpu = torch.cuda.is_available()

    if use_gpu:
        efficientdet = efficientdet.cuda()
    
    efficientdet = torch.nn.DataParallel(efficientdet).cuda()

    efficientdet.training = True
    

    optimizer = optim.Adam(efficientdet.parameters(), lr=1e-5)

    
    # TODO: Add learning rate scheduler
    # Calculate 5% of warm-up training
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    efficientdet.train()
    efficientdet.module.freeze_bn()
    
    print(f"Number of parameters: {sum(p.numel() for p in efficientdet.parameters())}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in efficientdet.parameters() if p.requires_grad)}")

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(parser.epochs):

        efficientdet.train()
        efficientdet.module.freeze_bn()
        
        epoch_loss = []
        
        print(('\n' + '%10s' * 5) % ('Epoch', 'gpu_mem', 'Loss', 'cls', 'rls'))
        
        pbar = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
        for iter_num, data in pbar:
            try:
                optimizer.zero_grad()
                
                classification_loss, regression_loss = efficientdet([data['img'].cuda().float(), data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss
                
                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(efficientdet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))
                
                loss = (loss * iter_num) / (iter_num + 1)  # update mean losses
                mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)
                s = ('%10s' * 2 + '%10.3g' * 3) % (
                    '%g/%g' % (epoch_num, parser.epochs - 1), '%.3gG' % mem, np.mean(loss_hist), float(regression_loss), float(classification_loss))
                pbar.set_description(s)
                
                del classification_loss
                del regression_loss
            except Exception as e:
                raise(e)
                continue

        if parser.dataset == 'coco':

            print('Evaluating dataset')

            coco_eval.evaluate_coco(dataset_val, efficientdet)

        elif parser.dataset == 'csv' and parser.csv_val is not None:

            print('Evaluating dataset')

            mAP = csv_eval.evaluate(dataset_val, efficientdet)

        
        scheduler.step(np.mean(epoch_loss))    

        torch.save(efficientdet.module, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))

    efficientdet.eval()

    torch.save(efficientdet, 'model_final.pt'.format(epoch_num))

if __name__ == '__main__':
 main()
