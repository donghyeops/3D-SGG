# -*- coding: utf-8 -*-

import os
import os.path as osp
import json
import shutil
import time
import random
import numpy as np
import numpy.random as npr
import argparse
import math
import pdb

import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from faster_rcnn import network
from faster_rcnn.RelNet import RelNet
from faster_rcnn.utils.timer import Timer
from faster_rcnn.fast_rcnn.config import cfg
from faster_rcnn.datasets.ai2thor_relation_dataset_loader import ai2thor_relation_dataset


parser = argparse.ArgumentParser()

# Training parameters
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--max_epoch', type=int, default=50)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--step_size', type=int, default=20, help='Step size for reduce learning rate')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--model_tag', type=str, default='#0')
parser.add_argument('--test', action='store_true', help='test model')
parser.add_argument('--output_dir', type=str, default='./output')

parser.add_argument('--normalization', action='store_true')
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--wo_class', action='store_true', help='do not use class as input feature')
parser.add_argument('--wo_img', action='store_true', help='do not use image as input feature')
args = parser.parse_args()


def main():
    ############### 하이퍼 파라미터 출력 ###############
    print('*' * 25)
    print('*')
    if args.test:
        print('* test_mode *')
    print('* lr :', args.lr)
    print('* step_size :', args.step_size)
    print('* optimizer :', args.optimizer)
    print('* normalization :', args.normalization)
    print('*')
    print('*' * 25)
    #############################################

    # To set the random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed + 1)
    torch.cuda.manual_seed(args.seed + 2)

    print("Loading training set and testing set...", end=' ')
    train_set = ai2thor_relation_dataset('normal', 'train')
    test_set = ai2thor_relation_dataset('normal', 'test')
    print("Done.")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8)


    net = RelNet(nhidden=args.mps_feature_len,
                predicate_loss_weight=train_set.class_weight_relationship,
                dropout=args.dropout,
                nlabel=train_set.num_object_classes,
                nrelation=train_set.num_predicate_classes)
    params = list(net.parameters())

    net.cuda()
    if args.optimizer == 'adam' or args.optimizer == 'Adam':
        optimizer_class = torch.optim.Adam
    if args.optimizer == 'sgd' or args.optimizer == 'SGD':
        optimizer_class = torch.optim.SGD

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.test:
        print('======= Testing Result =======')
        net.load_state_dict(torch.load('./output/RelNet_best.state'))
        test(test_loader, net, test_set.num_object_classes)
    else:
        top_Ns = [50, 100]
        best_acc = 0.
        optimizer = optimizer_class([
            {'params': params, 'lr': args.lr}
            ], lr=args.lr, weight_decay=0.0005)

        for epoch in range(0, args.max_epoch):
            # Training
            train(train_loader, net, optimizer, epoch)

            acc = test(test_loader, net, top_Ns, test_set.num_object_classes)

            if acc > best_acc:
                best_acc = acc

                save_name = os.path.join(args.output_dir, 'RelNet_best.pt')
                torch.save(net, save_name)
                print(('save model: {}'.format(save_name)))

            # updating learning policy
            # if epoch % args.step_size == 0 and epoch > 0:
            if epoch > 0 and epoch % args.step_size == 0:
                lr /= 10
                print('[learning rate: {}]'.format(lr))

                # update optimizer and correponding requires_grad state
                optimizer = optimizer_class([
                    {'params': params, 'lr': args.lr}
                    ], lr=lr, weight_decay=0.0005)
                # save_name = os.path.join(args.output_dir, 'RelNet.pt')
                # torch.save(net, save_name)
                # print(('save model: {}'.format(save_name)))
        ## model save
        #save_name = os.path.join(args.output_dir, 'RelNet.h5')
        #network.save_net(save_name, net)
        #print(('save model: {}'.format(save_name)))
        save_name = os.path.join(args.output_dir, 'RelNet.pt')
        torch.save(net, save_name)
        print(('save model: {}'.format(save_name)))


def train(train_loader, target_net, optimizer, epoch):
    batch_time = network.AverageMeter()
    data_time = network.AverageMeter()
    # Total loss
    train_loss = network.AverageMeter()
    # relationship cls loss
    train_rel_loss = network.AverageMeter()

    # object
    accuracy_rel = network.AccuracyMeter()

    target_net.train()
    end = time.time()
    # im_data.size() : (1, 3, H, W)
    # im_info.size() : (1, 3)
    # gt_objects.size() : (1, GT_obj#, 5[x1 ,y1 ,x2, y2, obj_class])
    # gt_relationships.size() : (1, GT_obj#, GT_obj#), int[pred_class]
    # gt_regions.size() : (1, GT_reg#, 15[?])
    for i, (boxes_3d, gt_relationships) in enumerate(train_loader):
        if len(boxes_3d.numpy()[0]) < 2: # 물체 2개 이상만 통과
            continue

        # measure the data loading time
        data_time.update(time.time() - end)

        target_net(boxes_3d[0], gt_relationships[0])

        loss = target_net.ce_relation
        train_rel_loss.update(target_net.ce_relation.data.cpu().numpy(), 1)
        accuracy_rel.update(target_net.tp_pred, target_net.tf_pred, target_net.fg_cnt_pred, target_net.bg_cnt_pred)

        if args.normalization:
            # L2 정규화
            l2_reg = torch.tensor(0.).cuda()
            for param in target_net.parameters():
                l2_reg += torch.norm(param)
            loss += l2_reg

        optimizer.zero_grad()  # 각 파라미터의 기울기 값을 0으로 만듦
        loss.backward()  # chain rule을 이용한 기울기 계산
        network.clip_gradient(target_net, 10.)
        optimizer.step()  # 가중치 업데이트

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Logging the training loss
        if (i + 1) % 2000 == 0:
            print('Epoch: [{0}][{1}/{2}] \n\tBatch_Time: {batch_time.avg: .3f}s'.format(
                epoch, i + 1, len(train_loader), batch_time=batch_time))
            print('\t[rel_Loss]\trel_cls_loss: %.4f,' % (train_rel_loss.avg))


def test(test_loader, net, top_Ns, num_object_classes, measure='sgg'):
    print('========== Testing =======')
    net.eval()
    # For efficiency inference

    rel_cm = np.zeros((9, 9))  # (pred, gt)

    def update_cm(pred, gt, cm):
        assert len(pred) == len(gt), f'len(pred):{len(pred)}, len(gt):{len(gt)}'
        for i in range(len(pred)):
            cm[pred[i]][gt[i]] += 1

    def get_metrics(cm):
        recalls = []
        precisions = []
        correct_cnt = 0
        for i in range(len(cm)):
            if np.sum(cm[:, i]) != 0:
                recalls.append(cm[i, i] / np.sum(cm[:, i]))
            if np.sum(cm[i, :]) != 0:
                precisions.append(cm[i, i] / np.sum(cm[i, :]))
            correct_cnt += cm[i, i]
        recall = np.average(recalls)
        precision = np.average(precisions)
        accuracy = correct_cnt / np.sum(cm)
        return recall, precision, accuracy

    for i, (boxes_3d, gt_relationships) in enumerate(test_loader):
        if len(boxes_3d.numpy()[0]) < 2:  # 물체 2개 이상만 통과
            continue

        # Forward pass
        rel_pred, rel_gt = net.evaluate(boxes_3d[0], gt_relationships[0])

        update_cm(rel_pred, rel_gt, rel_cm)

    print('====== Done Testing ====')
    # Restore the related states

    rel_metrics = get_metrics(rel_cm)
    print(f'relation: [Recall: {str(rel_metrics[0])[:7]}, Precision: {str(rel_metrics[1])[:5]}, Accuracy: {str(rel_metrics[2])[:5]}')
    # Restore the related states

    return rel_metrics[2]


if __name__ == '__main__':
    main()
