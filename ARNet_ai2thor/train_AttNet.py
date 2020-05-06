# -*- coding: utf-8 -*-

import argparse
import os
import random
import shutil
import time

import numpy as np
import torch
from torch.autograd import Variable

from faster_rcnn import network
from faster_rcnn.AttNet import AttNet
from faster_rcnn.datasets.ai2thor_attribute_dataset_loader import ai2thor_attribute_dataset

## 피쳐 사이즈 출력
show_iter = False  # iteration 출력
show_DB_shape = False  # Network input data 출력

parser = argparse.ArgumentParser()

# Training parameters
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--max_epoch', type=int, default=50)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--step_size', type=int, default=10, help='Step size for reduce learning rate')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--model_tag', type=str, default='#0')
parser.add_argument('--test', action='store_true', help='test model')
parser.add_argument('--vis', action='store_true', help='visualization output')
parser.add_argument('--output_dir', type=str, default='./output')

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
    print('* epochs :', args.max_epoch)
    print('* step_size :', args.step_size)
    print('* class feature :', not args.wo_class)
    print('* image feature :', not args.wo_img)
    print('* optimizer :', args.optimizer)
    print('*')
    print('*' * 25)
    #############################################

    # To set the random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed + 1)
    torch.cuda.manual_seed(args.seed + 2)

    print("Loading training set and testing set...", end=' ')
    train_set = ai2thor_attribute_dataset('normal', 'train')
    test_set = ai2thor_attribute_dataset('normal', 'test')
    print("Done.")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8)

    net = AttNet(nlabel=train_set.num_object_classes,
                 ncolor=train_set.num_color_classes,
                 nopen_state=train_set.num_open_state_classes,
                 class_weight_color=train_set.class_weight_color,
                 class_weight_os=train_set.class_weight_os,
                 use_class=not args.wo_class,
                 use_img=not args.wo_img)
    print(net)

    params = list(net.parameters())
    net.cuda()

    if args.optimizer == 'adam' or args.optimizer == 'Adam':
        optimizer_class = torch.optim.Adam
    if args.optimizer == 'sgd' or args.optimizer == 'SGD':
        optimizer_class = torch.optim.SGD

    if args.vis:
        # 결과 시각화
        net.eval()
        net.load_state_dict(torch.load('./output/AttNet.state'))
        # for param in net.parameters():
        #     print(param.size(), param.data.sum())
        visualize(test_loader, net)
        return
   
    if not args.test:
        network.set_trainable(net, False)
        network.set_trainable_param(params, True)
        
        optimizer = optimizer([
            {'params': params, 'lr': args.lr}
        ], lr=args.lr, weight_decay=0.0005)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    best_acc = 0.
    best_color_acc = 0.
    best_open_state_acc = 0.

    if args.test:
        net.load_state_dict(torch.load('./output/AttNet.state'))
        test(test_loader, net, test_set.num_object_classes)
    else:
        for epoch in range(0, args.max_epoch):
            # Training
            train(train_loader, net, optimizer, epoch)

            # Testing
            color_acc, open_state_acc = test(test_loader, net, train_set.num_object_classes)

            acc = (color_acc + open_state_acc) / 2.
            if acc > best_acc:
                best_acc = acc
                save_name = os.path.join(args.output_dir, 'AttNet_best.state')
                torch.save(net.state_dict(), save_name)
                print(('save model: {}'.format(save_name)))

            if color_acc > best_color_acc:
                best_color_acc = color_acc
            if open_state_acc > best_open_state_acc:
                best_open_state_acc = open_state_acc

            # print('Epoch[{epoch:d}]:'.format(epoch=epoch))
            # print('\t[att.color R] {recall:2.3f}%% (best: {best_recall:2.3f}%%)'.format(
            #     recall=color_recall * 100, best_recall=best_color_recall * 100))
            # print('\t[att.open_state R] {recall:2.3f}%% (best: {best_recall:2.3f}%%)'.format(
            #     recall=open_state_recall * 100, best_recall=best_open_state_recall * 100))

            # updating learning policy
            if epoch % args.step_size == 0 and epoch > 0:
                args.lr /= 10
                print('[learning rate: {}]'.format(args.lr))

                # update optimizer and correponding requires_grad state
                optimizer = optimizer_class([
                    {'params': params, 'lr': args.lr}
                ], lr=args.lr, weight_decay=0.0005)

        ## save model
        save_name = os.path.join(args.output_dir, 'AttNet.state')
        torch.save(net.state_dict(), save_name)
        print(('save model: {}'.format(save_name)))


def visualize(train_loader, target_net):
    for i, (im_data, im_info, objects, gt_colors, gt_open_states) in enumerate(train_loader):
        if len(objects.numpy()[0]) < 1:  # 물체 1개 이상만 통과
            continue
        im_data = Variable(im_data.cuda())

        target_net.visualize(im_data, im_info, objects.numpy()[0], gt_colors[0], gt_open_states[0])


def train(train_loader, target_net, optimizer, epoch):
    batch_time = network.AverageMeter()
    data_time = network.AverageMeter()
    # Total loss
    # object related loss
    train_att_color_loss = network.AverageMeter()
    train_att_open_state_loss = network.AverageMeter()

    target_net.train()
    end = time.time()
    # im_data.size() : (1, 3, H, W)
    # im_info.size() : (1, 3)
    # objects.size() : (1, GT_obj#, 5[x1 ,y1 ,x2, y2, obj_class])
    for i, (im_data, im_info, objects, gt_colors, gt_open_states) in enumerate(train_loader):
        if len(objects.numpy()[0]) < 1:  # 물체 1개 이상만 통과
            continue

        # measure the data loading time
        data_time.update(time.time() - end)
        im_data = Variable(im_data.cuda())

        # batch_size가 1이므로, [0]은 그냥 한꺼풀 꺼내는걸 의미함
        # print('im_data.shape:', im_data.shape)
        # print('im_data:', im_data)
        # print('input_objects.shape:', objects.numpy()[0].shape)
        # print('input_objects:', objects.numpy()[0])
        target_net(im_data, im_info, objects.numpy()[0], gt_colors[0], gt_open_states[0])

        im_data = im_data.data  ## 추가 !!

        # Determine the loss function
        train_att_color_loss.update(target_net.ce_color.data.cpu().numpy(), im_data.size(0))
        train_att_open_state_loss.update(target_net.ce_open_state.data.cpu().numpy(), im_data.size(0))

        loss = target_net.att_loss

        optimizer.zero_grad()
        loss.backward()
        network.clip_gradient(target_net, 10.)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Logging the training loss
        if (i + 1) % 500 == 0:
            print(('Epoch: [{0}][{1}/{2}]  Batch_Time: {batch_time.avg: .3f}s'.format(
                epoch, i + 1, len(train_loader), batch_time=batch_time)))

            print('\t[att_Loss]\tcolor_loss: %.4f\n\t\t\topen_state_loss: %.4f' %
                  (train_att_color_loss.avg, train_att_open_state_loss.avg))


def test(test_loader, net, num_object_classes):
    global args

    print('========== Testing =======')

    net.eval()

    color_cm = np.zeros((10, 10))  # (pred, gt)
    os_cm = np.zeros((3, 3))  # (pred, gt)

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

    for i, (im_data, im_info, objects, gt_colors, gt_open_states) in enumerate(test_loader):
        if len(objects.numpy()[0]) < 1:  # 물체 1개 이상만 통과
            continue
        im_data = Variable(im_data.cuda(), volatile=True)  ## 추가 !!

        # Forward pass
        # 각 result = (total_cnt_t, cnt_correct_t)
        (color_pred, color_gt), (os_pred, os_gt) = net.evaluate(im_data, im_info, objects.numpy()[0], gt_colors[0],
                                                                gt_open_states[0])
        update_cm(color_pred, color_gt, color_cm)
        update_cm(os_pred, os_gt, os_cm)

    print('====== Done Testing ====')

    color_metrics = get_metrics(color_cm)
    os_metrics = get_metrics(os_cm)
    print(f'color: [Recall: {color_metrics[0]}, Precision: {color_metrics[1]}, Accuracy: {color_metrics[2]}')
    print(f'open_state: [Recall: {os_metrics[0]}, Precision: {os_metrics[1]}, Accuracy: {os_metrics[2]}')
    # Restore the related states

    return color_metrics[2], os_metrics[2]


if __name__ == '__main__':
    main()

