# -*- coding: utf-8 -*-

import os
import sys
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

from faster_rcnn import network
from faster_rcnn.TransferNet import TransferNet
from faster_rcnn.utils.timer import Timer
from faster_rcnn.datasets.ai2thor_transfer_dataset_loader import ai2thor_transfer_dataset

sys.path.append('/home/ailab/DH/ai2thor')
temp_path = os.getcwd()
os.chdir('/home/ailab/DH/ai2thor')
from thor_utils import annotation_util as au
os.chdir(temp_path)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os.path as osp
import json


parser = argparse.ArgumentParser()

# Training parameters
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--max_epoch', type=int, default=50)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--step_size', type=int, default=10, help='Step size for reduce learning rate')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--threshold', type=float, default=0.3, help='GT decision threshold for test')
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--model_tag', type=str, default='TransferNet')
parser.add_argument('--test', action='store_true', help='test model')
parser.add_argument('--vis', action='store_true', help='visualization output')
parser.add_argument('--output_dir', type=str, default='./output')

parser.add_argument('--normalization', action='store_true')
parser.add_argument('--wo_bbox', action='store_true', help='do not use bbox as input feature')
parser.add_argument('--wo_class', action='store_true', help='do not use class as input feature')
parser.add_argument('--wo_img', action='store_true', help='do not use image as input feature')
parser.add_argument('--wo_default_box', action='store_true', help='directly predict 3d bbox')
args = parser.parse_args()


def main():
    ############### 하이퍼 파라미터 출력 ###############
    print('## train_TransferNet for AI2THOR ##')
    print('*' * 25)
    print('*')
    #print('* step_size :', args.step_size)
    if args.test:
        print('* test_mode *')
    print('* epoch :', args.max_epoch)
    print('* batch_size :', args.batch_size)
    print('* lr :', args.lr)
    print('* step_size :', args.step_size)
    print('* optimizer :', args.optimizer)
    print('* normalization :', args.normalization)
    print('* use_default_box:', not args.wo_default_box)
    print('* use_bbox:', not args.wo_bbox)
    print('* use_img:', not args.wo_img)
    print('* use_class:', not args.wo_class)
    print('* Acc@iou :', args.threshold)
    print('*')
    print('*' * 25)
    #############################################

    #print('Model name: {}'.format(args.model_name))

    # To set the random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed + 1)
    torch.cuda.manual_seed(args.seed + 2)

    print("Loading training set and testing set...", end=' ')
    train_set = ai2thor_transfer_dataset('normal', 'train', not args.wo_default_box)
    test_set = ai2thor_transfer_dataset('normal', 'test', not args.wo_default_box)
    print("Done.")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=1)

    # Model declaration
    net = TransferNet(
                    nlabel=train_set.num_object_classes,
                    use_bbox=not args.wo_bbox,
                    use_img=not args.wo_img,
                    use_class=not args.wo_class
    )
    net.cuda()
    print(net)
    if args.optimizer == 'adam' or args.optimizer == 'Adam':
        optimizer_class = torch.optim.Adam
    if args.optimizer == 'sgd' or args.optimizer == 'SGD':
        optimizer_class = torch.optim.SGD

    params = list(net.parameters())
    
    logger_path = "log/logger/{}".format(args.model_name)
    if os.path.exists(logger_path):
        shutil.rmtree(logger_path)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.test_mode:
        model_file = './output/TransferNet_best.state'
        net.load_state_dict(torch.load(model_file))
        net.eval()
        print('======= Testing Result =======')
        test(test_loader, net, threshold=args.threshold, debug=True)
    elif args.visualize:
        if args.visualize_dataset == 'test':
            visualize(test_set, test_loader, net)
        else:
            visualize(train_set, train_loader, net)
    else: # train
        max_miou = 0.
        best_loss = np.zeros(6) # (x, y, z, w, h, d)

        optimizer = optimizer_class([
            {'params': params, 'lr': args.lr}
            ], lr=lr, weight_decay=0.0005)

        for epoch in range(0, args.max_epoch):
            # Training
            print('Training Epoch :', epoch)
            train(train_loader, net, optimizer, epoch)

            result_losses, miou, _ = test(test_loader, net, threshold=args.threshold)
            for i in range(len(result_losses)):
                if best_loss[i] < result_losses[i]:
                    best_loss[i] = result_losses[i]

            # updating learning policy
            # if epoch % args.step_size == 0 and epoch > 0:
            if epoch > 0 and epoch % args.step_size == 0:
                lr *= 0.1
                print('[learning rate: {}]'.format(lr))

                # update optimizer and correponding requires_grad state
                optimizer = optimizer_class([
                    {'params': params, 'lr': lr}
                    ], lr=lr, weight_decay=0.0005)
            if miou > max_miou:
                max_miou = miou
                save_name = os.path.join(args.output_dir, f'TransferNet_best.state')
                torch.save(net.state_dict(), save_name)
                print(('save model: {}'.format(save_name)))

        #save_name = os.path.join(args.output_dir, 'TransferNet.h5')
        #network.save_net(save_name, net)
        #print(('\nsave model: {}'.format(save_name)))
        save_name = os.path.join(args.output_dir, f'TransferNet_e{epoch}_{opt}.state')
        torch.save(net.state_dict(), save_name)
        print(('save model: {}'.format(save_name)))


def train(train_loader, target_net, optimizer, epoch):
    batch_time = network.AverageMeter()

    target_net.train()
    end = time.time()
    # input data
    # 1. bbox (x, y, w, h, distance, label) # OD 결과값 (학습 시에는 OD의 정답 bbox가 들어감)
    # 2. agent (x, y, z, rotation) #rotation은 0~3 값
    # output data (answer)
    # 1. 3D bbox (x, y, z, w, h, d)
    count=0
    res_total_loss = 0.
    res_losses = np.zeros(6)
    for i, (im_data, depth, boxes, agent, targets, default_box, gt_boxes_3d) in enumerate(train_loader):
        if len(boxes.numpy()[0]) < 1: # 물체 1개 이상만 통과
            continue

        im_data = Variable(im_data.cuda())
        depth = Variable(depth.cuda())

        output, total_loss, losses = target_net(im_data, depth, boxes.numpy(), agent, targets)
        im_data = im_data.data
        if args.normalization:
            # L2 정규화
            l2_reg = torch.tensor(0.).cuda()
            for param in target_net.parameters():
                l2_reg += torch.norm(param)
            total_loss += l2_reg

        optimizer.zero_grad()
        total_loss.backward()
        network.clip_gradient(target_net, 10.)
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        res_total_loss += total_loss.data.cpu().numpy()
        for j in range(len(losses)):
            res_losses[j] += losses[j].data.cpu().numpy()
        count += 1
        ###
        # output = output.data.cpu().numpy()
        # targets = targets.data.cpu().numpy()
        # default_box = default_box.data.cpu().numpy()
        # agent = agent.data.cpu().numpy()
        # gt_boxes_3d = gt_boxes_3d.data.cpu().numpy()
        # 
        # output = targets + default_box
        # 
        # # print(output.shape)
        # results = np.zeros((len(output), 6))
        # for j in range(len(results)):
        #     rotate = str(agent[j][3])[:4]
        #     if rotate == '0.00' or rotate == '0.0':
        #         results[j, 0] = output[j, 0] + agent[j][0]
        #         results[j, 2] = output[j, 2] + agent[j][2]
        #     elif rotate == '0.33':
        #         results[j, 0] = output[j, 2] + agent[j][0]
        #         results[j, 2] = -output[j, 0] + agent[j][2]
        #     elif rotate == '0.66':
        #         results[j, 0] = -output[j, 0] + agent[j][0]
        #         results[j, 2] = -output[j, 2] + agent[j][2]
        #     elif rotate == '0.99':
        #         results[j, 0] = -output[j, 2] + agent[j][0]
        #         results[j, 2] = output[j, 0] + agent[j][2]
        #     else:
        #         assert False, f'error {rotate}'
        #     results[j, 1] = output[j, 1]  # + agent[j][1]
        #     results[j, 3:] = output[j, 3:]
        # 
        # miou = 0.
        # acc = 0.
        # for j in range(len(results)):
        #     iou = get_iou3D(results[j], gt_boxes_3d[j])
        #     # print(results[i])
        #     # print(gt_boxes_3d[i])
        #     # print(targets[i])
        #     # print(iou)
        #     # assert False, 'end'
        #     miou += iou
        #     if iou >= 0.5:
        #         acc += 1
        # miou /= len(results)
        # acc /= len(results)
        # print('\t miou: %.4f' % miou)
        # print('\t acc: %.4f' % acc)
        ###
        # Logging the training loss
        if (i + 1) % int(len(train_loader)/5) == 0:
            print('Epoch: [{0}][{1}/{2}]\n'
                   '\tBatch_Time: {batch_time.avg: .3f}s'.format(
                epoch, i + 1, len(train_loader), batch_time=batch_time))
            print('\t total_loss: %.4f' % (res_total_loss/count))

            ###
            output = output.data.cpu().numpy()
            targets = targets.data.cpu().numpy()
            default_box = default_box.data.cpu().numpy()
            agent = agent.data.cpu().numpy()
            gt_boxes_3d = gt_boxes_3d.data.cpu().numpy()
            if args.wo_default_box:
                output = output
            else:
                output = output + default_box

            # print(output.shape)
            results = np.zeros((len(output), 6))
            for j in range(len(results)):
                rotate = str(agent[j][3])[:4]
                if rotate == '0.00' or rotate == '0.0':
                    results[j, 0] = output[j, 0] + agent[j][0]
                    results[j, 2] = output[j, 2] + agent[j][2]
                elif rotate == '0.33':
                    results[j, 0] = output[j, 2] + agent[j][0]
                    results[j, 2] = -output[j, 0] + agent[j][2]
                elif rotate == '0.66':
                    results[j, 0] = -output[j, 0] + agent[j][0]
                    results[j, 2] = -output[j, 2] + agent[j][2]
                elif rotate == '0.99':
                    results[j, 0] = -output[j, 2] + agent[j][0]
                    results[j, 2] = output[j, 0] + agent[j][2]
                else:
                    assert False, f'error {rotate}'
                results[j, 1] = output[j, 1]*2# + agent[j][1]
                results[j, 3:] = output[j, 3:]

            miou = 0.
            acc = 0.
            for j in range(len(results)):
                iou = get_iou3D(results[j], gt_boxes_3d[j])
                # print(results[i])
                # print(gt_boxes_3d[i])
                # print(targets[i])
                # print(iou)
                # assert False, 'end'
                miou += iou
                if iou >= 0.5:
                    acc += 1
            miou /= len(results)
            acc /= len(results)
            print('\t miou: %.4f' % miou)
            print('\t acc: %.4f' % acc)
            ###

            print('\t x_loss: %.4f' % (res_losses[0] / count))
            print('\t y_loss: %.4f' % (res_losses[1] / count))
            print('\t z_loss: %.4f' % (res_losses[2] / count))
            print('\t w_loss: %.4f' % (res_losses[3] / count))
            print('\t h_loss: %.4f' % (res_losses[4] / count))
            print('\t d_loss: %.4f' % (res_losses[5] / count))
    return miou, acc

def test(test_loader, net, threshold=0.5, debug=False):

    print('========== Testing =======')
    net.eval()
    # For efficiency inference

    result_losses = np.zeros(6)
    l1_losses = np.zeros(6)
    miou = 0.
    correct_count = 0.
    iter_count=0
    obj_count = 0
    for i, (im_data, depth, boxes, agent, targets, default_box, gt_boxes_3d) in enumerate(test_loader):
        if len(boxes.numpy()[0]) < 1: # 물체 1개 이상만 통과
            continue

        im_data = Variable(im_data.cuda())
        depth = Variable(depth.cuda())

        output, total_loss, losses = net(im_data, depth, boxes.numpy(), agent, targets)

        output = output.data.cpu().numpy()
        targets = targets.data.cpu().numpy()
        default_box = default_box.data.cpu().numpy()
        agent = agent.data.cpu().numpy()
        gt_boxes_3d = gt_boxes_3d.data.cpu().numpy()
        if args.wo_default_box:
            output2 = output
        else:
            output2 = output + default_box
        # print(output.shape)

        results = np.zeros((len(output2), 6))
        for j in range(len(results)):
            rotate = str(agent[j][3])[:4]
            if rotate == '0.00' or rotate == '0.0':
                results[j, 0] = output2[j, 0] + agent[j][0]
                results[j, 2] = output2[j, 2] + agent[j][2]
            elif rotate == '0.33':
                results[j, 0] = output2[j, 2] + agent[j][0]
                results[j, 2] = -output2[j, 0] + agent[j][2]
            elif rotate == '0.66':
                results[j, 0] = -output2[j, 0] + agent[j][0]
                results[j, 2] = -output2[j, 2] + agent[j][2]
            elif rotate == '0.99':
                results[j, 0] = -output2[j, 2] + agent[j][0]
                results[j, 2] = output2[j, 0] + agent[j][2]
            else:
                assert False, f'error {rotate}'
            results[j, 1] = output2[j, 1]*2# + agent[j][1]
            results[j, 3:] = output2[j, 3:]

        boxes = boxes.data.cpu().numpy()
        for j in range(len(results)):
            iou = get_iou3D(results[j], gt_boxes_3d[j])
            # print(l1_losses[j, :].shape)
            # print(gt_boxes_3d[j].shape)
            # print(results[j].shape)
            l1_losses[:] += np.abs(gt_boxes_3d[j]-results[j])
            miou += iou
            obj_count+=1
            if iou >= threshold:
                correct_count += 1
            elif debug:
                print('output:', output[j])
                print('target:', targets[j])
                print('dif, iou:', targets[j]-output[j], iou)
                area = gt_boxes_3d[j][3] * gt_boxes_3d[j][4] * gt_boxes_3d[j][5]
                print('area, gt:', area, gt_boxes_3d[j])
                area = results[j][3] * results[j][4] * results[j][5]
                print('area, result:', area, results[j])
                area = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
                print('2d area:', area)
                print('class:', au.obj_i2s[boxes[j][-1]])
                print('')
                print('')


        for j, loss in enumerate(losses):
            result_losses[j] += loss.data.cpu().numpy()

        iter_count += 1

    result_losses /= iter_count
    miou /= obj_count
    acc = correct_count / obj_count
    print('====== Done Testing ====')
    print('\t total_loss: %.4f' % result_losses.sum())
    print('\t miou: %.4f' % miou)
    print('\t acc: %.4f [%d/%d]' % (acc, int(correct_count), obj_count))
    # print('\t x_loss: %.4f' % result_losses[0])
    # print('\t y_loss: %.4f' % result_losses[1])
    # print('\t z_loss: %.4f' % result_losses[2])
    # print('\t w_loss: %.4f' % result_losses[3])
    # print('\t h_loss: %.4f' % result_losses[4])
    # print('\t d_loss: %.4f' % result_losses[5])

    # print('\t xz_loss: %.4f' % (result_losses[0]))
    # print('\t y_loss: %.4f' % (result_losses[1]))
    # print('\t wd_loss: %.4f' % (result_losses[2]))

    print('\t pos_loss: %.4f, size_loss: %.4f' % (result_losses[:3].sum(), result_losses[3:].sum()))
    print('\t x_loss: %.4f [%.4f]' % (result_losses[0], l1_losses[0]/obj_count))
    print('\t y_loss: %.4f [%.4f]' % (result_losses[1], l1_losses[1]/obj_count))
    print('\t z_loss: %.4f [%.4f]' % (result_losses[2], l1_losses[2]/obj_count))
    print('\t w_loss: %.4f [%.4f]' % (result_losses[3], l1_losses[3]/obj_count))
    print('\t h_loss: %.4f [%.4f]' % (result_losses[4], l1_losses[4]/obj_count))
    print('\t d_loss: %.4f [%.4f]' % (result_losses[5], l1_losses[5]/obj_count))

    return result_losses, miou, acc

def visualize(vg, data_loader, net, show_region=False, target_index_list=None, start_idx=0, threshold=0.0):
    global args
    # 미구현
    print('======== Visualizing =======')
    #net.eval()

def get_iou3D(pos1, pos2):
    # pos: [x, y, z, w_x, h, w_z]
    pos1 = np.array([float(e) for e in pos1])
    pos2 = np.array([float(e) for e in pos2])

    pos1_min, pos1_max = _get_min_max_3dPos(pos1)
    pos2_min, pos2_max = _get_min_max_3dPos(pos2)
    inner_box_size = np.zeros(3)  # x_size, y_size, z_size
    for i in range(3):
        if pos1[i] > pos2[i]:
            inner_box_size[i] = pos2_max[i] - pos1_min[i]
        else:
            inner_box_size[i] = pos1_max[i] - pos2_min[i]
    for v in inner_box_size:
        if v <= 0:
            return 0.
    union = np.prod(pos1_max - pos1_min) + np.prod(pos2_max - pos2_min) - np.prod(inner_box_size)
    intersection = np.prod(inner_box_size)
    return intersection / union

def _get_min_max_3dPos(pos):
    half = pos[3:]/2

    return pos[:3]-half, pos[:3]+half

if __name__ == '__main__':
    main()
