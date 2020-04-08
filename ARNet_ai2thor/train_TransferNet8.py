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
# To log the training process
from tensorboard_logger import configure, log_value

from faster_rcnn import network
from faster_rcnn.TransferNet9 import TransferNet # transferNet9
from faster_rcnn.utils.timer import Timer
from faster_rcnn.fast_rcnn.config import cfg
from faster_rcnn.datasets.ai2thor_transfer_dataset_loader_8 import ai2thor_transfer_dataset # transferNet7
from faster_rcnn.utils.HDN_utils import get_model_name2, group_params

sys.path.append('/home/ailab/DH/ai2thor')
temp_path = os.getcwd()
os.chdir('/home/ailab/DH/ai2thor')
from thor_utils import annotation_util as au
os.chdir(temp_path)

# reload(sys)
# sys.setdefaultencoding('utf8') # 안하면 plt import 오류남
import matplotlib.pyplot as plt  ## 시각화 함수를 위해 추가
import matplotlib.patches as patches
from PIL import Image
import os.path as osp
import json

## 피쳐 사이즈 출력
show_iter = False  # iteration 출력
show_DB_shape = False  # Network input data 출력

TIME_IT = cfg.TIME_IT
parser = argparse.ArgumentParser('Options for training Hierarchical Descriptive Model in pytorch')

# Training parameters
parser.add_argument('--max_epoch', type=int, default=50, metavar='N', help='max iterations for training')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='percentage of past parameters to store')
parser.add_argument('--step_size', type=int, default=2, help='Step size for reduce learning rate')
parser.add_argument('--enable_clip_gradient', action='store_true', help='Whether to clip the gradient')
parser.add_argument('--use_normal_anchors', action='store_true', help='Whether to use kmeans anchors')

# structure settings
parser.add_argument('--no_dropout', action='store_true', help='To enables the dropout')
parser.add_argument('--nembedding', type=int, default=20, help='The size of word embedding')
# Environment Settings
parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--saved_model_path', type=str, default='model/pretrained_models/VGG_imagenet.npy',
                    help='The Model used for initialize')
parser.add_argument6('--dataset_option', type=str, default='small', help='The datasets to use (small | normal | fat)')
parser.add_argument('--output_dir', type=str, default='./output', help='Location to output the model')
parser.add_argument('--model_name', type=str, default='MSDN', help='The name for saving model.')
parser.add_argument('--optimizer', type=int, default=1,
                    help='which optimizer used for optimize language model [0: SGD | 1: Adam | 2: Adagrad]')

###############################
# train_hdn.py
parser.add_argument('--model_tag', type=str, default='TransferNet', help='모델명 정의')
parser.add_argument('--use_pin_memory', dest='use_pin_memory', action='store_true')
parser.add_argument('--no-use_pin_memory', dest='use_pin_memory', action='store_false')
parser.set_defaults(use_pin_memory=True)

parser.add_argument('--evaluate', action='store_true', help='Only use the testing mode')
parser.add_argument('--visualize', action='store_true', help='Only use the testing mode')
parser.add_argument('--visualize_index', type=int, default=-1, help='input datasets index')
parser.add_argument('--visualize_dataset', type=str, default='test', help='choose (test | train)')

# 기존 코드에서 아래가 없으면 오류남 ㅜㅜ
# MSDN.py, MSDN_base.py, Language_Model.py
parser.add_argument('--refiner_name', type=str, default='None', help='choose (None | mpu | 1cm_r | 1cm_c | 2cm)')

# MSDN.py, MSDN_base.py
parser.add_argument('--spatial_name', type=str, default='bbox', help='choose (None | bbox | mask)')

# MSDN_base.py
parser.add_argument('--cls_loss_name', type=str, default='cross_entropy', help='choose (cross_entropy | focal_loss)')

###############################


args = parser.parse_args()
# Overall loss logger
overall_train_loss = network.AverageMeter()
overall_train_rpn_loss = network.AverageMeter()

optimizer_select = 0

normalization = False

use_default_box = True
use_bbox = True
use_vis = True
use_class = True

def main():
    global args, optimizer_select

    lr = 0.001
    step_size = 10
    batch_size = 64
    threshold = 0.3
    test_mode = False


    opt = 'adam' # adam / sgd
    if opt == 'adam' or opt =='Adam':
        optimizer_class = torch.optim.Adam
    if opt == 'sgd' or opt =='SGD':
        optimizer_class = torch.optim.SGD

    ############### 하이퍼 파라미터 출력 ###############
    print('## train_TransferNet for AI2THOR ##')
    print('*' * 25)
    print('*')
    #print('* step_size :', args.step_size)
    print('* test mode :', test_mode)
    print('* epoch :', args.max_epoch)
    print('* lr :', lr)
    print('* step_size :', step_size)
    print('* optimizer :', opt)
    print('* normalization :', normalization)
    print('* use_default_box:', use_default_box)
    print('* use_bbox:', use_bbox)
    print('* use_vis:', use_vis)
    print('* use_class:', use_class)
    print('* batch_size :', batch_size)
    print('* Acc@iou :', threshold)
    print('*')
    print('*' * 25)
    #############################################

    #print('Model name: {}'.format(args.model_name))

    # To set the random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed + 1)
    torch.cuda.manual_seed(args.seed + 2)

    print("Loading training set and testing set...", end=' ')
    train_set = ai2thor_transfer_dataset('normal', 'train', use_default_box)
    test_set = ai2thor_transfer_dataset('normal', 'test', use_default_box)
    print("Done.")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1,
                                               pin_memory=args.use_pin_memory)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=1,
                                              pin_memory=args.use_pin_memory)

    train_set.inverse_weight_object = None  # 이 부분은 아직 미구현 (in dataset_loader)
    train_set.inverse_weight_predicate = None  # 이 부분은 아직 미구현 (in dataset_loader)
    # Model declaration
    net = TransferNet(
                    nlabel=train_set.num_object_classes,
                    use_bbox=use_bbox,
                    use_vis=use_vis,
                    use_class=use_class
    )
    print(net)

    params = list(net.parameters())
    # for param in params:
    #    print param.size()
    # print net

    # Setting the state of the training model
    net.cuda()  # 모델 파라미터들을 GPU로 옮김
    net.train()  # 네트워크를 트레인 모드로 바꿈 (Dropout이랑 BatchNorm에만 영향 줌)
    logger_path = "log/logger/{}".format(args.model_name)
    if os.path.exists(logger_path):
        shutil.rmtree(logger_path)
    configure(logger_path, flush_secs=5)  # setting up the logger

    network.set_trainable(net, False)

    args.train_all = True
    args.optimizer = 1  # 0: SGD, 1: Adam, 2: Adagrad

    network.set_trainable_param(params, True)
    # SGD, Adam, Adagrad
    optimizer = optimizer_class([
        {'params': params, 'lr': lr}
        ], lr=lr, weight_decay=0.0005)

    target_net = net
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    best_loss = np.zeros(6) # (x, y, z, w, h, d)


    if test_mode:
        #model_file = './output/TransferNet_e49_adam.state'
        model_file = './output/TransferNet9_full_best.state'
        net.load_state_dict(torch.load(model_file))
        net.eval()
        print('======= Testing Result =======')
        test(test_loader, net, threshold=threshold, debug=True)

    elif args.visualize:
        if args.visualize_dataset == 'test':
            visualize(test_set, test_loader, net)
        else:
            visualize(train_set, train_loader, net)
    else: # train
        max_miou = 0.

        for epoch in range(0, args.max_epoch):
            # Training
            print('Training Epoch :', epoch)
            train(train_loader, target_net, optimizer, epoch)
            # snapshot the state
            '''
            ## model save !!
            save_name = os.path.join(args.output_dir, '{}_epoch_{}.h5'.format(args.model_name, epoch))
            network.save_net(save_name, net)
            print(('save model: {}'.format(save_name)))
            '''

            # Testing
            # network.set_trainable(net, False) # Without backward(), requires_grad takes no effect

            result_losses, miou, acc = test(test_loader, net, threshold=threshold)
            for i in range(len(result_losses)):
                if best_loss[i] < result_losses[i]:
                    best_loss[i] = result_losses[i]

            # updating learning policy
            # if epoch % args.step_size == 0 and epoch > 0:
            if epoch > 0 and epoch % step_size == 0:
                lr *= 0.1
                print('[learning rate: {}]'.format(lr))

                # update optimizer and correponding requires_grad state
                optimizer = optimizer_class([
                    {'params': params, 'lr': lr}
                    ], lr=lr, weight_decay=0.0005)
            if miou > max_miou:
                max_miou = miou
                save_name = os.path.join(args.output_dir, f'TransferNet9_best.state')
                torch.save(net.state_dict(), save_name)
                print(('save model: {}'.format(save_name)))

        #save_name = os.path.join(args.output_dir, 'TransferNet.h5')
        #network.save_net(save_name, net)
        #print(('\nsave model: {}'.format(save_name)))
        save_name = os.path.join(args.output_dir, f'TransferNet_e{epoch}_{opt}.state')
        torch.save(net.state_dict(), save_name)
        print(('save model: {}'.format(save_name)))


def train(train_loader, target_net, optimizer, epoch):
    global args
    # Overall loss logger
    global overall_train_loss
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
        # show_iter = False ## iteration(i) 출력
        if show_iter:  ##
            print(('\n' + '*' * 8 + ' [iter:{}] '.format(i) + '*' * 8))
        # show_DB_shape = False ## DB shape 출력
        if show_DB_shape:  ##
            print('-- show_DB_shape --')
            print('boxes :', boxes.size())
            print('agent :', agent.size())
            print('targets :', targets.size())
            print('boxes[0] :', boxes[0])
            print('agent :', agent)
            print('targets[0] :', targets[0])
            print('-------------------')

        im_data = Variable(im_data.cuda())  ## 추가 !!
        depth = Variable(depth.cuda())

        # batch_size가 1이므로, [0]은 그냥 한꺼풀 꺼내는걸 의미함
        output, total_loss, losses = target_net(im_data, depth, boxes.numpy(), agent, targets)
        im_data = im_data.data  ## 추가 !!
        if normalization:
            # L2 정규화
            l2_reg = torch.tensor(0.).cuda()
            for param in target_net.parameters():
                l2_reg += torch.norm(param)
            total_loss += l2_reg

        optimizer.zero_grad()  # 각 파라미터의 기울기 값을 0으로 만듦
        total_loss.backward()  # chain rule을 이용한 기울기 계산
        if args.enable_clip_gradient:
            network.clip_gradient(target_net, 10.)

        optimizer.step()  # 가중치 업데이트

        # measure elapsed time

        batch_time.update(time.time() - end)
        end = time.time()

        # print('testing !! break train') ##
        # break ##
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
            if use_default_box:
                output = output + default_box
            else:
                output = output

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

            # TransferNet7 전용
            print('\t x_loss: %.4f' % (res_losses[0] / count))
            print('\t y_loss: %.4f' % (res_losses[1] / count))
            print('\t z_loss: %.4f' % (res_losses[2] / count))
            print('\t w_loss: %.4f' % (res_losses[3] / count))
            print('\t h_loss: %.4f' % (res_losses[4] / count))
            print('\t d_loss: %.4f' % (res_losses[5] / count))
    return miou, acc

def test(test_loader, net, threshold=0.5, debug=False):
    global args

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
        if show_iter:  ##
            print(('\n' + '*' * 8 + ' [iter:{}] '.format(i) + '*' * 8))
        # show_DB_shape = False ## DB shape 출력
        if show_DB_shape:  ##
            print('-- show_DB_shape --')
            print('boxes :', boxes.size())
            print('agent :', agent.size())
            print('targets :', targets.size())
            print('boxes[0] :', boxes[0].shape)
            print('agent :', agent)
            print('targets[0] :', targets[0].shape)
            print('-------------------')

        im_data = Variable(im_data.cuda())  ## 추가 !!
        depth = Variable(depth.cuda())  ## 추가 !!

        # batch_size가 1이므로, [0]은 그냥 한꺼풀 꺼내는걸 의미함
        output, total_loss, losses = net(im_data, depth, boxes.numpy(), agent, targets)

        output = output.data.cpu().numpy()
        targets = targets.data.cpu().numpy()
        default_box = default_box.data.cpu().numpy()
        agent = agent.data.cpu().numpy()
        gt_boxes_3d = gt_boxes_3d.data.cpu().numpy()
        if use_default_box:
            output2 = output + default_box
        else:
            output2 = output
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

    # TransferNet6 전용
    # print('\t xz_loss: %.4f' % (result_losses[0]))
    # print('\t y_loss: %.4f' % (result_losses[1]))
    # print('\t wd_loss: %.4f' % (result_losses[2]))

    # TransferNet7 전용
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
    # x, y, z는 중앙값이라는 전제 (사실 상관없음)
    half = pos[3:]/2

    return pos[:3]-half, pos[:3]+half

if __name__ == '__main__':
    main()
