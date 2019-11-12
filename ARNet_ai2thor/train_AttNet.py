# -*- coding: utf-8 -*-
import argparse
import os
import random
import shutil
import time

import numpy as np
import torch
# To log the training process
from tensorboard_logger import configure
from torch.autograd import Variable

from faster_rcnn import network

# from faster_rcnn.AttNet import AttNet
# from faster_rcnn.AttNet2 import AttNet
# from faster_rcnn.AttNet3 import AttNet
from faster_rcnn.AttNet5 import AttNet
# from faster_rcnn.AttNet_t import AttNet

from faster_rcnn.datasets.ai2thor_attribute_dataset_loader import ai2thor_attribute_dataset
from faster_rcnn.fast_rcnn.config import cfg
from faster_rcnn.utils.HDN_utils import get_model_name2

## 피쳐 사이즈 출력
show_iter = False  # iteration 출력
show_DB_shape = False  # Network input data 출력

TIME_IT = cfg.TIME_IT
parser = argparse.ArgumentParser('Options for training Hierarchical Descriptive Model in pytorch')

# Training parameters
parser.add_argument('--lr', type=float, default=99999, metavar='LR', help='base learning rate for training')
parser.add_argument('--max_epoch', type=int, default=50, metavar='N', help='max iterations for training')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='percentage of past parameters to store')
parser.add_argument('--step_size', type=int, default=99999, help='Step size for reduce learning rate')
parser.add_argument('--enable_clip_gradient', action='store_true', help='Whether to clip the gradient')
parser.add_argument('--use_normal_anchors', action='store_true', help='Whether to use kmeans anchors')

# structure settings
parser.add_argument('--disable_language_model', action='store_true', help='To disable the Lanuage Model ')
parser.add_argument('--mps_feature_len', type=int, default=1024, help='The expected feature length of message passing')
parser.add_argument('--dropout', action='store_true', help='To enables the dropout')
parser.add_argument('--nembedding', type=int, default=128, help='The size of word embedding')
parser.add_argument('--use_kernel_function', action='store_true')
# Environment Settings
parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--saved_model_path', type=str, default='model/pretrained_models/VGG_imagenet.npy',
                    help='The Model used for initialize')
parser.add_argument('--dataset_option', type=str, default='small', help='The dataset to use (small | normal | fat)')
parser.add_argument('--output_dir', type=str, default='./output', help='Location to output the model')
parser.add_argument('--model_name', type=str, default='MSDN', help='The name for saving model.')
parser.add_argument('--nesterov', action='store_true', help='Set to use the nesterov for SGD')
parser.add_argument('--finetune_language_model', action='store_true',
                    help='Set to disable the update of other parameters')
parser.add_argument('--optimizer', type=int, default=1,
                    help='which optimizer used for optimize language model [0: SGD | 1: Adam | 2: Adagrad]')

###############################
# train_hdn.py
parser.add_argument('--model_tag', type=str, default='#0', help='모델명 정의')
parser.add_argument('--use_pin_memory', dest='use_pin_memory', action='store_true')
parser.add_argument('--no-use_pin_memory', dest='use_pin_memory', action='store_false')
parser.set_defaults(use_pin_memory=True)
parser.add_argument('--measure', type=str, default='sgg', help='sgg(default) | others | categorical_pred')

# MSDN.py, MSDN_base.py, Language_Model.py
parser.add_argument('--refiner_name', type=str, default='None', help='choose (None | mpu | 1cm_r | 1cm_c | 2cm)')

# MSDN.py, MSDN_base.py
parser.add_argument('--spatial_name', type=str, default='bbox', help='choose (None | bbox | mask)')

# MSDN_base.py
parser.add_argument('--cls_loss_name', type=str, default='cross_entropy', help='choose (cross_entropy | focal_loss)')

parser.add_argument('--evaluate', action='store_true', help='Only use the testing mode')
parser.add_argument('--evaluate_target', type=str, default='sg', help='choose (sg | caption)')
parser.add_argument('--visualize', action='store_true', help='Only use the testing mode')
parser.add_argument('--visualize_index', type=int, default=-1, help='input dataset index')
parser.add_argument('--visualize_dataset', type=str, default='test', help='choose (test | train)')
###############################


args = parser.parse_args()
# Overall loss logger
overall_train_loss = network.AverageMeter()


def main():
    global args
    lr = 0.001
    step_size = 10
    use_class = True
    use_vis = True
    test_mode = False

    args = get_model_name2(args)

    visualization = False
    opt = 'Adam'  # adam / sgd

    if opt == 'adam' or opt == 'Adam':
        optimizer_class = torch.optim.Adam
    if opt == 'sgd' or opt == 'SGD':
        optimizer_class = torch.optim.SGD
    ############### 하이퍼 파라미터 출력 ###############
    print('*' * 25)
    print('*')
    print('* test_mode :', test_mode)
    print('* lr :', lr)
    print('* epochs :', args.max_epoch)
    print('* step_size :', step_size)
    print('* use_class :', use_class)
    print('* use_vis :', use_vis)
    print('* pin_memory :', args.use_pin_memory)


    print('* optimizer :', opt)
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

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=8,
                                               pin_memory=args.use_pin_memory)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8,
                                              pin_memory=args.use_pin_memory)

    train_set.inverse_weight_object = None  # 이 부분은 아직 미구현 (in dataset_loader)
    train_set.inverse_weight_predicate = None  # 이 부분은 아직 미구현 (in dataset_loader)
    # Model declaration

    net = AttNet(nlabel=train_set.num_object_classes,
                 ncolor=train_set.num_color_classes,
                 nopen_state=train_set.num_open_state_classes,
                 class_weight_color=train_set.class_weight_color,
                 class_weight_os=train_set.class_weight_os,
                 use_class=use_class,
                 use_vis=use_vis)
    print(net)

    params = list(net.parameters())
    # for param in params:
    #    print param.size()
    # print net

    # Setting the state of the training model
    net.cuda()  # 모델 파라미터들을 GPU로 옮김

    if visualization:
        net2 = AttNet()
        net2.cuda()  # 모델 파라미터들을 GPU로 옮김
        net2.eval()
        net2.load_state_dict(torch.load('./output/AttNet.state'))  # ROI는 pickle 형태로 저장 불가하다함
        for param in net2.parameters():
            print(param.size(), param.data.sum())
        print('fc1_color', net2.fc1_color.fc.weight.data.sum())
        # print('conv3', net2.c_conv3[0].weight.data.sum())
        visualize(test_loader, net2)
        raise Exception("done")

    if not args.evaluate:
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

    best_acc = 0.
    best_color_acc = 0.
    best_open_state_acc = 0.

    if test_mode:
        net.load_state_dict(torch.load('./output/AttNet.state'))
        color_acc, open_state_acc = test(test_loader, net, test_set.num_object_classes)

        # print('======= Testing Result =======')
        # print('\t[att.color R] {acc:2.3f}%%'.format(recall=color_acc * 100))
        # print('\t[att.open_state R] {acc:2.3f}%%'.format(recall=open_state_acc * 100))
        # print('==============================')
    else:
        for epoch in range(0, args.max_epoch):
            # Training
            train(train_loader, target_net, optimizer, epoch)

            # Testing
            # network.set_trainable(net, False) # Without backward(), requires_grad takes no effect

            # color_recall, open_state_recall = test(test_loader, target_net, train_set.num_object_classes)
            color_acc, open_state_acc = test(train_loader, target_net, train_set.num_object_classes)  # train으로 테스트

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
            if epoch % step_size == 0 and epoch > 0:
                lr /= 10
                # lr *= 0.95
                # args.lr = lr
                print('[learning rate: {}]'.format(lr))

                # update optimizer and correponding requires_grad state
                optimizer = optimizer_class([
                    {'params': params, 'lr': lr}
                ], lr=lr, weight_decay=0.0005)
                # save_name = os.path.join(args.output_dir, 'AttNet.state')
                # torch.save(net.state_dict(), save_name)
                # print(('save model: {}'.format(save_name)))

        ## model save !!
        # save_name = os.path.join(args.output_dir, 'AttNet.h5')
        # network.save_net(save_name, net)
        # print(('save model: {}'.format(save_name)))
        save_name = os.path.join(args.output_dir, 'AttNet.state')
        torch.save(net.state_dict(), save_name)
        print(('save model: {}'.format(save_name)))


def visualize(train_loader, target_net):
    for i, (im_data, im_info, objects, gt_colors, gt_open_states) in enumerate(train_loader):
        if len(objects.numpy()[0]) < 1:  # 물체 1개 이상만 통과
            continue

        im_data = Variable(im_data.cuda())  ## 추가 !!

        target_net.visualize(im_data, im_info, objects.numpy()[0], gt_colors[0], gt_open_states[0])


def train(train_loader, target_net, optimizer, epoch):
    global args

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

        # show_iter = False ## iteration(i) 출력
        if show_iter:  ##
            print(('\n' + '*' * 8 + ' [iter:{}] '.format(i) + '*' * 8))
        # show_DB_shape = False ## DB shape 출력
        if show_DB_shape:  ##
            print('-- show_DB_shape --')
            print(('im_data :', im_data.size()))
            print(('im_info :', im_info.size()))
            print(('gt_objects :', objects.size()))
            print(('gt_objects.numpy()[0] :', objects.numpy()[0].shape))
            print('-------------------')
        # measure the data loading time
        data_time.update(time.time() - end)

        im_data = Variable(im_data.cuda())  ## 추가 !!

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

        optimizer.zero_grad()  # 각 파라미터의 기울기 값을 0으로 만듦
        loss.backward()  # chain rule을 이용한 기울기 계산
        if args.enable_clip_gradient:
            network.clip_gradient(target_net, 10.)
        optimizer.step()  # 가중치 업데이트

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print('testing !! break train') ##
        # break ##

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
    # For efficiency inference

    batch_time = network.AverageMeter()
    end = time.time()

    color_cm = np.zeros(
        (10, 10))  # (pred, gt) # http://text-analytics101.rxnlp.com/2014/10/computing-precision-and-recall-for.html
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

