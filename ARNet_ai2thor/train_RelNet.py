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
# To log the training process
from tensorboard_logger import configure, log_value
import sys
# reload(sys)
# sys.setdefaultencoding('utf8') # 안하면 plt import 오류남
import matplotlib.pyplot as plt  ## 시각화 함수를 위해 추가
import matplotlib.patches as patches
from PIL import Image

from faster_rcnn import network
from faster_rcnn.RelNet import RelNet  # 2 사용
from faster_rcnn.utils.timer import Timer
from faster_rcnn.fast_rcnn.config import cfg
from faster_rcnn.datasets.ai2thor_relation_dataset_loader import ai2thor_relation_dataset
from faster_rcnn.utils.HDN_utils import get_model_name2, group_params


# 피쳐 사이즈 출력
show_iter = False  # iteration 출력
show_DB_shape = False  # Network input data 출력

TIME_IT = cfg.TIME_IT
parser = argparse.ArgumentParser('Options for training Hierarchical Descriptive Model in pytorch')

# Training parameters
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='base learning rate for training')
parser.add_argument('--max_epoch', type=int, default=60, metavar='N', help='max iterations for training')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='percentage of past parameters to store')
parser.add_argument('--step_size', type=int, default=2, help='Step size for reduce learning rate')
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
parser.add_argument('--dataset_option', type=str, default='small', help='The datasets to use (small | normal | fat)')
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
parser.add_argument('--visualize_index', type=int, default=-1, help='input datasets index')
parser.add_argument('--visualize_dataset', type=str, default='test', help='choose (test | train)')
###############################


args = parser.parse_args()
# Overall loss logger
overall_train_loss = network.AverageMeter()
overall_train_rpn_loss = network.AverageMeter()

optimizer_select = 0
normalization = False

# python train_hdn.py
#    --load_RPN
#    --saved_model_path=./output/RPN/RPN_region_full_best.h5
#    --dataset_option=normal
#    --enable_clip_gradient
#    --step_size=2
#    --MPS_iter=1
#    --rnn_type LSTM_normal
def main():
    global args, optimizer_select
    # To set the model name automatically
    print(args)
    lr = 0.0001
    step_size = 20
    test_mode = False

    opt = 'adam'  # adam / sgd
    if opt == 'adam' or opt =='Adam':
        optimizer_class = torch.optim.Adam
    if opt == 'sgd' or opt =='SGD':
        optimizer_class = torch.optim.SGD
    dropout = False
    ############### 하이퍼 파라미터 출력 ###############
    print('*' * 25)
    print('*')
    print('* test_mode :', test_mode)
    print('* lr :', lr)
    print('* step_size :', step_size)
    print('* optimizer :', opt)
    print('* normalization :', normalization)
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

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=8,
                                               pin_memory=args.use_pin_memory)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8,
                                              pin_memory=args.use_pin_memory)

    train_set.inverse_weight_object = None  # 이 부분은 아직 미구현 (in dataset_loader)
    train_set.inverse_weight_predicate = None  # 이 부분은 아직 미구현 (in dataset_loader)
    # Model declaration
    net = RelNet(nhidden=args.mps_feature_len,
                n_object_cats=train_set.num_object_classes,
                n_predicate_cats=train_set.num_predicate_classes,
                object_loss_weight=train_set.inverse_weight_object,
                predicate_loss_weight=train_set.class_weight_relationship,
                dropout=args.dropout,
                use_kmeans_anchors=not args.use_normal_anchors,
                nembedding=args.nembedding,
                refiner_name=args.refiner_name,
                spatial_name=args.spatial_name,
                cls_loss_name=args.cls_loss_name,
                nlabel=train_set.num_object_classes,
                nrelation=train_set.num_predicate_classes)

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

    top_Ns = [50, 100]
    best_acc = 0.

    if test_mode:
        print('======= Testing Result =======')
        net.load_state_dict(torch.load('./output/RelNet_best.state'))
        color_acc, open_state_acc = test(test_loader, net, test_set.num_object_classes)
    elif args.visualize:
        if args.visualize_dataset == 'test':
            visualize(test_set, test_loader, net)
        else:
            visualize(train_set, train_loader, net)
    else:
        for epoch in range(0, args.max_epoch):
            # Training
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

            acc = test(test_loader, net, top_Ns, test_set.num_object_classes)

            if acc > best_acc:
                best_acc = acc

                save_name = os.path.join(args.output_dir, 'RelNet_best.pt')
                torch.save(net, save_name)
                print(('save model: {}'.format(save_name)))
            #
            # print('Epoch[{epoch:d}]:'.format(epoch=epoch))
            # print('\t[rel R@] {recall:2.3f}%% (best: {best_recall:2.3f}%%)'.format(
            #     recall=rel_recall * 100, best_recall=best_rel_recall * 100))
            # print('')

            # updating learning policy
            # if epoch % args.step_size == 0 and epoch > 0:
            if epoch > 0 and epoch % step_size == 0:
                lr /= 10
                print('[learning rate: {}]'.format(lr))

                # update optimizer and correponding requires_grad state
                optimizer = optimizer_class([
                    {'params': params, 'lr': lr}
                    ], lr=lr, weight_decay=0.0005)
                # save_name = os.path.join(args.output_dir, 'RelNet.pt')
                # torch.save(net, save_name)
                # print(('save model: {}'.format(save_name)))
        ## model save !!
        #save_name = os.path.join(args.output_dir, 'RelNet.h5')
        #network.save_net(save_name, net)
        #print(('save model: {}'.format(save_name)))
        save_name = os.path.join(args.output_dir, 'RelNet.pt')
        torch.save(net, save_name)
        print(('save model: {}'.format(save_name)))


def train(train_loader, target_net, optimizer, epoch):
    global args
    # Overall loss logger
    global overall_train_loss

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

        # show_iter = False ## iteration(i) 출력
        if show_iter:  ##
            print(('\n' + '*' * 8 + ' [iter:{}] '.format(i) + '*' * 8))
        # show_DB_shape = False ## DB shape 출력
        if show_DB_shape:  ##
            print('-- show_DB_shape --')
            print(('boxes_3d :', boxes_3d.size()))
            print(('gt_relationships :', gt_relationships.size()))
            print(('boxes_3d.numpy()[0] :', boxes_3d.numpy()[0].shape))
            print(('gt_relationships.numpy()[0] :', gt_relationships.numpy()[0].shape))
            print('-------------------')
        # measure the data loading time
        data_time.update(time.time() - end)

        # batch_size가 1이므로, [0]은 그냥 한꺼풀 꺼내는걸 의미함
        target_net(boxes_3d[0], gt_relationships[0])

        loss = target_net.ce_relation
        train_rel_loss.update(target_net.ce_relation.data.cpu().numpy(), 1)
        accuracy_rel.update(target_net.tp_pred, target_net.tf_pred, target_net.fg_cnt_pred, target_net.bg_cnt_pred)

        if normalization:
            # L2 정규화
            l2_reg = torch.tensor(0.).cuda()
            for param in target_net.parameters():
                l2_reg += torch.norm(param)
            loss += l2_reg

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
        if (i + 1) % 2000 == 0:
            print('Epoch: [{0}][{1}/{2}] \n\tBatch_Time: {batch_time.avg: .3f}s'.format(
                epoch, i + 1, len(train_loader), batch_time=batch_time))
            print('\t[rel_Loss]\trel_cls_loss: %.4f,' % (train_rel_loss.avg))
            # logging to tensor board
            log_value('FRCNN loss', train_rel_loss.avg, train_rel_loss.count)


def test(test_loader, net, top_Ns, num_object_classes, measure='sgg'):
    global args

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


def visualize(vg, data_loader, net, show_region=False, target_index_list=None, start_idx=0, threshold=0.0):
    global args
    # target_index_list=[14, 16, 80, 168, 182, 192, 242]
    # target_index_list=[192, 410, 868]
    target_index_list = [1795]
    threshold = 0.4
    object_threshold = 0.3
    start_idx = 1700
    print('======== Visualizing =======')
    net.eval()

    object_colors = plt.cm.hsv(np.linspace(0, 1, 151)).tolist()
    # im_data.size() : (1, 3, H, W)
    # im_info.size() : (1, 3[image_height, image_width, scale_ratios])
    # gt_objects.size() : (1, GT_obj#, 5[x1 ,y1 ,x2, y2, obj_class])
    # gt_relationships.size() : (1, GT_obj#, GT_obj#), int[pred_class]
    # gt_regions.size() : (1, GT_reg#, 15[?])

    for idx, (im_data, im_info, gt_objects, gt_relationships, gt_regions) in enumerate(data_loader):
        if target_index_list is not None:
            if not idx in target_index_list:
                if idx > target_index_list[-1]:
                    break
                continue
        elif idx < start_idx:
            continue
        V_im_data = Variable(im_data.cuda(), volatile=True)  ## 추가 !!
        total_cnt_t, rel_cnt_correct_t, object_result, subject_result, object_inds, subject_inds, predicate_inds, obj_scores, obj_boxes, obj_inds, region_result, objectiveness = net.evaluate_for_visualize(
            V_im_data, im_info, gt_objects.numpy()[0], gt_relationships.numpy()[0], gt_regions.numpy()[0],
            top_Ns=[100], nms=True)
        # cls_prob_object, bbox_object, object_rois = object_result[:3]
        # cls_prob_predicate, mat_phrase = predicate_result[:2]
        # region_caption, bbox_pred, region_rois, logprobs = region_result[:]

        print('index :', idx)
        print('  im_data :', im_data.shape, type(im_data))

        # ground truth object & region
        sh_gt_objects = mk_gt_list(gt_objects)
        sh_gt_regions = mk_gt_list(gt_regions)
        # ground truth triplet
        sh_gt_triplet = mk_gt_triplet(sh_gt_objects, gt_relationships)

        # make all object & object & subject & region list
        all_object = mk_result_all_object(obj_scores, obj_boxes, obj_inds, threshold=object_threshold)
        result_object_list = mk_result_list(object_result, object_inds)
        result_subject_list = mk_result_list(subject_result, subject_inds)
        result_region_list = mk_region_list(region_result[2].data)

        # make result triplet
        result_triplet_list = mk_result_triplet(all_object, result_subject_list, result_object_list, predicate_inds)

        # print "objectiveness",objectiveness.data

        # evaluate image
        percentage = evaluate_image(sh_gt_objects, all_object)  # object detection recall
        recall = evaluate_recall(sh_gt_triplet, result_triplet_list)  # triple recall
        print("  #percentage =", percentage * 100, "%")
        print('  #recall =', recall)
        if target_index_list is None:
            # if recall < threshold: ## 트리플 잘 찾은것만 거르기
            #    continue
            # if percentage < 0.4 or len(obj_boxes)<5 or len(result_triplet_list) < 5: # good case
            #    continue
            # if len(result_triplet_list) < 8: # 트리플이 너무 작으면 제외
            #    continue
            # if recall > 0.11 or percentage < 0.5: # bad case. 물체는 잘 잡았지만, 트리플을 못잡음
            #    continue
            # if percentage > 0.2 or len(obj_boxes)<5: # bad case. 물체 자체를 못잡음
            #    continue
            pass

        # make result triplet
        # result_triplet = mk_triplet(subject_inds,object_inds,predicate_inds,check_ind)

        # ================================================draw ground truth================================================
        output_image = np.array(Image.open(osp.join(cfg.IMG_DATA_DIR, json.load(open(osp.join(osp.join(cfg.DATA_DIR, \
                                                                                                       'visual_genome',
                                                                                                       'top_150_50'),
                                                                                              'test_small.json')))[idx][
            'path'])), dtype=np.uint8)
        cats = json.load(open(osp.join(osp.join(cfg.DATA_DIR, 'visual_genome', 'top_150_50'), 'categories.json')))
        cat_objects = cats['object']
        cat_predicates = cats['predicate']

        fig = plt.figure(figsize=(im_info[0][1] / 100.0 * 2.3, im_info[0][0] / 100.0),
                         dpi=100.0)  # figsize=(width, height)
        ax1 = plt.subplot(1, 2, 1)  # (1,2) 그리드에서 첫 번째
        ax1.axis('off')  # 눈금 제거
        # ax1.set_title("Ground Truth")
        ax2 = plt.subplot(1, 2, 2)
        ax2.axis('off')
        # ax2.set_title("Model Prediction")

        ax1.imshow(output_image)
        ax2.imshow(output_image)
        fontsize = 16
        linewidth = 3
        # phrase truth
        print(" <ground truth>")
        print("  #object =", len(sh_gt_objects))
        print("  #region =", len(sh_gt_regions))
        print("  #predicate =", len(sh_gt_triplet))

        # print triples
        print('-- gt triples --')
        for j in range(0, len(sh_gt_triplet)):
            print(cat_objects[int(sh_gt_triplet[j][0]) - 1], cat_predicates[int(sh_gt_triplet[j][2]) - 1],
                  cat_objects[int(sh_gt_triplet[j][1]) - 1])

        # object ground truth
        for j in range(0, len(sh_gt_objects)):
            rect = patches.Rectangle((sh_gt_objects[j][0], sh_gt_objects[j][1]), sh_gt_objects[j][2],
                                     sh_gt_objects[j][3], linewidth=linewidth,
                                     edgecolor=object_colors[int(sh_gt_objects[j][4]) - 1], facecolor='none')
            ax1.text(sh_gt_objects[j][0], sh_gt_objects[j][1] - 5, cat_objects[int(sh_gt_objects[j][4]) - 1],
                     style='italic',
                     bbox={'facecolor': object_colors[int(sh_gt_objects[j][4]) - 1], 'alpha': 0.5}, fontsize=fontsize)
            ax1.add_patch(rect)

        # region ground truth
        if show_region:
            for j in range(0, len(sh_gt_regions)):
                rect = patches.Rectangle((sh_gt_regions[j][0], sh_gt_regions[j][1]), sh_gt_regions[j][2],
                                         sh_gt_regions[j][3], linewidth=linewidth, edgecolor='b', facecolor='none')
                ax1.add_patch(rect)

        print("------------------")

        # ================================================draw result================================================

        # print triple
        print(" <result triplet>")
        print("  #object =", len(all_object))
        print("  #region =", len(result_region_list))
        print("  #predicate =", len(result_triplet_list))

        # print triples
        print('-- pred triples --')
        for j in range(0, len(result_triplet_list)):
            print(cat_objects[result_triplet_list[j][0] - 1], cat_predicates[result_triplet_list[j][2] - 1],
                  cat_objects[result_triplet_list[j][1] - 1])

        # draw object result
        for j in range(0, len(all_object)):
            rect = patches.Rectangle((all_object[j][0], all_object[j][1]), all_object[j][2], all_object[j][3],
                                     linewidth=linewidth, edgecolor=object_colors[all_object[j][4] - 1],
                                     facecolor='none')
            ax2.text(all_object[j][0], all_object[j][1] - 5, cat_objects[all_object[j][4] - 1], style='italic',
                     bbox={'facecolor': object_colors[all_object[j][4] - 1], 'alpha': 0.5}, fontsize=fontsize)
            ax2.add_patch(rect)

        print("------------------")
        ax1.set_title("GT. object#: {}, phrase#: {}".format(len(sh_gt_objects), len(sh_gt_triplet)))
        ax2.set_title(
            "pred. object#: {}, phrase#: {}, recall: {}".format(len(all_object), len(result_triplet_list), recall))

        # plt.tight_layout()
        plt.show()
        print('')


def evaluate_recall(gt_triple, pred_triple):
    gt_num = len(gt_triple)
    pred_num = len(pred_triple)
    correct_count = 0
    for pred_idx in range(pred_num):
        for gt_idx in range(gt_num):
            if int(pred_triple[pred_idx][0]) == int(gt_triple[gt_idx][0]) \
                    and int(pred_triple[pred_idx][1]) == int(gt_triple[gt_idx][1]) \
                    and int(pred_triple[pred_idx][2]) == int(gt_triple[gt_idx][2]):
                correct_count += 1
                break
    return float(correct_count) / gt_num


# make ground truth box list
def mk_gt_list(box):
    # result = [[x,y,w,h,ind],[]...]
    result = []
    for i in range(0, len(box[0])):
        x1 = box[0][i][0]
        y1 = box[0][i][1]
        x2 = box[0][i][2]
        y2 = box[0][i][3]
        gt_class = box[0][i][4]

        pre_result = []

        result_x = x1
        result_width = x2 - x1

        result_y = y1
        result_height = y2 - y1

        pre_result.append(result_x)
        pre_result.append(result_y)
        pre_result.append(result_width)
        pre_result.append(result_height)
        pre_result.append(gt_class)
        result.append(pre_result)

    return result


# make all object result box
def mk_result_all_object(obj_scores, box, obj_inds, threshold=0.5):
    # result = [[x,y,w,h,ind],[]...]
    result = []
    for i in range(0, len(obj_scores)):
        if obj_scores[i] > threshold:
            # for i in range(0,len(box)):
            x1 = box[i][0]
            y1 = box[i][1]
            x2 = box[i][2]
            y2 = box[i][3]
            idx = obj_inds[i]

            pre_result = []

            result_x = x1
            result_width = x2 - x1

            result_y = y1
            result_height = y2 - y1

            pre_result.append(result_x)
            pre_result.append(result_y)
            pre_result.append(result_width)
            pre_result.append(result_height)
            pre_result.append(idx)
            result.append(pre_result)

    return result


# make object & subject result box
def mk_result_list(box, box_inds):
    # result = [[x,y,w,h,ind],[]...]
    result = []
    for i in range(0, len(box)):
        x1 = box[i][0]
        y1 = box[i][1]
        x2 = box[i][2]
        y2 = box[i][3]
        index = box_inds[i]

        pre_result = []

        result_x = x1
        result_width = x2 - x1

        result_y = y1
        result_height = y2 - y1

        pre_result.append(result_x)
        pre_result.append(result_y)
        pre_result.append(result_width)
        pre_result.append(result_height)
        pre_result.append(index)
        result.append(pre_result)

    return result


# make region result box
def mk_region_list(box):
    # result = [[x,y,w,h],[]...]
    result = []
    for i in range(0, len(box)):
        x1 = box[i][1]
        y1 = box[i][2]
        x2 = box[i][3]
        y2 = box[i][4]

        pre_result = []

        result_x = x1
        result_width = x2 - x1

        result_y = y1
        result_height = y2 - y1

        pre_result.append(result_x)
        pre_result.append(result_y)
        pre_result.append(result_width)
        pre_result.append(result_height)
        result.append(pre_result)
    return result


# make ground truth triplet
def mk_gt_triplet(object_list, box):
    # result = [[subject_index,object_index,predicate_index],[]...]
    result = []
    for i in range(0, len(box[0])):
        for j in range(0, len(box[0][i])):
            if box[0][i][j] != 0:
                pre_result = []
                subject_index = object_list[i][4]
                object_index = object_list[j][4]
                predicate_index = box[0][i][j]

                pre_result.append(subject_index)
                pre_result.append(object_index)
                pre_result.append(predicate_index)
                result.append(pre_result)

    return result


# make result triplet
def mk_result_triplet(all_object, subject_list, object_list, predicate_list):
    # result = [[subject_index,object_index,predicate_index],[]...]
    result = []
    pred_num = len(predicate_list)

    for i in range(0, len(all_object)):
        if i == pred_num:
            break
        if subject_list[i] in all_object:
            if object_list[i] in all_object:
                if [subject_list[i][4], object_list[i][4], predicate_list[i]] not in result:
                    pre_result = []

                    pre_result.append(subject_list[i][4])
                    pre_result.append(object_list[i][4])
                    pre_result.append(predicate_list[i])
                    result.append(pre_result)
                if len(result) == 0:
                    pre_result = []

                    pre_result.append(subject_list[i][4])
                    pre_result.append(object_list[i][4])
                    pre_result.append(predicate_list[i])
                    result.append(pre_result)
    return result


# evaluate one image
def evaluate_image(gt_box, result_box):
    all_count = len(gt_box)
    correct_count = 0
    for i in range(0, len(gt_box)):
        for j in range(0, len(result_box)):
            if gt_box[i][4] == result_box[j][4]:
                gt_box_area = gt_box[i][2] * gt_box[i][3]
                result_box_area = result_box[j][2] * result_box[j][3]
                # inter_box_area = (max(x1,x'1)-min(x2,x'2))*(max(y1,y'1)-min(y2,y'2))
                inter_box_area = (max(gt_box[i][0], result_box[j][0]) - min(gt_box[i][0] + gt_box[i][2],
                                                                            result_box[j][0] + result_box[j][2])) * (
                                         max(gt_box[i][1], result_box[j][1]) - min(gt_box[i][1] + gt_box[i][3],
                                                                                   result_box[j][1] + result_box[j][3]))
                union_box_area = gt_box_area + result_box_area - inter_box_area
                IOU = (inter_box_area) / (union_box_area)
                if IOU > 0.5:
                    correct_count += 1
                    break
    return float(correct_count) / float(all_count)


if __name__ == '__main__':
    main()
