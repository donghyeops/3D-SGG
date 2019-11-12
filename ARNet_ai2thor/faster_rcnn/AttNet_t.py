# -*- coding: utf-8 -*-
# code for AttNet


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import network
from fast_rcnn.nms_wrapper import nms
from network import Conv2d, FC, my_FC
from roi_pooling.modules.roi_pool import RoIPool
from matplotlib import pyplot as plt

PRINT_MODEL_PROCESS = False


def nms_detections(pred_boxes, scores, nms_thresh, inds=None):
    dets = np.hstack((pred_boxes,
                      scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, nms_thresh)
    if inds is None:
        return pred_boxes[keep], scores[keep], keep
    return pred_boxes[keep], scores[keep], inds[keep], keep


class AttNet(nn.Module):
    def __init__(self,
                 nembedding=128,
                 nlabel=25, ncolor=10, nopen_state=3,
                 class_weight_color=None,
                 class_weight_os=None):
        super(AttNet, self).__init__()
        self.class_weight_color = None #class_weight_color
        self.class_weight_os = None #class_weight_os
        print('**** AttNet.py config ****')

        self.use_label = False
        self.use_visual_feature = False
        self.use_row_visual_for_color = False
        self.dropout = False
        print('# use_label:', self.use_label)
        print('# visual_feature:', self.use_visual_feature)
        print('nembedding: ', nembedding)
        print('nlabel: ', nlabel)
        print('ncolor: ', ncolor)
        print('nopen_state: ', nopen_state)
        print('dropout: ', self.dropout)
        print('************************')

        self.roi_pool_object = RoIPool(32, 32, 1.0)
        # cnn
        bn = True
        self.conv1 = nn.Sequential(Conv2d(3, 64, 3, same_padding=True, bn=False),
                                   Conv2d(64, 64, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(Conv2d(64, 128, 3, same_padding=True, bn=False),
                                   Conv2d(128, 128, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(Conv2d(128, 256, 3, same_padding=True, bn=False),
                                   Conv2d(256, 256, 3, same_padding=True, bn=bn))

        self.fc1_color = FC(16384, 300)
        self.fc2_color = FC(300, 300)
        self.pred_color = FC(300, ncolor, relu=False)

        self.fc1_open_state = FC(16384, 300)
        self.fc2_open_state = FC(300, 300)
        self.pred_open_state = FC(300, nopen_state, relu=False)

        network.weights_normal_init(self.pred_color, 0.01)
        network.weights_normal_init(self.pred_open_state, 0.01)

    def forward(self, im_data, im_info, objects, gt_colors=None, gt_open_states=None):
        # objects.size() : (obj#, 5[xmin, ymin, xmax, ymax, obj_class]) t:numpy

        if PRINT_MODEL_PROCESS:
            print('im_data.shape:', im_data.shape)
            print('gt_objects.shape:', objects.shape)
        bboxes = np.zeros((len(objects), 5))
        # bboxes : (obj#, 5[0, xmin, ymin, xmax, ymax])
        bboxes[:, 1:] = objects[:, :4] # roi pooling 함수를 위해 0을 넣음
        bboxes = network.np_to_variable(bboxes, is_cuda=True)
        labels = network.np_to_variable(np.expand_dims(objects[:, 4], -1), is_cuda=True)
        labels = labels.long()

        ### common img ###
        # im_data : (1, 3, 600, 600)
        # roi_pool_object에 들어가는 좌표 데이터는 (obj#, 5[0, xmin, ymin, xmax, ymax]), cuda_tensor 여야함
        obj_vis = self.roi_pool_object(im_data, bboxes)
        box = obj_vis.data.cpu().numpy()
        im_data = im_data.data.cpu().numpy()[0]
        #box = np.transpose(box[0], (1, 2, 0))
        #box[0]
        print(im_data)
        print(im_data.shape)
        print(im_data.dtype)
        for i in range(len(objects)):
            print('obj:', objects[i], 'color:', gt_colors[i], 'os:', gt_open_states[i])
        w = 20
        h = 10
        fig = plt.figure(figsize=(8, 8))
        columns = len(box)
        rows = 1
        for i in range(1, columns * rows + 1):
            img = np.random.randint(10, size=(h, w))
            fig.add_subplot(rows, columns, i)
            plt.imshow(np.transpose(box[i-1], (1, 2, 0)))
        fig.add_subplot(2, 1, 2)
        plt.imshow(np.transpose(im_data, (1, 2, 0)))
        plt.show()

        return None, None


    def evaluate(self, im_data, im_info, objects, gt_colors, gt_open_states):
        # original code
        color_prob, open_state_prob = self(im_data, im_info, objects, gt_colors, gt_open_states)

        color_cnt, color_correct_cnt = self.get_recall(color_prob, gt_colors)
        os_cnt, os_correct_cnt = self.get_recall(open_state_prob, gt_open_states)

        return (color_cnt, color_correct_cnt), (os_cnt, os_correct_cnt)

    @staticmethod
    def get_recall(pred_prob, gt):
        pred_prob = pred_prob.data.cpu().numpy()
        gt = gt.data.cpu().numpy()

        gt_cnt = len(gt)

        pred = pred_prob.argmax(1)
        rel_correct_cnt = np.sum(np.equal(pred, gt))

        return gt_cnt, rel_correct_cnt

    def build_loss_att(self, color_score, open_state_score, gt_colors, gt_open_states):
        if self.class_weight_color is not None:
            color_weights = np.sqrt(self.class_weight_color)  # 값이 너무 큰 것들이 있어서, 루트씌워줌 (MSDN_base.py에서도 이렇게함)
            color_weights = color_weights.cuda()
            ce_color = F.cross_entropy(color_score, gt_colors, weight=color_weights)
        else:
            ce_color = F.cross_entropy(color_score, gt_colors)
        if self.class_weight_os is not None:
            #os_weights = np.sqrt(self.class_weight_os)
            os_weights = self.class_weight_os.cuda()
            ce_open_state = F.cross_entropy(open_state_score, gt_open_states, weight=os_weights)
        else:
            ce_open_state = F.cross_entropy(open_state_score, gt_open_states)
        total_loss = 1.5 * ce_color + ce_open_state  # color가 학습이 잘 안되서 1.5곱해봄

        return total_loss, (ce_color, ce_open_state)

