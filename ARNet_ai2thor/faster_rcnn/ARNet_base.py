# -*- coding: utf-8 -*-

import os.path as osp

import cv2
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F

import network


class ARNet_base(nn.Module):
    def __init__(self, predicate_loss_weight):
        super(ARNet_base, self).__init__()
        self.predicate_loss_weight = predicate_loss_weight

        # loss
        self.ce_color = None
        self.ce_open_state = None
        self.ce_relation = None

    @property
    def loss(self):
        return self.ce_color + self.ce_open_state + \
               self.ce_relation * 1

    @property
    def att_loss(self):
        return self.ce_color + self.ce_open_state

    @property
    def rel_loss(self):
        return self.ce_relation * 1

    def build_loss_att(self, color_score, open_state_score, gt_colors, gt_open_states):
        #gt_colors = gt_colors.squeeze()
        #gt_open_states = gt_open_states.squeeze()
        self.obj_cnt = len(gt_colors)
        gt_colors = network.np_to_variable(gt_colors, is_cuda=True, dtype=torch.LongTensor)
        gt_open_states = network.np_to_variable(gt_open_states, is_cuda=True, dtype=torch.LongTensor)


        ce_color = F.cross_entropy(color_score, gt_colors)
        maxv, color_predict = color_score.data.max(1)
        ce_open_state = F.cross_entropy(open_state_score, gt_open_states)
        maxv, open_state_predict = open_state_score.data.max(1)

        self.color_correct = torch.sum(color_predict[:].eq(gt_colors.data[:]))
        self.open_state_correct = torch.sum(open_state_predict[:].eq(gt_open_states.data[:]))

        # print loss_box
        return ce_color, ce_open_state

    # phrase classification
    def build_loss_rel(self, cls_score, labels):

        labels = labels.squeeze()  #

        fg_cnt = torch.sum(labels.data.ne(0))
        bg_cnt = labels.data.numel() - fg_cnt
        if self.predicate_loss_weight is not None:
            #ce_weights = np.sqrt(self.predicate_loss_weight)
            ce_weights = self.predicate_loss_weight
            #ce_weights[0] = float(fg_cnt) / (bg_cnt + 1e-5)
            #print(ce_weights)
            ce_weights = ce_weights.cuda()
            cls_loss = F.cross_entropy(cls_score, labels, weight=ce_weights)

        else:
            cls_loss = F.cross_entropy(cls_score, labels)
        _, predict = cls_score.data.max(1)

        if fg_cnt == 0:
            tp = 0
        else:
            tp = torch.sum(predict[bg_cnt:].eq(labels.data[bg_cnt:]))
        tf = torch.sum(predict[:bg_cnt].eq(labels.data[:bg_cnt]))
        fg_cnt = fg_cnt
        bg_cnt = bg_cnt

        return cls_loss, tp, tf, fg_cnt, bg_cnt
