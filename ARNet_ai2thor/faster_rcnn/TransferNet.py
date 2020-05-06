# -*- coding: utf-8 -*-
# code for ARNet

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os.path as osp


from utils.timer import Timer
from fast_rcnn.nms_wrapper import nms
from rpn_msr.proposal_target_layer_hdn import proposal_target_layer as proposal_target_layer_py
from fast_rcnn.config import cfg

import network
from network import my_FC, FC, Conv2d
from roi_pooling.modules.roi_pool import RoIPool
from cbp.compact_bilinear_pooling import CompactBilinearPooling as CBP

show_RPN_results_size = False  ## RPN output 출력
show_proposal_target_layer_results_size = False  ## proposal_target_layer output 출력

DEBUG = False
TIME_IT = cfg.TIME_IT
PRINT_MODEL_PROCESS = False

# 8부터는 roidb 사용
# 9: only vision

class TransferNet(nn.Module):
    def __init__(self, nlabel=25, output_size=6, use_img=True, use_class=True, use_bbox=True, use_angle=True):
        super(TransferNet, self).__init__()
        print('**** TransferNet ****')
        # input data
        # 1. roi_image (batch, 3, 64, 64)
        # 2. roi_depth (batch, 1, 64, 64)
        # 2. bbox (batch, 5[(x, y, w, h, label]) # OD 결과값 (학습 시에는 OD의 정답 bbox가 들어감)
        # 2. agent (4[(x, y, z, rotation]) #rotation은 0~3 값

        # output data (answer)
        # (batch, 5[pos, size])
        # 1. 3D bbox pos (x, y, z)
        # 1. 3D bbox size (wd, h)
        
        self.use_img = use_img
        self.use_class = use_class
        self.use_bbox = use_bbox
        self.use_angle = use_angle
        
        self.dropout = False
        nembedding = 20
        print('nlabel: ', nlabel)
        print('output_size: ', output_size)
        print('nembedding: ', nembedding)
        print('use_img: ', self.use_img)
        print('use_class: ', self.use_class)
        print('use_bbox: ', self.use_bbox)
        print('use_angle: ', self.use_angle)
        print('dropout: ', self.dropout)
        print('************************')

        bn = False
        # input: 64,64
        combine_feature_size = 0
        if self.use_img:
            end_dims = 64
            self.convs = nn.Sequential(Conv2d(3, 16, 3, same_padding=True),
                                       Conv2d(16, 32, 3, same_padding=True),
                                       nn.MaxPool2d(2),
                                       Conv2d(32, 32, 3, same_padding=True),
                                       Conv2d(32, 64, 3, same_padding=True),
                                       nn.MaxPool2d(2),
                                       Conv2d(64, 64, 3, same_padding=True),
                                       Conv2d(64, end_dims, 3, same_padding=True),
                                       nn.MaxPool2d(2))
            
            self.dconvs = nn.Sequential(Conv2d(1, 16, 3, same_padding=True),
                                        Conv2d(16, 32, 3, same_padding=True),
                                        nn.MaxPool2d(2),
                                        Conv2d(32, 32, 3, same_padding=True),
                                        Conv2d(32, 64, 3, same_padding=True),
                                        nn.MaxPool2d(2),
                                        Conv2d(64, 64, 3, same_padding=True),
                                        Conv2d(64, end_dims, 3, same_padding=True),
                                        nn.MaxPool2d(2))
    
            self.vf_fc1 = my_FC(8*8*end_dims*2, 512, activation='relu')
            self.vf_fc2 = my_FC(512, 512, activation='relu')
            combine_feature_size += 512

        # mid_a = 'relu'
        # output_a = 'None'
        if self.use_class:
            self.word_embedding = nn.Embedding(nlabel, nembedding)
            self.fc_embedding = my_FC(nembedding, 32)
            combine_feature_size += 32

        if self.use_bbox:
            self.base_fc1 = my_FC(20, 512, activation='relu')
            self.base_fc2 = my_FC(512, 512, activation='relu')
            combine_feature_size += 512

        if self.use_angle:
            combine_feature_size += 1

        self.combine_fc1 = my_FC(combine_feature_size, 1024, activation='relu')
        self.combine_fc2 = my_FC(1024, 1024, activation='relu')

        self.pred_fc = my_FC(1024, output_size, activation='None')

        #network.weights_normal_init(self.pred_fc, 0.01)

    def forward(self, roi_imgs, roi_depths, objects, agent=None, targets=None):
        if PRINT_MODEL_PROCESS:
            print('objects.shape:', objects.shape)
            print('agent.shape:', agent.shape)
            print('gt_boxes_3d.shape:', targets.shape)
        features = [] # 마지막 fc의 입력으로 사용될 피쳐들 등록
        ### img ###
        if self.use_img:
            #obj_vis = torch.cat((roi_imgs, roi_depths), 1)  # (batch, 3, 600, 600) + (batch, 1, 600, 600)
            
            conv_f = self.convs(roi_imgs)
            flat_conv_f = conv_f.view(roi_imgs.size()[0], -1)
    
            dconv_f = self.dconvs(roi_depths)
            flat_dconv_f = dconv_f.view(roi_depths.size()[0], -1)
            
            obj_vis = torch.cat((flat_conv_f, flat_dconv_f), 1)
            vf1 = self.vf_fc1(obj_vis)
            if self.dropout:
                vf1 = F.dropout(vf1, p=0.3, training=self.training)
            vf2 = self.vf_fc2(vf1)
            if self.dropout:
                vf2 = F.dropout(vf2, p=0.3, training=self.training)
            features.append(vf2)

        ### word ###
        if self.use_class:
            object_label = network.np_to_variable(objects[:, 4, np.newaxis], is_cuda=True)
            object_label = object_label.long()
            embedding = self.word_embedding(object_label).view(object_label.size()[0], -1)
            wf = self.fc_embedding(embedding)
            features.append(wf)
            
        ### 2d bbox ###
        if self.use_bbox:
            bboxes = np.zeros((len(objects), 20))
            bboxes[:, :4] = objects[:, :4]
            #bboxes[:] -= 300 # center base
            bboxes = bboxes / 600
            bboxes[:, 4] = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
            bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
            bboxes[:, 1] = 1 - bboxes[:, 1] # reverse
    
            index=5
            for i in range(5):
                for j in range(5):
                    if j<i:
                        continue
                    bboxes[:, index]=bboxes[:, i]*bboxes[:, j]
                    index += 1
    
            factors = network.np_to_variable(bboxes, is_cuda=True)
            #s_factors = factors * factors
            b_fc1 = self.base_fc1(factors)
            b_fc2 = self.base_fc2(b_fc1)
            if self.dropout:
                b_fc2 = F.dropout(b_fc2, p=0.1, training=self.training)
            features.append(b_fc2)
        
        if self.use_angle:
            angles = np.zeros((len(objects), 1))
            angles[:, 0] = agent[:, -1]
            angles = network.np_to_variable(angles, is_cuda=True)
            features.append(angles)

        ### combine (base + context) ###
        # base 피쳐에 context 피쳐를 결합하여 최종 결과 예측
        if len(features) > 1: 
            combine_fc1 = self.combine_fc1(torch.cat(features, 1))
        else:
            combine_fc1 = self.combine_fc1(features[0])
            
        if self.dropout:
            combine_fc1 = F.dropout(combine_fc1, p=0.3, training=self.training)
        combine_fc2 = self.combine_fc2(combine_fc1)
        if self.dropout:
            combine_fc2 = F.dropout(combine_fc2, p=0.3, training=self.training)
        output = self.pred_fc(combine_fc2)
        
        if targets is not None:
            total_loss, losses = self.build_loss(output, targets.cuda())
        else:
            total_loss, losses = None, None # test시엔 이걸로
        return output, total_loss, losses


    def build_loss(self, output, targets):
        #print(targets)
        # output : [내기준 x 차이값, y 차이값, 내기준 너비]
        loss_func = F.l1_loss#F.smooth_l1_loss
        x_loss = loss_func(output[:, 0], targets[:, 0]) # xz 축 검사
        y_loss = loss_func(output[:, 1], targets[:, 1])
        z_loss = loss_func(output[:, 2], targets[:, 2])
        w_loss = loss_func(output[:, 3], targets[:, 3])
        h_loss = loss_func(output[:, 4], targets[:, 4])
        d_loss = loss_func(output[:, 5], targets[:, 5])
        #F.l1_loss
        #F.smooth_l1_loss()
        #F.mse_loss()

        total_loss = 10*(x_loss + y_loss + z_loss) + w_loss + h_loss + d_loss
        losses = [x_loss, y_loss, z_loss, w_loss, h_loss, d_loss]
        return total_loss, losses

