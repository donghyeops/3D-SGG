# -*- coding: utf-8 -*-
# code for ARNet

import cv2
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F

from fast_rcnn.nms_wrapper import nms
from fast_rcnn.config import cfg
from utils.cython_bbox import bbox_overlaps

import network
from network import Conv2d, FC, my_FC
from ARNet_base import ARNet_base
import pdb

show_RPN_results_size = False  ## RPN output 출력
show_proposal_target_layer_results_size = False  ## proposal_target_layer output 출력

DEBUG = False
TIME_IT = cfg.TIME_IT
PRINT_MODEL_PROCESS = False

def nms_detections(pred_boxes, scores, nms_thresh, inds=None):
    dets = np.hstack((pred_boxes,
                      scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, nms_thresh)
    if inds is None:
        return pred_boxes[keep], scores[keep], keep
    return pred_boxes[keep], scores[keep], inds[keep], keep


class RelNet(ARNet_base):
    def __init__(self, nhidden, n_object_cats, n_predicate_cats,
                 object_loss_weight,
                 predicate_loss_weight,
                 dropout=False,
                 use_kmeans_anchors=False,
                 nembedding=300,
                 refiner_name='None', spatial_name='None', cls_loss_name='cross_entropy',
                 nlabel=25, nrelation=9):
        print('**** RelNet.py config ****')
        print(('show_proposal_target_layer_results_size :', show_proposal_target_layer_results_size))
        print('************************')
        super(RelNet, self).__init__(nhidden, n_object_cats, n_predicate_cats,
                                                             object_loss_weight, predicate_loss_weight,
                                                             dropout, use_kmeans_anchors, nembedding,
                                                             refiner_name, spatial_name, cls_loss_name)
        self.predicate_loss_weight = predicate_loss_weight

        self.dropout = True
        print('relnet3')
        print('*'*40)

        self.coordinate_fc1 = my_FC(6*2, 512, activation='relu')
        self.coordinate_fc2 = my_FC(512, 512, activation='relu')
        #self.coordinate_fc3 = my_FC(512, 512, activation='tanh')

        self.pred_relation = FC(512, nrelation, relu=False)

        network.weights_normal_init(self.pred_relation, 0.01)

    def forward(self, boxes_3d, gt_relationships=None):
        # input data (global coordinate of objects)
        # 1. 3D bbox (x, y, z, w, h, d, label)

        # output data
        # 1. relations (box#, box#)

        # sub_boxes_3d : (box#, 6) 마지막 label을 분리함
        # sub_label : (box#, 1)
        sub_boxes_3d, obj_boxes_3d, sub_label, obj_label, target_relationship, \
        mat_object, mat_phrase = \
            self.make_relationship_connection(boxes_3d, gt_relationships)

        #### relationship ####
        # idea1 : 라벨과 좌표로부터 독립적으로 관계를 구하고, max_pooling해서 보간해보자.

        coordinate_feature = torch.cat((sub_boxes_3d, obj_boxes_3d), 1)
        #print(coordinate_feature)
        coordinate_fc1 = self.coordinate_fc1(coordinate_feature)
        if self.dropout:
            coordinate_fc1 = F.dropout(coordinate_fc1, training=self.training)
        coordinate_fc2 = self.coordinate_fc2(coordinate_fc1)
        if self.dropout:
            coordinate_fc2 = F.dropout(coordinate_fc2, training=self.training)
        #coordinate_fc3 = self.coordinate_fc3(coordinate_fc2)

        rel_score = self.pred_relation(coordinate_fc2)
        rel_prob = F.softmax(rel_score, -1)

        if self.training:
            self.ce_relation, self.tp_pred, self.tf_pred, self.fg_cnt_pred, self.bg_cnt_pred = \
                self.build_loss_rel(rel_score, target_relationship)
            # print 'accuracy: %2.2f%%' % (((self.tp_pred + self.tf_pred) / float(self.fg_cnt_pred + self.bg_cnt_pred)) * 100)
            # self.timer.tic()

        return boxes_3d[:, :6], rel_prob, target_relationship, mat_phrase

    def make_relationship_connection(self, boxes_3d, gt_relationships):
        boxes_3d = boxes_3d.data.cpu().numpy()
        if gt_relationships is not None:
            gt_relationships = gt_relationships.data.cpu().numpy()
        n_object = len(boxes_3d)

        # (obj#, obj#)의 인덱스를 구해줌 (gt_relationships == gt_relationships[sub_ids, obj_ids])
        sub_ids, obj_ids = np.meshgrid(list(range(n_object)), list(range(n_object)), indexing='ij')

        sub_ids = sub_ids.reshape(-1)
        obj_ids = obj_ids.reshape(-1)

        sub_boxes_3d = boxes_3d[sub_ids, :6] # (box#, 6)
        obj_boxes_3d = boxes_3d[obj_ids, :6] # (box#, 6)
        sub_label = np.expand_dims(boxes_3d[sub_ids, 6], 1) # (box#, 1)
        obj_label = np.expand_dims(boxes_3d[obj_ids, 6], 1) # (box#, 1)

        target_relationship = np.zeros_like(sub_ids)
        if gt_relationships is not None:
            for i in range(len(target_relationship)):
                target_relationship[i] = gt_relationships[sub_ids[i], obj_ids[i]]

        sub_boxes_3d = network.np_to_variable(sub_boxes_3d, is_cuda=True)
        obj_boxes_3d = network.np_to_variable(obj_boxes_3d, is_cuda=True)
        sub_label = network.np_to_variable(sub_label, is_cuda=True, dtype=torch.LongTensor)
        obj_label = network.np_to_variable(obj_label, is_cuda=True, dtype=torch.LongTensor)
        target_relationship = network.np_to_variable(target_relationship, is_cuda=True, dtype=torch.LongTensor)

        # 해당 물체가 해당 관계에서 sub인지, obj인지 구함
        # 0이 1이면 sub, 1이 1이면 obj
        mat_object = np.zeros((n_object, 2, len(sub_ids)), dtype=np.int64)

        # 해당 관계에서 sub와 obj 물체 인덱스 값
        mat_phrase = np.zeros((len(sub_ids), 2), dtype=np.int64)
        mat_phrase[:, 0] = sub_ids
        mat_phrase[:, 1] = obj_ids

        for i in range(len(sub_ids)):
            mat_object[sub_ids[i], 0, i] = 1
            mat_object[obj_ids[i], 1, i] = 1

        return sub_boxes_3d, obj_boxes_3d, sub_label, obj_label, target_relationship, mat_object, mat_phrase

    def evaluate_acc(self, boxes_3d, gt_relationships):
        # original code
        _, rel_prob, target_relationship, _ = \
            self(boxes_3d, gt_relationships)

        target_relationship = target_relationship.data.cpu().numpy()
        rel_prob = rel_prob.data.cpu().numpy()

        rel_cnt, rel_correct_cnt = self.get_recall(rel_prob, target_relationship)

        return rel_cnt, rel_correct_cnt

    def evaluate(self, boxes_3d, gt_relationships):
        # original code
        _, rel_prob, target_relationship, _ = self(boxes_3d, gt_relationships)

        def get_result(pred_prob, gt):
            pred_prob = pred_prob.data.cpu().numpy()
            gt = gt.data.cpu().numpy()

            pred = pred_prob.argmax(1)
            return pred, gt

        rel_pred, rel_gt = get_result(rel_prob, target_relationship)

        return rel_pred, rel_gt

    def get_recall(self, rel_prob, target_relationship):
        rel_cnt = len(target_relationship[target_relationship!=0])

        pred = rel_prob.argmax(1)
        #print(target_relationship)

        rel_correct_cnt = np.sum(np.logical_and(np.equal(pred, target_relationship), target_relationship!=0))
        #np.sum(np.equal(pred, target_relationship))
        #print(rel_cnt, rel_correct_cnt)
        return rel_cnt, rel_correct_cnt
