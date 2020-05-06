#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb
from cython_bbox import bbox_overlaps, bbox_intersections


def get_model_name2(arguments):
    arguments.model_name += '_{}'.format(arguments.model_tag)
    arguments.model_name += '_{}'.format(arguments.lr)

    if arguments.optimizer == 0:
        arguments.model_name += '_SGD'
        arguments.solver = 'SGD'
    elif arguments.optimizer == 1:
        arguments.model_name += '_Adam'
        arguments.solver = 'Adam'
    elif arguments.optimizer == 2:    
        arguments.model_name += '_Adagrad'
        arguments.solver = 'Adagrad'
    arguments.model_name += '_{}'.format(arguments.refiner_name)
    arguments.model_name += '_{}'.format(arguments.spatial_name)
    arguments.model_name += '_{}'.format(arguments.cls_loss_name)
    return arguments
    
    
def group_params(net_):
    vgg_params_fix = list(net_.cnn.features.parameters())[:8]
    vgg_params_var = list(net_.cnn.features.parameters())[8:]
    vgg_params_len = len(list(net_.cnn.features.parameters()))
    network_params = list(net_.parameters())[vgg_params_len:]
    print('vgg feature length:', vgg_params_len)
    print('HDN feature length:', len(network_params))
    return vgg_params_fix, vgg_params_var, network_params


def check_recall(rois, gt_objects, top_N, thres=0.5):
    overlaps = bbox_overlaps(
        np.ascontiguousarray(rois.cpu().data.numpy()[:top_N, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_objects[:, :4], dtype=np.float))

    overlap_gt = np.amax(overlaps, axis=0)
    correct_cnt = np.sum(overlap_gt >= thres)
    total_cnt = overlap_gt.size 
    return correct_cnt, total_cnt


def check_relationship_recall(gt_objects, gt_relationships, 
        subject_inds, object_inds, predicate_inds, 
        subject_boxes, object_boxes, top_Ns, thres=0.5, use_gt_boxes=False, only_predicate=False):
    # rearrange the ground truth
    gt_rel_sub_idx, gt_rel_obj_idx = np.where(gt_relationships > 0) # ground truth number
    gt_sub = gt_objects[gt_rel_sub_idx, :5] # (gt_rel#, 5)
    gt_obj = gt_objects[gt_rel_obj_idx, :5] # (gt_rel#, 5)
    gt_rel = gt_relationships[gt_rel_sub_idx, gt_rel_obj_idx] # (gt_rel#)
    # subject_inds : (100)
    # object_inds : (100)
    # predicate_inds : (100)
    # subject_boxes : (100, 4)
    # object_boxes : (100, 4)
    
    #print 'gt_sub.shape :', gt_sub.shape
    #print 'gt_obj.shape :', gt_obj.shape
    #print 'gt_rel.shape :', gt_rel.shape
    #print 'subject_inds.shape :', subject_inds.shape
    #print 'object_inds.shape :', object_inds.shape
    #print 'predicate_inds.shape :', predicate_inds.shape
    
    rel_cnt = len(gt_rel)
    rel_correct_cnt = np.zeros(len(top_Ns))
    max_topN = max(top_Ns)
    if len(subject_inds) == 0: ## 관계가 없으면
        return rel_cnt, rel_correct_cnt
    
    # compute the overlap
    # sub_overlaps : (100, gt_rel#)
    sub_overlaps = bbox_overlaps(
        np.ascontiguousarray(subject_boxes[:max_topN], dtype=np.float),
        np.ascontiguousarray(gt_sub[:, :4], dtype=np.float))
    # obj_overlaps : (100, gt_rel#)
    obj_overlaps = bbox_overlaps(
        np.ascontiguousarray(object_boxes[:max_topN], dtype=np.float),
        np.ascontiguousarray(gt_obj[:, :4], dtype=np.float))
        
    for idx, top_N in enumerate(top_Ns): # [50, 100]으로, 2번 실행됨

        for gt_id in range(rel_cnt): # gt relationship 개수만큼 반복
            # gt sub, obj와 iou일정 이상을 갖는 triple의 아이디를 구함
            fg_candidate = np.where(np.logical_and(
                sub_overlaps[:top_N, gt_id] >= thres, 
                obj_overlaps[:top_N, gt_id] >= thres))[0]
            
            for candidate_id in fg_candidate: # box가 맞는 트리플 수 만큼 반복
                if only_predicate:
                    if predicate_inds[candidate_id] == gt_rel[gt_id]: # 관계가 맞으면 +1
                        rel_correct_cnt[idx] += 1
                        break
                else:
                    if subject_inds[candidate_id] == gt_sub[gt_id, 4] and \
                            predicate_inds[candidate_id] == gt_rel[gt_id] and \
                            object_inds[candidate_id] == gt_obj[gt_id, 4]: # 관계랑 두 box class가 맞으면 +1

                        rel_correct_cnt[idx] += 1 
                        break
    return rel_cnt, rel_correct_cnt
    

# 관계 카테고리별로 총 개수와 맞춘 개수를 구함
def check_categorical_relationship_recall(gt_objects, gt_relationships, 
        subject_inds, object_inds, predicate_inds, 
        subject_boxes, object_boxes, top_N=5, thres=0.5):
    # rearrange the ground truth
    gt_rel_sub_idx, gt_rel_obj_idx = np.where(gt_relationships > 0) # ground truth number
    gt_sub = gt_objects[gt_rel_sub_idx, :5] # (gt_rel#, 5)
    gt_obj = gt_objects[gt_rel_obj_idx, :5] # (gt_rel#, 5)
    gt_rel = gt_relationships[gt_rel_sub_idx, gt_rel_obj_idx] # (gt_rel#)
    # subject_inds : (100)
    # object_inds : (100)
    # predicate_inds : (100)
    # subject_boxes : (100, 4)
    # object_boxes : (100, 4)
    
    #print 'gt_sub.shape :', gt_sub.shape
    #print 'gt_obj.shape :', gt_obj.shape
    #print 'gt_rel.shape :', gt_rel.shape
    #print 'subject_inds.shape :', subject_inds.shape
    #print 'object_inds.shape :', object_inds.shape
    #print 'predicate_inds.shape :', predicate_inds.shape
    
    cat_rel_cnt = np.zeros(51)
    for rel in gt_rel:
        cat_rel_cnt[rel] += 1
    cat_rel_correct_cnt = np.zeros(51)
    
    done_idx = []
    for gt_id in range(len(gt_rel)): # gt relationship 개수만큼 반복
    
        for candidate_id in range(len(predicate_inds)): # box가 맞는 트리플 수 만큼 반복
            if candidate_id in done_idx: # 이미 카운트된 인덱스는 제외
                #continue
                pass
            if predicate_inds[candidate_id] == gt_rel[gt_id]: # 관계가 맞으면 +1
                cat_rel_correct_cnt[gt_rel[gt_id]] += 1
                done_idx.append(candidate_id)
                break
    return cat_rel_cnt, cat_rel_correct_cnt


def check_att_recall(gt_objects, gt_atts, pred_objects, atts_prob, thres=0.5, TEST_MODE=False):
    overlaps = bbox_overlaps(
        np.ascontiguousarray(pred_objects[:, :], dtype=np.float),
        np.ascontiguousarray(gt_objects[:, :4], dtype=np.float))
    n_class = atts_prob.size(1)
    if TEST_MODE:
        print('check_att_recall')
        print('overlaps_size:', overlaps.size)
        print('len(gt_objects)', len(gt_objects))
        print('len(pred_objects)', len(pred_objects))
        print('len(pred_objects[0])', len(pred_objects[0]))
        print('gt_atts.shape', gt_atts.shape)
        print('len(atts_prob)', len(atts_prob))
        print('len(atts_prob[0])', len(atts_prob[0]))
        print('n_class', n_class)
        print('')

    n_gt_obj = len(gt_objects)
    n_pred_obj = len(pred_objects)
    scores, pred_atts = atts_prob.data.max(1)
    scores, pred_atts = scores.cpu().numpy(), pred_atts.cpu().numpy()

    cnt = np.zeros(n_class)
    for att in gt_atts:
        if TEST_MODE:
            print(type(att))
            print(att)
        cnt[att] += 1
    correct_cnt = np.zeros(n_class)
    if TEST_MODE:
        print('type(pred_atts)', type(pred_atts))
        print('pred_atts', pred_atts)

    for i in range(n_pred_obj):
        for j in range(n_gt_obj):
            if overlaps[i][j] >= thres and pred_atts[i] == gt_atts[j]:
                correct_cnt[pred_atts[i]] += 1
    if TEST_MODE:
        print('gt_objects', gt_objects)
        print('pred_objects', pred_objects)
        print('overlaps', overlaps)
        print('gt_atts', gt_atts)
        print('atts_prob', atts_prob)
        print('cnt', cnt)
        print('correct_cnt', correct_cnt)
    return cnt.sum(), correct_cnt.sum()