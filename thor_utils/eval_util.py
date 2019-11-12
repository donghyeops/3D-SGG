# -*- coding: utf-8 -*-

import numpy as np
import os
import os.path as osp
import json
import copy
import math
from pprint import pprint as pp


def evalutate_gsg_histories_from_dir(pred_gh_dir, gt_gh_dir, iou_threhold=0.5, gt_target=[], partial_gsg=False):
    rooms = os.listdir(pred_gh_dir)  # gsg_pred 폴더 안의 room 이름들

    obj_count_per_metrics = dict()
    obj_sum_per_metrics = dict()

    sgg_count_per_step = {}
    sgg_sum_per_step = dict()
    for room in rooms:
        # print('count_per_step', count_per_step) ##
        # print('sum_per_step', sum_per_step) ##
        json_files = os.listdir(osp.join(pred_gh_dir, room))  # seed 안쓰면 길이가 1임
        for json_file in json_files:
            with open(osp.join(pred_gh_dir, room, json_file), 'r') as f:
                pred_gh_json = json.load(f)
            with open(osp.join(gt_gh_dir, room, json_file), 'r') as f:
                gt_gh_json = json.load(f)
            try:
                obj_scores, sgg_scores = evaluate_gsg_history(pred_gh_json['gsg_history'], gt_gh_json['gsg_history'],
                                                              iou_threhold=iou_threhold, gt_target=gt_target, partial_gsg=partial_gsg)
            except:
                print(room)
                raise Exception()

            for key, obj_score in obj_scores.items():
                if obj_score is None:
                    continue
                if key in obj_count_per_metrics:
                    obj_count_per_metrics[key] += 1
                    obj_sum_per_metrics[key] += obj_score
                else:
                    obj_count_per_metrics[key] = 1
                    obj_sum_per_metrics[key] = obj_score
                # print(key, obj_score) ##

            for i, sgg_score in enumerate(sgg_scores.values()):
                # 스탭(행동카운트) 별 스코어들의 합이랑(sgg_sum_per_step), 개수를 구함(sgg_count_per_step).
                if len(sgg_score.keys()) == 0 or sgg_score['total'] is None:  # 관계 없어서 score 못구하는 경우
                    continue

                for k, v in sgg_score.items():
                    if v is None:
                        continue
                    else:
                        if i not in sgg_count_per_step:
                            sgg_count_per_step[i] = {
                                'attribute': 0,
                                'relationship': 0,
                                'total': 0
                            }
                        sgg_count_per_step[i][k] += 1

                        if i not in sgg_sum_per_step:
                            sgg_sum_per_step[i] = {
                                'attribute': 0.,
                                'relationship': 0.,
                                'total': 0.
                            }
                        sgg_sum_per_step[i][k] += v

    obj_score_per_metrics = dict()
    for key in obj_count_per_metrics.keys():
        # print(obj_sum_per_metrics)
        # print(obj_count_per_metrics)
        obj_score_per_metrics[key] = float(obj_sum_per_metrics[key]) / obj_count_per_metrics[key]

    total_sggen = {
        'attribute': 0,
        'relationship': 0,
        'total': 0,
    }
    for i in range(len(sgg_count_per_step)):
        for k, v in sgg_sum_per_step[i].items():
            total_sggen[k] += v

    for k, v in total_sggen.items():
        n_count = sum([v[k] for v in sgg_count_per_step.values()])

        total_sggen[k] /= n_count

    sgg_score_per_step = dict()
    for i in range(len(sgg_count_per_step)):
        sgg_score_per_step[i] = {}
        for k in total_sggen.keys():
            if sgg_count_per_step[i][k] == 0:
                continue
            sgg_score_per_step[i][k] = float(sgg_sum_per_step[i][k]) / sgg_count_per_step[i][k]

    return obj_score_per_metrics, total_sggen, sgg_score_per_step


def evaluate_gsg_history(pred_gh, gt_gh, last_step=None, iou_threhold=0.5, gt_target=[], partial_gsg=False):
    # last_step: 해당 스텝 수까지만의 성능을 구함
    if len(gt_gh) != len(pred_gh):
        last_step = len(pred_gh) if len(gt_gh) >= len(pred_gh) else len(gt_gh)
    keys = sorted(list(map(int, gt_gh.keys())))  # [1, 2, 3, 4, ...]
    if last_step is not None:
        keys = [key for key in keys if key <= last_step]
    sgg_scores = {key: {} for key in keys}

    obj_scores = dict()
    obj_count = dict()

    # print('len(gt_gh)', len(gt_gh))
    # print('len(pred_gh)', len(pred_gh))
    # print('gt_gh', keys)
    # print('pred_gh', pred_gh.keys())
    for key in keys:
        if str(key) not in pred_gh:
            continue
        pred_gsg = pred_gh[str(key)]
        gt_gsg = gt_gh[str(key)]
        try:
            obj_score, sgg_scores[key] = evalutate_gsg(pred_gsg, gt_gsg, iou_threhold=iou_threhold, gt_target=gt_target,
                                                       partial_gsg=partial_gsg)
        except:
            print(key)
            raise Exception()

        for k, s in obj_score.items():
            if s is None:  # 관계 없어서 score 못구하는 경우
                continue
            if k in obj_scores:
                obj_scores[k] += s
                obj_count[k] += 1
            else:
                obj_scores[k] = s
                obj_count[k] = 1
    for k in obj_scores.keys():
        obj_scores[k] /= obj_count[k]

    return obj_scores, sgg_scores


def evalutate_gsg(pred_gsg, gt_gsg, iou_threhold=0.5, gt_target=[], partial_gsg=False):
    # 입력값: json 포멧 그대로 사용
    pred_objs = pred_gsg['objects']
    gt_objs = gt_gsg['objects']

    pred_rels = pred_gsg['relations']
    gt_rels = gt_gsg['relations']

    correct_obj = 0.

    gt_objs_ = [obj for obj in gt_objs.values() if obj['detection']]  # 둘 다 현재 detection 된 물체여야함
    pred_objs_ = [obj for obj in pred_objs.values() if obj['detection']]  # 둘 다 현재 detection 된 물체여야함
    pred_chekced_idx = []
    for i, gt_obj in enumerate(gt_objs_):
        a = 0
        for j, pred_obj in enumerate(pred_objs_):
            if j in pred_chekced_idx:  # 이미 체크된 물체면 통과
                a = -1
                continue
            if gt_obj['label'] != pred_obj['label']:  # 라벨 같으면 통과
                a = -2
                continue
            iou = get_iou3D(gt_obj['box3d'], pred_obj['box3d'])
            if iou < iou_threhold:  # iou 일정 수준 겹치면 통과
                a = -3
                continue
            pred_chekced_idx.append(j)
            correct_obj += 1
            a = 0
            break
        if a != 0:
            # print(a, gt_obj)
            # print([pred_obj for j, pred_obj in enumerate(pred_objs_) if j not in pred_chekced_idx])
            # print('')
            # raise Exception()
            pass
    obj_scores = dict()
    if not len(gt_objs_):
        obj_scores['recall'] = None
    else:
        obj_scores['recall'] = correct_obj / len(gt_objs_)
    if not len(pred_objs_):
        obj_scores['precision'] = None
    else:
        obj_scores['precision'] = correct_obj / len(pred_objs_)
    obj_scores['mAP'] = get_mAP(gt_objs_, pred_objs_, iou_threhold)
    # print(obj_scores)

    n_att_triple = 0
    n_att_correct_triple = 0
    n_rel_triple = 0
    n_rel_correct_triple = 0
    sgg_scores = {
        'attribute': None,
        'relationship': None,
        'total': None,
    }

    #print(gt_objs)
    #print(pred_objs)

    if partial_gsg:
        gt_objs = {k: v for k, v in gt_objs.items() if v['detection']}
        pred_objs = {k: v for k, v in pred_objs.items() if v['detection']}

        gt_rels = {k: v for k, v in gt_rels.items() if
                   (str(v['subject_id']) in gt_objs and str(v['object_id']) in gt_objs)}
        pred_rels = {k: v for k, v in pred_rels.items() if
                   (str(v['subject_id']) in pred_objs and str(v['object_id']) in pred_objs)}


    # attribute triple
    for i, gt_obj in enumerate(gt_objs.values()):
        a = 0
        target_obj = None
        for j, pred_obj in enumerate(pred_objs.values()):
            if gt_obj['label'] == pred_obj['label']:
                iou = get_iou3D(gt_obj['box3d'], pred_obj['box3d'])
                if iou > iou_threhold:  # iou 일정 수준 겹치면 통과
                    target_obj = pred_obj
                    break

        if gt_obj['open_state'] != 0:
            n_att_triple += 1
            if (target_obj is not None) and ('att' in gt_target or gt_obj['open_state'] == target_obj['open_state']):
                n_att_correct_triple += 1

        if gt_obj['color'] != 0:
            n_att_triple += 1
            if (target_obj is not None) and ('att' in gt_target or gt_obj['color'] == target_obj['color']):
                n_att_correct_triple += 1
    if n_att_triple != 0:
        sgg_scores['attribute'] = n_att_correct_triple / n_att_triple

    # relationship triple
    for gt_rel in gt_rels.values():
        if gt_rel['rel_class'] == 0:
            continue
        n_rel_triple += 1

        pop_key_list = []
        for key, pred_rel in pred_rels.items():
            # 관계 클래스 체크
            if gt_rel['rel_class'] != pred_rel['rel_class']:
                continue

            # 물체들 체크
            for target in ['subject_id', 'object_id']:
                gt_obj = gt_objs[str(gt_rel[target])]
                pred_obj = pred_objs[str(pred_rel[target])]

                # 물체 클래스 체크
                if gt_obj['label'] != pred_obj['label']:
                    break

                # 속성 체크 (예전 버전)
                # if 'att' not in gt_target:
                #    if gt_obj['color'] != pred_obj['color']:
                #        break
                #    if gt_obj['open_state'] != pred_obj['open_state']:
                #        break

                # iou 체크
                iou = get_iou3D(gt_obj['box3d'], pred_obj['box3d'])
                if iou < iou_threhold:
                    break

            else:  # break 발생 안하면 (모든 체크 통과하면)
                n_rel_correct_triple += 1
                pop_key_list.append(key)
        for key in pop_key_list:
            pred_rels.pop(key)  # 정답 판별된 관계는 제거
    if n_rel_triple != 0:
        sgg_scores['relationship'] = n_rel_correct_triple / n_rel_triple

    if sgg_scores['attribute'] is None:
        sgg_scores['total'] = sgg_scores['relationship']
    elif sgg_scores['relationship'] is None:
        sgg_scores['total'] = sgg_scores['attribute']
    else:
        if n_att_triple + n_rel_triple != 0:
            sgg_scores['total'] = (n_att_correct_triple + n_rel_correct_triple) / (n_att_triple + n_rel_triple)
    # print(recall, correct, len(gt_rels))  ## 잘됨
    return obj_scores, sgg_scores


def get_mAP(pred_objs, gt_objs, iou_threhold):
    if not len(pred_objs):
        return None
    precisions = np.zeros(len(pred_objs))
    pred_objs = sorted(pred_objs, key=lambda v: v['score'], reverse=True)
    for i in range(len(pred_objs)):
        target = pred_objs[:i+1]
        precisions[i] = _get_precision(target, copy.deepcopy(gt_objs), iou_threhold)

    return np.average(precisions)


def _get_precision(pred_objs, gt_objs, iou_threhold):
    correct = 0
    for pred_obj in pred_objs:
        for gt_obj in gt_objs:
            if gt_obj['label'] != pred_obj['label']:
                continue
            iou = get_iou3D(gt_obj['box3d'], pred_obj['box3d'])
            if iou < iou_threhold:  # iou 일정 수준 겹치면 통과
                continue
            gt_objs.remove(gt_obj)
            #gt_objs = np.delete(gt_objs, gt_obj)
            correct += 1
            break
    return correct / len(pred_objs)


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
    half = pos[3:] / 2

    return pos[:3] - half, pos[:3] + half


def get_avg_score(scores):
    return float(np.average(list(scores.values())))


if __name__ == '__main__':
    # GSG_history들 전체 평가 (GSG 별로 평가하여 평균냄)
    # gt_target은 해당 부분은 gt로 평가하는걸 의미 (현재 att만 사용 가능)
    #gt_gh_dir = '/home/ailab/DH/ai2thor/datasets/1_paper/static_env/gsg_gt/gsg'
    gt_gh_dir = '/home/ailab/DH/ai2thor/datasets/190514 gsg_gt/gsg'


    def result_processing(obj_scores, total_sggen, sgg_scores, test_target, iou_threhold, test_results):
        print('[{}] obj_precision : {}'.format(test_target, obj_scores['precision']))
        print('[{}] obj_recall : {}'.format(test_target, obj_scores['recall']))
        print('[{}] obj_mAP : {}'.format(test_target, obj_scores['mAP']))
        print('[{}] mSGGen[att] : {}'.format(test_target, total_sggen['attribute']))
        print('[{}] mSGGen[rel] : {}'.format(test_target, total_sggen['relationship']))
        print('[{}] mSGGen[all] : {}'.format(test_target, total_sggen['total']))
        test_results[test_target] = {'iou_threhold': iou_threhold, 'object_scores': obj_scores,
                                     'sgg_score': total_sggen, 'sgg_score_per_step': sgg_scores}
        print('done.\n')


    for iou_threhold in [0.3]:
        print('evaluate GSGs')

        print('gt_gh_dir :', gt_gh_dir)
        print('iou_threhold :', iou_threhold)

        test_results = dict()

        # test_target = 'R'
        # print(f'*** {test_target} test ***')
        # pred_gh_dir = '/home/ailab/DH/ai2thor/datasets/1_paper/static_env/gsg_pred_R/gsg'
        # print('pred_gh_dir :', pred_gh_dir)
        # gt_target = ['att']
        # print('gt_target :', gt_target)
        # obj_scores, total_sggen, sgg_scores = evalutate_gsg_histories_from_dir(pred_gh_dir=pred_gh_dir,
        #                                                                        gt_gh_dir=gt_gh_dir,
        #                                                                        iou_threhold=0.8, gt_target=gt_target)
        # result_processing(obj_scores, total_sggen, sgg_scores, test_target, iou_threhold, test_results)
        #
        # test_target = 'OR'
        # print(f'*** {test_target} test ***')
        # pred_gh_dir = '/home/ailab/DH/ai2thor/datasets/1_paper/static_env/gsg_pred_OAR/gsg'
        # print('pred_gh_dir :', pred_gh_dir)
        # gt_target = ['att']
        # print('gt_target :', gt_target)
        # obj_scores, total_sggen, sgg_scores = evalutate_gsg_histories_from_dir(pred_gh_dir=pred_gh_dir,
        #                                                                        gt_gh_dir=gt_gh_dir,
        #                                                                        iou_threhold=iou_threhold,
        #                                                                        gt_target=gt_target)
        # result_processing(obj_scores, total_sggen, sgg_scores, test_target, iou_threhold, test_results)
        #
        # test_target = 'OAR'
        # print(f'*** {test_target} test ***')
        # pred_gh_dir = '/home/ailab/DH/ai2thor/datasets/1_paper/static_env/gsg_pred_OAR/gsg'
        # print('pred_gh_dir :', pred_gh_dir)
        # gt_target = []
        # print('gt_target :', gt_target)
        # obj_scores, total_sggen, sgg_scores = evalutate_gsg_histories_from_dir(pred_gh_dir=pred_gh_dir, gt_gh_dir=gt_gh_dir,
        #                                                           iou_threhold=iou_threhold, gt_target=gt_target)
        # result_processing(obj_scores, total_sggen, sgg_scores, test_target, iou_threhold, test_results)
        #
        # test_target = 'OATR'
        # print(f'*** {test_target} test ***')
        # pred_gh_dir = '/home/ailab/DH/ai2thor/datasets/1_paper/static_env/gsg_pred_OATR/gsg'
        # print('pred_gh_dir :', pred_gh_dir)
        # gt_target = []
        # print('gt_target :', gt_target)
        # obj_scores, total_sggen, sgg_scores = evalutate_gsg_histories_from_dir(pred_gh_dir=pred_gh_dir, gt_gh_dir=gt_gh_dir,
        #                                                           iou_threhold=iou_threhold, gt_target=gt_target)
        # result_processing(obj_scores, total_sggen, sgg_scores, test_target, iou_threhold, test_results)

        for mode in ['', 'only_PM', 'am_only', 'merge1', 'merge2']:
            test_target = 'OATR'+'_'+mode
            print(f'*** {test_target} test ***')
            pred_gh_dir = '/home/ailab/DH/ai2thor/datasets/190514 gsg_pred_OATR/gsg'+'_'+mode
            print('pred_gh_dir :', pred_gh_dir)
            gt_target = []
            print('gt_target :', gt_target)
            obj_scores, total_sggen, sgg_scores = evalutate_gsg_histories_from_dir(pred_gh_dir=pred_gh_dir, gt_gh_dir=gt_gh_dir,
                                             iou_threhold=iou_threhold, gt_target=gt_target)
            result_processing(obj_scores, total_sggen, sgg_scores, test_target, iou_threhold, test_results)

        with open(f'./outputs_iou.{iou_threhold}.json', 'w') as f:
            json.dump(test_results, f, indent='\t')
        print('save output')
