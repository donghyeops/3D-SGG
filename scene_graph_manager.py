# -*- coding: utf-8 -*-

from math import pow, sqrt
import numpy as np
from thor_utils import annotation_util as au
import copy

class DynamicGlobalSceneGraphManager():
    def __init__(self, thorCtrl, use_dnn, dnn_check_dict, use_history=False):
        self.thorCtrl = thorCtrl
        self.use_dnn = use_dnn
        self.dnn_check_dict = dnn_check_dict

        # 참고.
        # dnn_check_dict = {
        #     'od': self.ui.check_dnn_od.isChecked(),
        #     'att': self.ui.check_dnn_att.isChecked(),
        #     'transfer': self.ui.check_dnn_transfer.isChecked(),
        #     'rel': self.ui.check_dnn_rel.isChecked()
        # }

        if use_dnn:
            from dnn_manager import DNNManager
            self.dnnMgr = DNNManager()
            self.dnnMgr.load_models()

        self.sceneMap = SceneMap()
        self.use_history = use_history  # GSG history 쓸 지 말지 결정
        self.use_gt_box3d = False # transNet 쓸지 안쓸지 결정
        self.iou_threshold = 0.3

        print('* use_gt_box3d:', self.use_gt_box3d)
        print('* iou_threshold:', self.iou_threshold)

    def reset(self):
        self.sceneMap.reset()

    def set_use_dnn(self, use_dnn, dnn_check_dict=None):
        self.use_dnn = use_dnn
        if dnn_check_dict is not None:
            self.dnn_check_dict = dnn_check_dict

        if not hasattr(self, 'dnnMgr'):
            from dnn_manager import DNNManager
            self.dnnMgr = DNNManager()
            self.dnnMgr.load_models()

    def set_use_history(self, use_history):
        self.use_history = use_history

    def get_sceneMap(self):
        return self.sceneMap

    def get_od_results(self):
        return self.sceneMap.get_od_results()

    def get_gsg(self):
        return self.sceneMap.get_gsg()

    def apply_action_model(self, inputs, merge_results=True):
        self.sceneMap.tick_time()  # time +1
        action = inputs['action']
        if 'target_obj' in inputs:
            target_obj_gt = inputs['target_obj']

        if not merge_results:
            self.detect_all()
        elif action in ['go', 'back', 'go_right', 'go_left', 'right', 'left', 'up', 'down']:
            if self.sceneMap.get_agent_object() is not None: # carry
                # 소유 물체 위치 변경
                self.sceneMap.update_agent_object_position(self.thorCtrl.get_event().metadata['agent'])
                owned_obj_id = self.sceneMap.get_agent_object_id() # 소유 물체 id 가져오기
                self.sceneMap.remove_relations_by_object(owned_obj_id) # 관계 리셋 (owned 관계는 따로 명시하기 때문에 리셋 안됨)

            self.detect_all() # carry의 경우도 고려하여 관계 추가 인식하도록 구현
            self.sceneMap.set_before_action({
                'action':action,
                'carry_flag':self.sceneMap.get_agent_object() is not None,
                'success':inputs['success'] if 'success' in inputs else True
            })

        elif action in ['open', 'close']:
            target_obj = self.find_obj_from_sceneMap(target_obj_gt, gt_box=self.use_gt_box3d, iou_threshold=self.iou_threshold)
            self.detect_all()
            tid = target_obj['id'] if target_obj is not None else None
            self.sceneMap.set_before_action({
                'action': action,
                'target_object_id': tid,
                'success':inputs['success'] if 'success' in inputs else True
            })

            # if target_obj is not None: # 확률값으로 합치기 !
            #     self.sceneMap.set_object(target_obj['id'], 'open_state', au.openState_s2i[action]) # detect_all 전에 부르면, 잘못된 인식 결과가 들어갈 수 있음

        elif action == 'pickup':
            # 조건 : pickup된 물체가 현재 sceneMap에 있어야함
            target_obj = self.find_obj_from_sceneMap(target_obj_gt, gt_box=self.use_gt_box3d, iou_threshold=self.iou_threshold)
            before = len(self.sceneMap.od_results['boxes'])
            #print('before', self.sceneMap.od_results['boxes'], before)
            self.detect_all()  # 물체를 들어서, 새로운 물체가 보인다면 인식도 추가해야함 (OD 시각화 갱신을 위해 사용)

            target_obj_id = None
            if target_obj is not None: # 처음에 물체가 있었는지.
                target_obj_id = target_obj['id']
                after = len(self.sceneMap.od_results['boxes'])
                #print('after', self.sceneMap.od_results['boxes'], after)
                #if before != after:
                #self.sceneMap.remove_object(target_obj['id'])
                self.sceneMap.set_agent_object(target_obj['id'])
                self.sceneMap.remove_relations_by_object(target_obj['id'])

            self.sceneMap.set_before_action({
                'action': action,
                'target_object_id': target_obj_id,
                'success':inputs['success'] if 'success' in inputs else True
            })

        elif action == 'put':
            self.detect_all() # 새로운 관계 인식 (추가된 물체의 상태가 잘못인식될 것 같으면, set_object를 이후에 추가해야함)
            added_objects = self.sceneMap.get_new_objects()
            if added_objects is not None and len(added_objects) == 1: # 새로 추가된 물체가 단일이라면, target_obj으로 덤프
                target_obj = self.sceneMap.pop_agent_object()
                if target_obj['label'] == added_objects[0]['label']:
                    self.sceneMap.dump_object_info(target_id=added_objects[0]['id'], input_obj=target_obj,
                                                   without=['box3d', 'detection'])

            support_obj = self.find_obj_from_sceneMap(target_obj_gt, gt_box=self.use_gt_box3d,
                                                     iou_threshold=self.iou_threshold)
            self.sceneMap.set_before_action({
                'action': action,
                'target_object_id': support_obj['id'],
                'success':inputs['success'] if 'success' in inputs else True
            })
        else:
            raise Exception(f'wrong action :{action}')

        self.sceneMap.keep_gsg()
        if self.use_history:
            self.sceneMap.record_gsg()

    def detect_all(self):
        if self.use_dnn:
            self.detect_all_using_dnn(iou_threshold=self.iou_threshold)  # dnn_check_dict에 따라 적용
            # iou_threshold: hit box로 테스트할땐 0.3, gt 적용 시에만 hit box 쓸땐 0.95 사용 ! (transnet 쓸거면 상관없음)
        else:
            self.detect_all_using_gt()

    def detect_all_using_dnn(self, iou_threshold=0.5):
        # dnn_check_dict는 현재 [Object+Att]('od'), [Transfer]('transfer')만 적용함

        frame = self.thorCtrl.get_image()  # RGB
        frame2 = self.thorCtrl.get_image('opencv')  # RGB

        if self.dnn_check_dict['od']:
            boxes, scores, classes = self.dnnMgr.detect_objects(frame)  # ObjNet
        else:
            vis_objs = self.thorCtrl.get_visible_objects()
            boxes = np.zeros((len(vis_objs), 4), np.int)
            classes = np.empty((len(vis_objs)), '<U20')
            scores = np.ones((len(boxes)))

            for i, vis_obj in enumerate(vis_objs):
                boxes[i] = list(map(int, vis_obj['bb'])) if 'bb' in vis_obj else [-1, -1, -1, -1]
                classes[i] = vis_obj['objectType']

        self.sceneMap.set_od_results(boxes=boxes, scores=scores, classes=classes)

        def recognize_owned_object_relationships(new_objects=None):
            ### agent's object의 관계를 구하기 위해.. ####
            # 지금 문제점 : 기존에 발견한 물체만 대상으로 삼음. 새로 찾은 물체도 대상으로 삼아야함
            output_relations = []
            owned_obj_id = self.sceneMap.get_agent_object_id()
            if owned_obj_id is not None:
                exist_objs = self.sceneMap.get_objects_as_input()  # 현재 뷰에는 없지만, 기존에 관측한 물체

                if len(exist_objs) > 0:
                    owned_obj = None
                    # print('owned_obj_id', owned_obj_id)
                    # print('exist_objs#', len(exist_objs))
                    for del_idx, _obj in enumerate(exist_objs):
                        if _obj[-1] == owned_obj_id:
                            owned_obj = copy.deepcopy(_obj)
                            break
                    assert owned_obj is not None, 'There is no owned_obj !!'
                    exist_objs = np.delete(exist_objs, del_idx, 0)  # agent' object를 신상으로.
                    global_coor = owned_obj[np.newaxis, :6]
                    classes_to_number = owned_obj[np.newaxis, 6]

                    total_coor = np.concatenate((global_coor, exist_objs[:, :6]), 0)
                    total_label = np.concatenate((classes_to_number, exist_objs[:, 6]), 0)

                    if new_objects is not None:
                        # print('new_objects#', len(new_objects))
                        # 0: agent's obj, 1~X: other exist_obj, X+1~: new obj
                        new_obj_index = np.arange(len(total_coor), len(new_objects) + len(total_coor), 1)
                        total_coor = np.concatenate((total_coor, new_objects[:, :6]), 0)
                        total_label = np.concatenate((total_label, new_objects[:, 6]), 0)
                    exist_index = np.arange(len(global_coor), len(total_coor), 1)


                    # relations = self.dnnMgr.predict_relationships(global_coor, classes_to_number) # classes_to_number은 당장 안씀
                    # 기존 box들 간의 관계는 안씀
                    # print('[input] total_coor :', total_coor)
                    # print('[input] total_label :', total_label)
                    # print('[input] exist_index :', exist_index)
                    owned_obj_relations = self.dnnMgr.predict_relationships(total_coor,
                                                                            total_label,
                                                                            exist_box_index=exist_index)
                    owned_obj_relations = [rel for rel in owned_obj_relations if
                                           au.rel_i2s[rel['rel_class']] != 'background']  # background 제거

                    # add_scene_info를 위해서, 사용
                    # 기존 물체(exist_objs)를 input으로 안넣기 위해, 두 경우로 나눠서 id 부여. (새 물체 : local_id, 기존 물체 : 해당 물체의 id)
                    # 이거 안하면, pred_obj_info에 exist_objs도 추가해야함(local_id에 맞는 물체가 있어야하므로)
                    for i, rel in enumerate(owned_obj_relations):
                        if int(rel['subject_id']) >= len(global_coor):
                            if (new_objects is not None) and (int(rel['subject_id']) in new_obj_index):
                                owned_obj_relations[i]['subject_id'] = 'n_{}'.format(
                                    int(rel['subject_id']) - int(new_obj_index[0])) # 0부터 시작하니까, 첫 번째 값을 0으로 만듦
                            else:
                                owned_obj_relations[i]['subject_id'] = 'e_{}'.format(
                                    int(exist_objs[rel['subject_id'] - len(global_coor)][-1]))
                        else:
                            owned_obj_relations[i]['subject_id'] = 'e_{}'.format(int(owned_obj_id))

                        if int(rel['object_id']) >= len(global_coor):
                            if (new_objects is not None) and (int(rel['object_id']) in new_obj_index):
                                owned_obj_relations[i]['object_id'] = 'n_{}'.format(
                                    int(rel['object_id']) - int(new_obj_index[0])) # 0부터 시작하니까, 첫 번째 값을 0으로 만듦
                            else:
                                owned_obj_relations[i]['object_id'] = 'e_{}'.format(
                                    int(exist_objs[rel['object_id'] - len(global_coor)][-1]))
                        else:
                            owned_obj_relations[i]['object_id'] = 'e_{}'.format(int(owned_obj_id))
                    output_relations += owned_obj_relations
                    # print('agent obj_relations#', len(output_relations))
                    # print('*'*40)
            return output_relations

        if len(boxes) > 0:
            classes_to_number = [au.obj_s2i[cls] for cls in classes]  # background가 0임
            # 이미 인식한 물체는 속성 또 인식하지 않게 바꾸기 !
            if self.dnn_check_dict['att']:
                colors, open_states, _, open_state_prob = self.dnnMgr.predict_attributes(frame2, boxes,
                                                                                         classes_to_number)  # AttNet
            else:
                hit_objs, keep_idx = self.thorCtrl.find_object_from_od(boxes, classes, threshold=iou_threshold)
                colors = np.zeros(len(boxes))
                open_states = np.zeros(len(boxes))
                for hit_obj_idx, keep in enumerate(keep_idx):
                    hit_obj = hit_objs[hit_obj_idx]
                    colors[keep] = hit_obj['color']
                    if hit_obj['openable']:
                        open_states[keep] = 1 if hit_obj['isopen'] else 2
                    else:
                        open_states[keep] = 0


            classes_to_number = np.array(classes_to_number)

            if not self.dnn_check_dict['transfer']:
                # TransNet 대신 hit box 사용
                hit_objs, keep_idx = self.thorCtrl.find_object_from_od(boxes, classes, threshold=iou_threshold)
                global_coor = np.zeros((len(hit_objs), 6))
                for idx, obj in enumerate(hit_objs):
                    position = obj['position']
                    size = obj['bounds3D']

                    global_coor[idx] = [position['x'], position['y'], position['z']] + \
                                       [size[3] - size[0], size[4] - size[1], size[5] - size[2]]

                classes_to_number = classes_to_number[keep_idx]
                scores = scores[keep_idx]
                colors = colors[keep_idx]
                open_states = open_states[keep_idx]
                open_state_prob = open_state_prob[keep_idx]

                boxes = boxes[keep_idx]
                print('use hit box. [iou_threshold:{}]'.format(iou_threshold))
            else:
                # TransNet 적용
                agent_info = self.thorCtrl.get_event().metadata['agent']
                depth = self.thorCtrl.get_depth_image()
                global_coor = self.dnnMgr.transfer_to_global_map(frame2, depth, boxes, classes_to_number, agent_info)
                print('use transferNet')

            pred_obj_info = np.concatenate((global_coor, np.expand_dims(classes_to_number, 1),
                                            np.expand_dims(scores, 1), np.expand_dims(colors, 1),
                                            np.expand_dims(open_states, 1), np.ones((len(global_coor), 1)) * -1, boxes), -1)


            #self.sceneMap.get_gsg()['objects']

            if len(global_coor) > 0 and self.dnn_check_dict['rel']:
                exist_objs = self.sceneMap.get_objects_as_input()  # 현재 뷰에는 없지만, 기존에 관측한 물체

                total_coor = np.concatenate((global_coor, exist_objs[:, :6]), 0)
                total_label = np.concatenate((classes_to_number, exist_objs[:, 6]), 0)
                exist_index = np.arange(len(global_coor), len(total_coor), 1)

                # relations = self.dnnMgr.predict_relationships(global_coor, classes_to_number) # classes_to_number은 당장 안씀
                # 기존 box들 간의 관계는 안씀
                relations = self.dnnMgr.predict_relationships(total_coor,
                                                              total_label,
                                                              exist_box_index=exist_index)
                relations = [rel for rel in relations if au.rel_i2s[rel['rel_class']] != 'background']  # background 제거

                # add_scene_info를 위해서, 사용
                # 기존 물체(exist_objs)를 input으로 안넣기 위해, 두 경우로 나눠서 id 부여. (새 물체 : local_id, 기존 물체 : 해당 물체의 id)
                # 이거 안하면, pred_obj_info에 exist_objs도 추가해야함(local_id에 맞는 물체가 있어야하므로)
                for i, rel in enumerate(relations):
                    if int(rel['subject_id']) >= len(global_coor):
                        relations[i]['subject_id'] = 'e_{}'.format(
                            int(exist_objs[rel['subject_id'] - len(global_coor)][-1]))
                    else:
                        relations[i]['subject_id'] = 'n_{}'.format(int(rel['subject_id']))
                    if int(rel['object_id']) >= len(global_coor):
                        relations[i]['object_id'] = 'e_{}'.format(
                            int(exist_objs[rel['object_id'] - len(global_coor)][-1]))
                    else:
                        relations[i]['object_id'] = 'n_{}'.format(int(rel['object_id']))

                new_obj_info = np.concatenate((global_coor, classes_to_number[:, np.newaxis]), 1)
                relations += recognize_owned_object_relationships(new_objects=new_obj_info)
            else:
                relations = []
                relations += recognize_owned_object_relationships()

            agent_info = self.thorCtrl.get_event().metadata['agent']

            self.sceneMap.add_scene_info(
                objects=pred_obj_info,
                relations=relations,
                agent_pos=agent_info,
                gt_box=False,
                iou_threshold=0.2,
                open_state_prob=open_state_prob
            )
        else:
            relations = []
            if self.dnn_check_dict['rel']:
                relations += recognize_owned_object_relationships() # 새로 발견한 물체는 없으니, carry 행동만 체크


            agent_info = self.thorCtrl.get_event().metadata['agent']
            self.sceneMap.add_scene_info(
                objects=[],
                relations=relations,
                agent_pos=agent_info,
                gt_box=False,
                iou_threshold=0.2
            )

    def detect_all_using_gt(self):
        # 미적용 시, full annotation (후에, none / dnn / ann 식으로 라디오박스로 만들어야할듯)
        vis_objs = self.thorCtrl.get_visible_objects()
        boxes = np.zeros((len(vis_objs), 4), np.int)
        classes = np.empty((len(vis_objs)), '<U20')
        classes_to_number = np.zeros((len(vis_objs)), np.int)
        scores = np.ones((len(boxes)))
        colors = np.zeros((len(vis_objs)), np.int)
        open_states = np.zeros((len(vis_objs)), np.int)
        global_coor = np.zeros((len(boxes), 6))

        obj_ids = self.thorCtrl.object_global_ids

        attributes = au.get_attributes(objects=vis_objs, objects_id=obj_ids,
                                       image=self.thorCtrl.get_image('opencv'),
                                       instance_masks=self.thorCtrl.get_event().instance_masks)  # BGR 영상을 줌

        for i, vis_obj in enumerate(vis_objs):
            boxes[i] = list(map(int, vis_obj['bb'])) if 'bb' in vis_obj else [-1, -1, -1, -1]
            classes[i] = vis_obj['objectType']
            classes_to_number[i] = au.obj_s2i[vis_obj['objectType']]

            if vis_obj['openable']:
                open_states[i] = 1 if vis_obj['isopen'] else 2
            else:
                open_states[i] = 0
            colors[i] = attributes[obj_ids[vis_obj['objectId']]]['color']

            size = vis_obj['bounds3D']
            global_coor[i] = [vis_obj['position']['x'], vis_obj['position']['y'], vis_obj['position']['z']] + \
                               [size[3] - size[0], size[4] - size[1], size[5] - size[2]]

        self.sceneMap.set_od_results(boxes=boxes, scores=scores, classes=classes) #

        # boxes = np.array([obj['bounding_box'] for obj in scene_info['visible_object']])
        # classes = np.array([au.obj_i2s[obj['obj_class']] for obj in scene_info['visible_object']])
        # scores = np.ones((len(boxes)))

        def recognize_owned_object_relationships(new_gt_objects=None):
            ### agent's object의 관계를 구하기 위해.. ####
            # 지금 문제점 : 기존에 발견한 물체만 대상으로 삼음. 새로 찾은 물체도 대상으로 삼아야함
            output_relations = []
            owned_obj_id = self.sceneMap.get_agent_object_id()
            if owned_obj_id is not None:
                exist_objs = self.sceneMap.get_objects_as_input()  # 현재 뷰에는 없지만, 기존에 관측한 물체

                if len(exist_objs) > 0:
                    owned_obj = None
                    # print('owned_obj_id', owned_obj_id)
                    # print('exist_objs#', len(exist_objs))
                    for del_idx, _obj in enumerate(exist_objs):
                        if _obj[-1] == owned_obj_id:
                            owned_obj = copy.deepcopy(_obj)
                            break
                    assert owned_obj is not None, 'There is no owned_obj !!'
                    # print('before_obj_id:', [obj[-1] for obj in exist_objs])
                    # print('del_idx:', del_idx)
                    exist_objs = np.delete(exist_objs, del_idx, 0)  # agent' object를 신상으로.

                    inventory_obj = self.thorCtrl.get_inventory()
                    if inventory_obj is None:
                        return output_relations
                    agent_position = self.thorCtrl.get_event().metadata['agent']['position']
                    current_gt_objs = [{
                        'objectId':inventory_obj['objectId'],
                        'objectType':inventory_obj['objectType'],
                        'position':agent_position,
                        'receptacleObjectIds':[],
                        'parentReceptacle':'__'
                    }]

                    exist_gt_objs = self.thorCtrl.get_object_from_3d(exist_objs[:, :6])
                    total_gt_objs = current_gt_objs + exist_gt_objs # 순서대로 쌓음
                    # print('objectId: ', [obj['objectId'] for obj in total_gt_objs])
                    # print('after_exist_obj_id:', [obj[-1] for obj in exist_objs])

                    if new_gt_objects is not None:
                        # print('new_objects#', len(new_gt_objects))
                        # 0: agent's obj, 1~X: other exist_obj, X+1~: new obj
                        new_obj_index = np.arange(len(total_gt_objs), len(new_gt_objects) + len(total_gt_objs), 1)

                        total_gt_objs += new_gt_objects

                    exist_index = np.arange(len(exist_gt_objs), len(total_gt_objs), 1)

                    gt_relations = au.get_global_relations(objects=total_gt_objs, objects_id=None)

                    owned_obj_relations = [rel for rel in gt_relations if
                                           au.rel_i2s[rel['rel_class']] != 'background']  # background 제거

                    # add_scene_info를 위해서, 사용
                    # 기존 물체(exist_objs)를 input으로 안넣기 위해, 두 경우로 나눠서 id 부여. (새 물체 : local_id, 기존 물체 : 해당 물체의 id)
                    # 이거 안하면, pred_obj_info에 exist_objs도 추가해야함(local_id에 맞는 물체가 있어야하므로)
                    for i, rel in enumerate(owned_obj_relations):
                        if int(rel['subject_id']) >= len(current_gt_objs):
                            if (new_gt_objects is not None) and (int(rel['subject_id']) in new_obj_index):
                                owned_obj_relations[i]['subject_id'] = 'n_{}'.format(
                                    int(rel['subject_id']) - int(new_obj_index[0])) # 0부터 시작하니까, 첫 번째 값을 0으로 만듦
                            else:
                                owned_obj_relations[i]['subject_id'] = 'e_{}'.format(
                                    int(exist_objs[rel['subject_id'] - len(current_gt_objs)][-1]))
                        else:
                            owned_obj_relations[i]['subject_id'] = 'e_{}'.format(int(owned_obj_id))

                        if int(rel['object_id']) >= len(current_gt_objs):
                            if (new_gt_objects is not None) and (int(rel['object_id']) in new_obj_index):
                                owned_obj_relations[i]['object_id'] = 'n_{}'.format(
                                    int(rel['object_id']) - int(new_obj_index[0])) # 0부터 시작하니까, 첫 번째 값을 0으로 만듦
                            else:
                                owned_obj_relations[i]['object_id'] = 'e_{}'.format(
                                    int(exist_objs[rel['object_id'] - len(current_gt_objs)][-1]))
                        else:
                            owned_obj_relations[i]['object_id'] = 'e_{}'.format(int(owned_obj_id))
                    output_relations += owned_obj_relations
                    # print('agent obj_relations#', len(output_relations))
                    # print('output_relations :', output_relations)
                    # print('*'*40)
            return output_relations

        if len(boxes) > 0:
            pred_obj_info = np.concatenate((global_coor, np.expand_dims(classes_to_number, 1),
                                            np.expand_dims(scores, 1), np.expand_dims(colors, 1),
                                            np.expand_dims(open_states, 1), np.ones((len(global_coor), 1)) * -1, boxes), -1)

            exist_objs = self.sceneMap.get_objects_as_input()  # 현재 뷰에는 없지만, 기존에 관측한 물체

            current_gt_objs = vis_objs
            exist_gt_objs = self.thorCtrl.get_object_from_3d(exist_objs[:, :6])

            # 에이전트 소유 물체를 sceneMap object안에서 안지우는 걸로 변경해서, pickup후에 gt가 하나 모자람
            #assert len(exist_objs) == len(exist_gt_objs), f'fail get_object_from_3d(): {len(exist_objs)}, {len(exist_gt_objs)}'
            total_gt_objs = current_gt_objs + exist_gt_objs # 순서대로 쌓음

            gt_relations = au.get_global_relations(objects=total_gt_objs, objects_id=None)
            relations = []
            for idx, rel in enumerate(gt_relations):
                # background 제거
                if au.rel_i2s[rel['rel_class']] == 'background':
                    continue

                # 기존 물체 간의 관계는 제거
                if rel['subject_id'] >= len(current_gt_objs) and rel['object_id'] >= len(current_gt_objs):
                    continue

                # 기존 물체(exist_objs)를 input으로 안넣기 위해, 두 경우로 나눠서 id 부여. (새 물체 : local_id, 기존 물체 : 해당 물체의 id)
                new_rel = {}
                if rel['subject_id'] >= len(current_gt_objs):
                    new_rel['subject_id'] = f'e_{int(exist_objs[rel["subject_id"]-len(current_gt_objs)][-1])}'
                else:
                    new_rel['subject_id'] = f'n_{int(rel["subject_id"])}'
                if rel['object_id'] >= len(current_gt_objs):
                    new_rel['object_id'] = f'e_{int(exist_objs[rel["object_id"]-len(current_gt_objs)][-1])}'
                else:
                    new_rel['object_id'] = f'n_{int(rel["object_id"])}'
                new_rel['rel_class'] = rel['rel_class']

                relations.append(new_rel)

            agent_info = self.thorCtrl.get_event().metadata['agent']

            relations += recognize_owned_object_relationships(new_gt_objects=current_gt_objs)

            self.sceneMap.add_scene_info(
                objects=pred_obj_info,
                relations=relations,
                agent_pos=agent_info
            )
        else:
            agent_info = self.thorCtrl.get_event().metadata['agent']

            relations = recognize_owned_object_relationships()

            self.sceneMap.add_scene_info(
                objects=[],
                relations=relations,
                agent_pos=agent_info
            )

    def find_obj_from_sceneMap(self, obj, gt_box=True, iou_threshold=0.5):
        # obj는 gt 형태여야함.
        bounds3D = obj['bounds3D']  # (xmin, ymin, zmin, xmax, ymax, zmax)
        box3d = list(obj['position'].values()) + \
                [bounds3D[3] - bounds3D[0], bounds3D[4] - bounds3D[1], bounds3D[5] - bounds3D[2]]

        target_obj = self.sceneMap.find_object(box3d=box3d, gt_box=gt_box, iou_threshold=iou_threshold)
        #assert target_obj is not None, 'fail to find target object'
        if target_obj is None:
            print(f'fail to find target object, time:{self.sceneMap.time}')
        return target_obj

class SceneMap():
    def __init__(self):
        '''
        objects : {id: {
                          'id':int,
                          'box3d':[x, y, z, w, h, d],
                          'label':int,
                          'score':float,
                          'color':int,
                          'open_state':int,
                          'detection':bool # 현재 time에서 에이전트가 인식한 물체인지 아닌지.(새로운 물체인지는 상관X)
                          }
                      }
        relations : {'key': {
                          'key':str('sid/oid'),
                          'subject_id':int,
                          'object_id':int,
                          'rel_class':int
                          }
                      }
        agent : {
                  'position':{'x':float, 'y':float, 'z':float},
                  'rotation':{'x':float, 'y':float, 'z':float},
                  'cameraHorizon':int (60, 30, 0, -30)
                  'object_id':int
                  }
        gsg : {
                  'time':int
                  'objects':objects,
                  'relations':relations,
                  'agent':agent
                }
        od_results : {'boxes', 'classes', 'scores'} # od 시각화할때만 사용됨 (GSG에 안들어감, 휘발성)
        gsg_history : {'time': gsg}
        previous_gsg : gsg
        '''
        self.objects = {}
        self.last_oid = -1 # object의 마지막 id를 의미. -1은 아무것도 없는 상태(초기상태)
        self.relations = {}
        self.agent = {} # agent의 id는 999. [relation]에서 사용됨
        self.time = 0
        self.gsg_history = {} # GSG를 gsg_history에 씀 (원래는 외부에서 저장하는데, 자체 테스트를 위해 사용)
        self.od_results = {} # od 시각화에만 사용 (GSG에 안들어감)
        self.previous_gsg = None # action 직전의 결과를 저장해둠
        self.before_action = {}

    def reset(self):
        # 기존 정보들을 모두 초기화 (gsg_history 포함)
        self.__init__()

    def set_before_action(self, action):
        self.before_action = action

    def add_scene_info(self, objects, relations, agent_pos=None, gt_box=False, iou_threshold=0.5, renew_exist=False, **kwargs):
        # objects : [(x, y, z, w, h, d, label, score, color, open_state, id)]
        # relations : [{'subject_id', 'object_id', 'rel_class'}]
        # agent_info : .metadata['agent']
        # gt_box : 3d 위치 정보를 GT로 사용하는 지 여부
        #       True => ==로 위치 비교
        #       False => 3D iou로 위치 비교
        # iou_threshold : 3D iou로 비교 시, 임계값(역치)으로 사용됨
        # retain_exist : 물체가 중복될 때, 기존 물체의 라벨/스코어/색/개폐를 새 정보로 덮을 지 결정
        assert isinstance(objects, np.ndarray) or isinstance(objects, list), f'[objects] only np.ndarray or list : {type(objects)}'
        assert isinstance(relations, list) or isinstance(relations, np.ndarray), f'[relations] only np.ndarray or list: {type(relations)}'
        if 'open_state_prob' in kwargs:
            open_state_prob = kwargs['open_state_prob']
            self.objId_to_osProb = {}

        localIdx_to_id = {} # 입력된 obj index가 매핑된 id를 저장 (relations을 위해 사용)
        for local_idx, input_obj in enumerate(objects):
            #assert len(input_obj) == 11, f'need 11 values, but got {len(input_obj)} values'
            for exist_obj in self.objects.values():
                # 기존 물체와의 중복 검사
                if gt_box:
                    condition = (input_obj[:6] == exist_obj['box3d']) # GT 좌표를 쓰기에 가능 (속도가 더 빠름)
                else:
                    condition = (self._get_iou3D(input_obj[:6], exist_obj['box3d']) > iou_threshold)  # 3D IoU 비교

                if condition: # 중복일 때, [현재는 위치만으로 비교. label 비교는 안함]
                    localIdx_to_id[local_idx] = exist_obj['id'] # relation은 기존 물체에 추가
                    # 이 아래 과정은 새 정보를 기존 정보에 덮어씀. (DNN의 결과는 계속 달라질 수 있기 때문)
                    if renew_exist:
                        self.objects[exist_obj['id']]['label'] = input_obj[6]
                        self.objects[exist_obj['id']]['score'] = input_obj[7]
                    self.objects[exist_obj['id']]['color'] = input_obj[8]  # if문 안에 넣으면 처음 인식한 값으로 픽스
                    if au.obj_i2s[input_obj[6]] not in ['Microwave', 'Fridge']:
                        self.objects[exist_obj['id']]['open_state'] = 0
                    else:
                        self.objects[exist_obj['id']]['open_state'] = input_obj[9] #
                    if len(input_obj) == 15: # box2d가 있는 경우
                        self.objects[exist_obj['id']]['box2d'] = input_obj[11:15]
                    self.objects[exist_obj['id']]['detection'] = True
                    break # 중복된 물체이기 때문에, 배제함 (배제안할거면 continue)
            else: # 새로 발견된 물체일 때 (기존 물체들과의 중복 검사를 통과했을 때)
                new_obj = {}
                if input_obj[10] == -1:
                    new_obj['id'] = self.last_oid + 1
                    self.last_oid += 1
                else:
                    new_obj['id'] = input_obj[10]
                new_obj['box3d'] = input_obj[:6]
                new_obj['label'] = input_obj[6]
                new_obj['score'] = input_obj[7]
                new_obj['color'] = input_obj[8]
                if au.obj_i2s[new_obj['label']] not in ['Microwave', 'Fridge']:
                    new_obj['open_state'] = 0
                else:
                    new_obj['open_state'] = input_obj[9]
                if len(input_obj) == 15:
                    new_obj['box2d'] = input_obj[11:15]
                new_obj['detection'] = True
                assert new_obj['id'] not in self.objects, 'already exist id'
                self.objects[new_obj['id']] = new_obj # 새 물체 추가
                localIdx_to_id[local_idx] = new_obj['id']
                if 'open_state_prob' in kwargs:
                    self.objId_to_osProb[new_obj['id']] = open_state_prob[local_idx]

        if agent_pos: # agent의 새 위치/방향 정보가 들어왔을 때만.
            if 'position' in self.agent and self.agent['position'] != agent_pos['position']: # 위치가 변경됬을 때
                self.remove_relations_by_object(999)  # agent와의 기존 관계 모두 제거 # 현재 agent 관계를 추가시킬수없음 (로컬관점 문제 때문에..)
            self.agent['position'] = agent_pos['position']
            self.agent['rotation'] = agent_pos['rotation']
            self.agent['cameraHorizon'] = round(agent_pos['cameraHorizon']) # 60, 30, 0(base), -30 [아래에서 위로 각도]
        #print('localIdx_to_id', localIdx_to_id)
        #print('input', relations)
        for input_rel in relations:
            # 중복 검출은 안하고 그냥 다 덮어씀 (중복 검출하는거랑 안하는거랑 연산량이 비슷)
            # 만약 중복 검출 한다면, key 생성하고 relation이 똑같은지 비교해야함
            new_rel = {}

            id_seg = input_rel['subject_id'].split('_')
            if id_seg[0] == 'e': # 이미 있던 물체 (objects에는 없음)
                new_rel['subject_id'] = int(id_seg[1])
            elif id_seg[0] == 'n': # 새 물체 (objects에 있음)
                new_rel['subject_id'] = localIdx_to_id[int(id_seg[1])]
            else:
                raise Exception('wrong subject_id format')
            id_seg = input_rel['object_id'].split('_')
            if id_seg[0] == 'e': # 이미 있던 물체 (objects에는 없음)
                new_rel['object_id'] = int(id_seg[1])
            elif id_seg[0] == 'n': # 새 물체 (objects에 있음)
                new_rel['object_id'] = localIdx_to_id[int(id_seg[1])]
            else:
                raise Exception('wrong subject_id format')
            if new_rel['subject_id'] == new_rel['object_id']:
                continue
            new_rel['rel_class'] = input_rel['rel_class']
            new_rel['key'] = '/'.join(map(str, [new_rel['subject_id'], new_rel['object_id']]))
            self.relations[new_rel['key']] = new_rel

        # time 갱신은 외부에서 행동 시 마다 부르도록 설계 (왜냐하면 action에 따라, 이 함수를 안부를 수도 있기 때문)

    def add_object(self, obj):
        # sceneMap 형식의 완성된 object만 받음
        assert 'id' in obj and 'box3d' in obj and 'label' in obj, 'wrong object format'
        assert obj['id'] not in self.objects, 'id duplicate'
        self.objects[obj['id']] = obj

    def set_object(self, obj_id, type, value):
        # 특정 object의 값 수정
        self.objects[obj_id][type] = value

    def set_relation(self, sub_id, obj_id, value):
        # 특정 relation의 값 수정
        self.relations[f'{sub_id}/{obj_id}'] = value

    def set_agent_object(self, object_id):
        # agent의 인벤토리 물체 수정/제거
        if object_id is not None:
            object = self.objects[object_id]
            # 아래 수정값들은 self.objects에서도 바뀜(문법 문제 없음)
            object['detection'] = False
            object['box2d'] = [0, 0, 0, 0] # 별 의미 없는데, 그냥 box2d는 개념적으로 사라지므로, 0으로 세팅
            object['box3d'][:3] = list(self.agent['position'].values()) # agent의 위치로 바꿈

        self.agent['object_id'] = object_id # None이면 물체 없는 상태로 변경됨

    def update_agent_object_position(self, agent_info):
        if ('object_id' not in self.agent) or self.agent['object_id'] is None:
            assert False, 'Wrong method call (update_agent_object)'
        agent_pos = [agent_info['position']['x'], agent_info['position']['y'],
                     agent_info['position']['z']]
        object = self.objects[self.agent['object_id']]
        object['box3d'][:3] = agent_pos

    def get_agent_object(self):
        # agent의 인벤토리 물체 조회
        if 'object_id' in self.agent:
            if self.agent['object_id'] is None:
                return None
            return self.objects[self.agent['object_id']]
        else:
            return None

    def get_agent_object_id(self):
        if ('object_id' not in self.agent) or self.agent['object_id'] is None:
            return None
        else:
            return self.agent['object_id']

    def pop_agent_object(self):
        # 인벤토리 물체 반환 (제거 및 return)
        object = self.get_agent_object() # 조회
        self.set_agent_object(None) # 제거
        return object

    def remove_object(self, obj_id):
        # 특정 object 정보(물체,관계) 제거
        self.objects.pop(obj_id) # 물체 제거
        self.remove_relations_by_object(obj_id) # 물체에 해당하는 관계 제거

    def remove_relations_by_object(self, obj_id):
        # 특정 object가 주격/보조 물체로 사용되는 관계를 모두 제거 (물체 제거 시 사용)
        candidates = [k for k in self.relations.keys() if str(obj_id) in k.split('/')]
        for key in candidates:
            self.relations.pop(key)

    def find_object(self, box3d=None, obj_id=None, label=None, gt_box=True, iou_threshold=0.5):
        # 물체 id 또는 3D bbox로 물체 찾아서 반환
        if obj_id is not None:
            return self.objects[obj_id]
        elif box3d is not None:
            for obj in self.objects.values():
                if gt_box:
                    if (obj['box3d'] == box3d).all():
                        if label is not None:
                            if obj['label'] == label:
                                return obj
                        else:
                            return obj
                else:
                    if self._get_iou3D(obj['box3d'], box3d) >= iou_threshold:
                        if label is not None:
                            if obj['label'] == label:
                                return obj
                        else:
                            return obj
        else:
            raise Exception('input box3d or obj_id')
        return None

    def get_new_objects(self):
        # 이번 action 결과로 새로 추가된 물체를 찾음
        if self.previous_gsg is not None:
            pre_obj_ids = {obj['id'] for obj in self.previous_gsg['objects'].values() if obj['detection']}
        else:
            return None
        cand_obj_ids = {obj['id'] for obj in self.objects.values() if obj['detection']}
        new_ids = cand_obj_ids - pre_obj_ids
        new_objs = [self.objects[id] for id in new_ids]

        return new_objs

    def dump_object_info(self, target_id, input_obj, without=['id']):
        # without에 해당하는 정보 말고, 나머지 정보들을 덮어씌움
        for k in self.objects[target_id].keys():
            if k in without:
                continue
            self.objects[target_id][k] = input_obj[k]

        if 'id' not in without:
            # id 수정 시, 기존 object의 키, 관련 relation들을 수정해야함.
            obj = self.objects.pop(target_id) # 기존 id 제거
            self.objects[obj['id']] = obj # 새 id로 물체 추가

            rel_keys = list(self.relations.keys()) # key들을 바꿀꺼라 미리 본뜸
            for k in rel_keys:
                s, o = k.split('/')
                if str(int(target_id)) == s: # subject가 타켓 물체일 때
                    self.relations[k]['subject_id'] = obj['id'] # obj id 변경
                    rel_temp = self.relations.pop(k) # rel 키 제거
                    self.relations[f'{int(obj["id"])}/{o}'] = rel_temp # 새 키로 rel 추가
                elif str(int(target_id)) == o: # object가 타켓 물체일 때
                    self.relations[k]['object_id'] = obj['id']
                    rel_temp = self.relations.pop(k)
                    self.relations[f'{s}/{int(obj["id"])}'] = rel_temp

    def get_objects_as_input(self):
        # 모든 물체를 반환하되, add_scene_info에서 input objects 형태로 반환
        objects_as_input = np.zeros((len(self.objects), 11))
        for idx, obj in enumerate(self.objects.values()):
            objects_as_input[idx, :6] = obj['box3d']
            objects_as_input[idx, 6] = obj['label']
            objects_as_input[idx, 7] = obj['score']
            objects_as_input[idx, 8] = obj['color']
            objects_as_input[idx, 9] = obj['open_state']
            objects_as_input[idx, 10] = obj['id']

        return objects_as_input

    def tick_time(self):
        self.time += 1
        # detection update (한 텀 지나갔으므로 모두 False로 변경)
        for key in self.objects.keys():
            self.objects[key]['detection'] = False

    def keep_gsg(self):
        # 현재 gsg 일시저장 (다음 행동 모델에 사용하기위해)
        self.previous_gsg = copy.deepcopy(self.get_gsg())

    def get_gsg(self):
        gsg = {}
        gsg['time'] = self.time
        gsg['objects'] = self.objects
        gsg['relations'] = self.relations
        gsg['agent'] = self.agent
        gsg['before_action'] = self.before_action
        if hasattr(self, 'objId_to_osProb'):
            gsg['objId_to_osProb'] = self.objId_to_osProb
        return gsg

    def record_gsg(self, use_numpy=False):
        # GSG를 gsg_history에 씀 (원랜 필요없는데 자체 테스트를 위해 사용)
        gsg = copy.deepcopy(self.get_gsg())

        if not use_numpy:
            for key, obj in gsg['objects'].items():
                gsg['objects'][key]['box3d'] = obj['box3d'].tolist()
                gsg['objects'][key]['label'] = int(obj['label'])
                gsg['objects'][key]['score'] = float(obj['score'])
                gsg['objects'][key]['color'] = int(obj['color'])
                gsg['objects'][key]['open_state'] = int(obj['open_state'])
                if 'box2d' in obj:
                    gsg['objects'][key]['box2d'] = list(obj['box2d'])

            for key, rel in gsg['relations'].items():
                gsg['relations'][key]['subject_id'] = int(rel['subject_id'])
                gsg['relations'][key]['object_id'] = int(rel['object_id'])
                gsg['relations'][key]['rel_class'] = int(rel['rel_class'])
            if 'objId_to_osProb' in gsg:
                for key, prob in gsg['objId_to_osProb'].items():
                    gsg['objId_to_osProb'][key] = prob.tolist()
        #self.print_dict(gsg)
        #assert False, 'end'
        self.gsg_history[gsg['time']] = gsg

    def print_dict(self, data, key=''):
        if isinstance(data, dict):
            for k, v in data.items():
                self.print_dict(v, key=k)
        elif isinstance(data, list):
            for v in data:
                #print(f'K:{key}, T:{type(data)}, V:...')
                self.print_dict(v)
        else:
            if not (data is None \
                or isinstance(data, int) \
                or isinstance(data, float) \
                or isinstance(data, str)):
                print(f'K:{key}, T:{type(data)}, V:{data}')

    def get_gsg_history(self):
        return self.gsg_history

    def get_gsg_history_only_now(self):
        return {self.time:copy.deepcopy(self.get_gsg())}

    def set_od_results(self, boxes, scores, classes):
        self.od_results['boxes'] = boxes
        self.od_results['scores'] = scores
        self.od_results['classes'] = classes

    def get_od_results(self):
        return self.od_results

    def get_time(self):
        return self.time

    def _get_iou3D(self, pos1, pos2):
        # pos: [x, y, z, w_x, h, w_z]
        pos1 = np.array([float(e) for e in pos1])
        pos2 = np.array([float(e) for e in pos2])

        pos1_min, pos1_max = self._get_min_max_3dPos(pos1)
        pos2_min, pos2_max = self._get_min_max_3dPos(pos2)
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

    def _get_min_max_3dPos(self, pos):
        half = pos[3:]/2

        return pos[:3]-half, pos[:3]+half
