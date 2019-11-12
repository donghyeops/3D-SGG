# -*- coding:utf-8 -*-
# python 3.6 version
import ai2thor.controller
import os
import numpy as np
# set distance to 4.0 meters
# print(os.environ['AI2THOR_VISIBILITY_DISTANCE'])
os.environ['AI2THOR_VISIBILITY_DISTANCE'] = '10.0'


# start ai2thor here

class ThorCtrl:
    def __init__(self, positive_objects=None, smooth_movement=False, magnitude=0.05):
        self.CT = ai2thor.controller.Controller()
        self.scene_list = self.CT.scene_names()
        self.scene_name = self.scene_list[0]
        self.power = False
        self.object_file_open = False
        self.action_file_open = False
        self.image_dir_open = False
        self.is_recorded = False
        self.order_num = 0 # 현재는 레코더 스탭으로 사용
        self.step_count = 0 # 액션 시 마다 +1
        self.object_global_ids = {}  # 씬이 시작할 때, 씬 안의 object들의 id를 0번부터 정의해둠
        self.positive_objects = positive_objects
        self.smooth_movement = smooth_movement
        self.magnitude = magnitude

        self.bbox_to_visible = True # False: (default) 'visible'로 vis_objs 선정,
                                    # True: 'instance_detections2D'로 vis_objs 선정
                                    # 냉장고는 AI2THOR_VISIBILITY_DISTANCE 늘여도, visible하지 않아서 추가한 기능.
        self.PRINT_FAIL_LOG = False
    def is_powered(self):
        return self.power

    # init part (btn_start)
    def power_on(self, w=1100, h=700):
        if not self.is_powered():
            self.CT.start(player_screen_width=w, player_screen_height=h)
            self.power = True

    # init part (btn_start)
    def start(self, scene_name='FloorPlan1', gridSize=0.25, renderObjectImage=True, renderDepthImage=True):
        if not self.is_powered():
            raise Exception('안켜짐')
        self.CT.reset(scene_name)
        self.scene_name = scene_name
        self.renderObjectImage = renderObjectImage
        self.renderDepthImage = renderDepthImage
        self.event = self.CT.step(dict(action='Initialize', gridSize=gridSize, renderObjectImage=renderObjectImage, renderDepthImage=renderDepthImage))
        self.parse_objects_id() # object들의 global id 새 정의
        self.postProcessing()

    # init part (btn_start)
    def power_off(self):
        if self.is_powered():
            self.CT.stop()  # 내부의 server thread가 안꺼져서 강제로 꺼야하는데, 구현 못함
            # self.CT.server_thread._Thread_stop() # 어떻게 끄지...
            self.CT.server_thread = None  # 일단 기존 thread 무시하는 방향으로 ..
            self.power = False

    # 씬 초기화
    def reset(self, seed=None):
        self.CT.reset(self.scene_name)
        print('scene name : ' + self.scene_name)
        self.event = self.CT.step(dict(action='Initialize', gridSize=0.25, renderObjectImage=self.renderObjectImage, renderDepthImage=self.renderDepthImage))
        if seed is not None:
            self.event = self.CT.random_initialize(seed)
        self.parse_objects_id()  # object들의 global id 새 정의
        self.postProcessing()

    # 씬 변경
    def set_scene(self, scene_name='FloorPlan28', seed=None):
        if not scene_name in self.scene_list:
            print('wrong scene name !')
            return
        self.scene_name = scene_name
        self.CT.reset(self.scene_name)
        if seed is not None:
            self.event = self.CT.random_initialize(seed)
        print('scene name : ' + self.scene_name)
        if seed is not None:
            print('seed :', seed)
        self.event = self.CT.step(dict(action='Initialize', gridSize=0.25, renderObjectImage=self.renderObjectImage, renderDepthImage=self.renderDepthImage))
        self.parse_objects_id()  # object들의 global id 새 정의
        self.postProcessing()

    # 다음 씬으로 변경
    def next_scene(self):
        if self.scene_name is None:
            self.scene_name = self.scene_list[0]
        else:
            self.scene_name = self.scene_list[(self.scene_list.index(self.scene_name) + 1) % len(self.scene_list)]
        self.CT.reset(self.scene_name)
        print('scene name : ' + self.scene_name)
        self.event = self.CT.step(dict(action='Initialize', gridSize=0.25, renderObjectImage=self.renderObjectImage, renderDepthImage=self.renderDepthImage))
        self.parse_objects_id()  # object들의 global id 새 정의
        self.postProcessing()

    # 모든 씬 이름 반환
    def get_scene_names(self):
        return self.scene_list

    def get_scene_types(self):
        return ['Kitchens', 'Living Rooms', 'Bedrooms', 'Bathrooms']

    # 현재 이벤트 반환 (직접 event 호출해도 됨)
    def get_event(self):
        return self.event

    # MoveAhead를 repeat만큼 반복
    def go(self, repeat=1):
        for _ in range(repeat):
            if self.smooth_movement:
                for _ in range(int(0.25/self.magnitude)):
                    self.event = self.CT.step(dict(action='MoveAhead', moveMagnitude=self.magnitude, snapToGrid=False))
            else:
                self.event = self.CT.step(dict(action='MoveAhead'))
            self.step_count += 1
            if not self.check_success('MoveAhead'):  # 실패 시 중지
                return self
        self.postProcessing()
        return self  # (결과값을 통해, 연속적인 액션 가능 ex ct.go().left() ..)

    # MoveBack을 repeat만큼 반복
    def back(self, repeat=1):
        for _ in range(repeat):
            if self.smooth_movement:
                for _ in range(int(0.25 / self.magnitude)):
                    self.event = self.CT.step(dict(action='MoveBack', moveMagnitude=self.magnitude, snapToGrid=False))
            else:
                self.event = self.CT.step(dict(action='MoveBack'))
            self.step_count += 1
            if not self.check_success('MoveBack'):
                return self
        self.postProcessing()
        return self

    # RotateRight을 repeat만큼 반복
    def right(self, repeat=1):
        for _ in range(repeat):
            if self.smooth_movement:
                value = int(90./(1 / self.magnitude))
                for _ in range(int(1 / self.magnitude)):
                    now_angle = self.event.metadata['agent']['rotation']['y']
                    self.event = self.CT.step(dict(action='Rotate', rotation=now_angle+value))
            else:
                self.event = self.CT.step(dict(action='RotateRight'))
            self.step_count += 1
            if not self.check_success('RotateRight'):
                return self
        self.postProcessing()
        return self

    # RotateLeft을 repeat만큼 반복
    def left(self, repeat=1):
        for _ in range(repeat):
            if self.smooth_movement:
                value = int(90./(1 / self.magnitude))
                for _ in range(int(1 / self.magnitude)):
                    now_angle = self.event.metadata['agent']['rotation']['y']
                    self.event = self.CT.step(dict(action='Rotate', rotation=now_angle-value))
            else:
                self.event = self.CT.step(dict(action='RotateLeft'))
            self.step_count += 1
            if not self.check_success('RotateLeft'):
                return self
        self.postProcessing()
        return self

    # MoveRight을 repeat만큼 반복
    def go_right(self, repeat=1):
        for _ in range(repeat):
            if self.smooth_movement:
                for _ in range(int(0.25 / self.magnitude)):
                    self.event = self.CT.step(dict(action='MoveRight', moveMagnitude=self.magnitude, snapToGrid=False))
            else:
                self.event = self.CT.step(dict(action='MoveRight'))
            self.step_count += 1
            if not self.check_success('MoveRight'):
                return self
        self.postProcessing()
        return self

    # MoveLeft을 repeat만큼 반복
    def go_left(self, repeat=1):
        for _ in range(repeat):
            if self.smooth_movement:
                for _ in range(int(0.25 / self.magnitude)):
                    self.event = self.CT.step(dict(action='MoveLeft', moveMagnitude=self.magnitude, snapToGrid=False))
            else:
                self.event = self.CT.step(dict(action='MoveLeft'))
            self.step_count += 1
            if not self.check_success('MoveLeft'):
                return self
        self.postProcessing()
        return self

    # LookUp을 repeat만큼 반복
    def up(self, repeat=1):
        for _ in range(repeat):
            if self.smooth_movement:
                value = int(30./(1 / self.magnitude))
                for _ in range(int(1 / self.magnitude)):
                    now_angle = self.event.metadata['agent']['cameraHorizon']
                    self.event = self.CT.step(dict(action='Look', horizon=now_angle-value))
            else:
                self.event = self.CT.step(dict(action='LookUp'))
            self.step_count += 1
            if not self.check_success('LookUp'):
                return self
        self.postProcessing()
        return self

    # LookDown을 repeat만큼 반복
    def down(self, repeat=1):
        for _ in range(repeat):
            if self.smooth_movement:
                value = int(30./(1 / self.magnitude))
                for _ in range(int(1 / self.magnitude)):
                    now_angle = self.event.metadata['agent']['cameraHorizon']
                    self.event = self.CT.step(dict(action='Look', horizon=now_angle+value))
            else:
                self.event = self.CT.step(dict(action='LookDown'))
            self.step_count += 1
            if not self.check_success('LookDown'):
                return self
        self.postProcessing()
        return self

    def teleport(self, x=0, y=1, z=-1.5):  # y는 위아래. 1이 기본임
        self.event = self.CT.step(dict(action='Teleport', x=x, y=y, z=z))
        self.step_count += 1
        self.check_success('Teleport')
        self.postProcessing()
        return self

    # 액션 리스트를 수행함. ex) move_seq(['go', 'back' ...]) 근데 go(2).left()이런 식이 더 편한듯..
    def move_seq(self, action_list):
        for act in action_list:
            if act == 'go':
                self.go()
            elif act == 'back':
                self.back()
            elif act == 'right':
                self.right()
            elif act == 'left':
                self.left()
            elif act == 'go_right':
                self.go_right()
            elif act == 'go_left':
                self.go_left()
            elif act == 'up':
                self.up()
            elif act == 'down':
                self.down()
        return self

    # OpenObject 수행
    def open(self, object):
        # object : open할 물체(현재물체번호 or 물체딕션 or 물체 아이디) [openable 해야함] # 공백처리는 미구현
        #                   현재물체번호 : objects() 결과의 번호
        if type(object) == str:
            pass
        elif type(object) == dict:
            try:
                object = object['objectId']
            except:
                print('open(\'str objectId\' or \'dict object\')')
                return self
        elif type(object) == int:
            object = self.vis_objs[object]['objectId']
        else:
            print('open(\'str objectId\' or \'dict object\')')
            return self
        self.event = self.CT.step(dict(action='OpenObject', objectId=object))
        self.step_count += 1
        if self.check_success('OpenObject'):
            self.postProcessing()
        return self

    # CloseObject 수행
    def close(self, object):
        # object : close할 물체(현재물체번호 or 물체딕션 or 물체 아이디) [openable 해야함] # 공백처리는 미구현
        #                   현재물체번호 : objects() 결과의 번호
        if type(object) == str:
            pass
        elif type(object) == dict:
            try:
                object = object['objectId']
            except:
                print('close(\'str objectId\' or \'dict object\')')
                return self
        elif type(object) == int:
            object = self.vis_objs[object]['objectId']
        else:
            print('close(\'str objectId\' or \'dict object\')')
            return self
        self.event = self.CT.step(dict(action='CloseObject', objectId=object))
        self.step_count += 1
        if self.check_success('CloseObject'):
            self.postProcessing()
        return self

    # PickupObject 수행
    def pickup(self, object, forceVisible=True):
        # object : pickup할 물체(현재물체번호 or 물체딕션 or 물체 아이디) [pickupable 해야함] # 공백처리는 미구현
        #                   현재물체번호 : objects() 결과의 번호
        # forceVisible : 물체의 visible값이 False여도, pickup 가능하게 해줌
        if type(object) == str:
            pass
        elif type(object) == dict:
            try:
                object = object['objectId']
            except:
                print('pickup(\'str objectId\' or \'dict object\')')
                return self
        elif type(object) == int:
            object = self.vis_objs[object]['objectId']
        else:
            print('pickup(\'str objectId\' or \'dict object\')')
            return self
        self.event = self.CT.step(dict(action='PickupObject', objectId=object, forceVisible=forceVisible))
        self.step_count += 1
        if self.check_success('PickupObject'):
            self.postProcessing()
        return self

    # PutObject 수행
    def put(self, *object):
        # 로봇은 한 물체만 가질 수 있으므로, 대상 물체만 입력
        # 물체 입력안하면 자동으로 찾음. 놓는 대상이 2개이상이거나 없으면 오류반환
        # object : pickup할 물체(현재물체번호 or 물체딕션 or 물체 아이디) [pickupable 해야함] # 공백처리는 미구현
        #                   현재물체번호 : objects() 결과의 번호
        if len(object) == 0:
            rec_objs = self.get_receptacle()
            if len(rec_objs) == 1:
                object = rec_objs[0]
            elif len(rec_objs) == 0:
                print('couldn\'t find receptacle object in visible objects')
                return self
            else:
                print('many receptacle objects are in visible objects')
                return self
        elif len(object) == 1:
            object = object[0]
        else:
            print('put(empty or \'str objectId\' or \'dict object\')')
            return self

        if type(object) == str:
            pass
        elif type(object) == dict:
            try:
                object = object['objectId']
            except:
                print('put(empty or \'str objectId\' or \'dict object\')')
                return self
        elif type(object) == int:
            object = self.vis_objs[object]['objectId']
        else:
            print('put(empty or \'str objectId\' or \'dict object\')')
            return self
        inv_obj = self.get_inventory()
        if inv_obj is None:
            print('robot didn\'t have any item')
            return self
        self.event = self.CT.step(dict(action='PutObject', objectId=inv_obj['objectId'], receptacleObjectId=object))
        self.step_count += 1
        if self.check_success('PutObject'):
            self.postProcessing()
        return self

    def show_inventory(self):
        print(self.event.metadata['inventoryObjects'])

    # 로봇이 가진(pickup된) 물체 아이디를 반환
    def get_inventory(self):
        try:
            return self.event.metadata['inventoryObjects'][0]  # {'objectId':'~~', 'objectType':'Bread'}
        except:
            return None

    # 각 액션이 끝나면 수행
    def postProcessing(self):
        self.update_visible_objects(120) # 보이는 물체들을 인식하고 저장함


    # 보이는 물체들을 인식하고 저장함
    def update_visible_objects(self, area_threshold=120):
        # output visible objects
        vis_objs = []
        if self.renderObjectImage:
            bounding_boxes = self.event.instance_detections2D

        if not self.bbox_to_visible: # obj에 visible 속성이 true여야 추가
            for obj in self.event.metadata['objects']:
                if self.positive_objects is not None and not obj['objectType'] in self.positive_objects:
                    # 제한 범위가 있을 때, 포함되지 않는 물체이면 패스
                    continue
                if obj['visible']:
                    #print('[test] objectId:{}'.format(obj['objectId']))
                    if obj['objectId'] in bounding_boxes:
                        bbox = bounding_boxes[obj['objectId']]
                        if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < area_threshold:  # 물체의 끄트머리만 살짝 나온 경우
                            continue
                        obj['bb'] = bbox  # array([x1, y1, x2, y2])
                        #obj['name'] += '*' # test bb가 있으면 * 표기
                    #else: # 해당하는 bb가 없으면 unvisible로 처리함
                    #    continue
                    vis_objs.append(obj)
        else: # obj에 visible 속성이 true가 아니여도, bbox 목록에 잡히면 해당 물체 추가 (냉장고가 그럼)
            if not self.renderObjectImage:
                raise Exception("bbox_to_visible is True, but renderObjectImage is False")
            vis_obj_ids = list(bounding_boxes.keys())

            for obj in self.event.metadata['objects']:
                if self.positive_objects is not None and not obj['objectType'] in self.positive_objects:
                    # 제한 범위가 있을 때, 포함되지 않는 물체이면 패스
                    continue
                if obj['objectId'] in vis_obj_ids:
                    bbox = bounding_boxes[obj['objectId']]
                    #if (bbox[2]-bbox[0]) * (bbox[3]-bbox[1]) < area_threshold: # 물체의 끄트머리만 살짝 나온 경우
                    #    #print(obj['objectId'], (bbox[2]-bbox[0]) * (bbox[3]-bbox[1]))
                    #    continue

                    box3d = np.array(obj['bounds3D']) * 100
                    rate = np.sum(self.event.instance_masks[obj['objectId']]) / np.product(box3d[3:] - box3d[:3])

                    a = (bbox[2]-bbox[0]) / (bbox[3]-bbox[1])
                    b = (bbox[3]-bbox[1]) / (bbox[2]-bbox[0])
                    xy_rate = a if a > b else b
                    #print(obj['objectId'], ':', rate, xy_rate)
                    if rate < 0.02 or xy_rate > 4:
                        continue
                    obj['bb'] = bbox # array([x1, y1, x2, y2])
                    vis_objs.append(obj)
            #print('*')
        self.vis_objs = vis_objs

    # 보이는 물체들의 이름을 반환
    def get_visible_objects_name(self):
        # output visible objects name list
        return [obj['name'] for obj in self.vis_objs]

    # 보이는 물체들 반환
    def get_visible_objects(self):
        # output visible objects
        return self.vis_objs

    # 현재 씬 내의 모든 물체 반환
    def get_all_objects(self):
        filtered_objects = []
        for obj in self.event.metadata['objects']:
            if self.positive_objects is not None and not obj['objectType'] in self.positive_objects:
                # 제한 범위가 있을 때, 포함되지 않는 물체이면 패스
                continue
            else:
                filtered_objects.append(obj)
        return filtered_objects

    # 보이는 물체들 리스트를 번호별로 출력함 (번호는 액션에 사용가능)
    def objects(self):
        for idx, obj in enumerate(self.vis_objs):
            print('{}: {}'.format(idx, obj['name']))

    def get_receptacle(self):
        # output visible objects
        rec_objs = []
        for obj in self.event.metadata['objects']:
            if obj['visible'] and obj['receptacle']:
                rec_objs.append(obj)
                print('{}: {}'.format(len(rec_objs) - 1, obj['name']))
        return rec_objs

    def check_success(self, action=''):
        if self.event.metadata['lastActionSuccess']:
            return True
        elif self.PRINT_FAIL_LOG:
            print('Action Fail [{}]'.format(action))
            print('\tmsg:' + self.event.metadata['errorMessage'])
        return False

    def get_previous_action_result(self):
        isSuccess = self.event.metadata['lastActionSuccess']
        previous_action = self.event.metadata['lastAction']
        if isSuccess:
            errorMessage = ''
        else:
            errorMessage = self.event.metadata['errorMessage']
            if len(errorMessage) == 0 and previous_action in ['MoveAhead', 'MoveBack', 'MoveRight', 'MoveLeft']:
                errorMessage = 'blocked by something'

        return isSuccess, previous_action, errorMessage

    def get_image(self, mode='normal'):
        if mode == 'opencv':
            return self.event.cv2img
        else:
            return self.event.frame

    def get_depth_image(self):
        return self.event.depth_frame / 20

    def parse_objects_id(self): # 한 씬에서 고유한 id를 새로 정의함 (정렬된 순서대로 0부터 부여)
        self.object_global_ids = {}
        for id, obj in enumerate(self.event.metadata['objects']):
            self.object_global_ids[obj['objectId']] = id

    def find_object_from_od(self, boxes, classes, threshold=0.5):
        # 주어진 물체들과 현재 뷰에서 실재하는 물체들을 비교하고, 실재하는 물체들과 맞는 인덱스를 반환
        # input : (obj#, [x1, y1, x2, y2, label_str])
        hit_objs = []
        keep_idx = []
        for i, box in enumerate(boxes):
            for r_obj in self.get_visible_objects():
                if r_obj['objectType'] != classes[i]:
                    continue
                try:
                    if self._get_iou(r_obj['bb'], box) >= threshold:
                        if not (r_obj in hit_objs):
                            hit_objs.append(r_obj)
                            keep_idx.append(i)
                except KeyError:  # 'bb'가 없을 때.
                    continue
        return hit_objs, keep_idx

    def _get_iou(self, b1, b2):
        # x1, y1, x2, y2
        if b1[0] < b2[0]:
            w = b1[2] - b2[0]
        else:
            w = b2[2] - b1[0]
        if w <= 0:
            return 0.
        if b1[1] < b2[1]:
            h = b1[3] - b2[1]
        else:
            h = b2[3] - b1[1]
        if h <= 0:
            return 0.
        intersection = w*h
        union = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - intersection

        return intersection / union

    def get_object_from_3d(self, box3ds):
        # 3D 좌표값들을 주면, 현재 방에서 해당 좌표랑 같은 물체들을 리턴
        # box3d : [[x, y, z, w, h, d]]
        hit_objs = []
        for i, box in enumerate(box3ds):
            for r_obj in self.get_all_objects():
                position = r_obj['position']
                size = r_obj['bounds3D']
                r_box3d = [position['x'], position['y'], position['z']] + \
                                   [size[3] - size[0], size[4] - size[1], size[5] - size[2]]
                if False not in (box == r_box3d):
                    hit_objs.append(r_obj)

        return hit_objs