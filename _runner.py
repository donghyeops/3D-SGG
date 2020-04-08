# -*- coding:utf-8 -*-

from __future__ import print_function

# import run_thor.controller
import thor_controller.ThorCtrl

class CT:
    def __init__(self, w=1100, h=700):
        self.CT = run_thor.controller.Controller()
        self.CT.start(player_screen_width=w, player_screen_height=h)
        self.event = self.CT.step(dict(action='Initialize', gridSize=0.25))
        self.scene_list = self.CT.scene_names()
        self.scene_name = self.scene_list[0]
        self.vis_objs = self.get_visible_objects()
        
    # 씬 초기화
    def reset(self):
        self.CT.reset(self.scene_name)
        print('scene name : ' + self.scene_name)
        self.event = self.CT.step(dict(action='Initialize', gridSize=0.25))
        self.get_visible_objects()
        
    # 씬 변경
    def set_scene(self, scene_name='FloorPlan28'):
        if not scene_name in self.scene_list:
            print('wrong scene name !')
            return
        self.scene_name = scene_name
        self.CT.reset(self.scene_name)
        print('scene name : ' + self.scene_name)
        self.event = self.CT.step(dict(action='Initialize', gridSize=0.25))
        self.get_visible_objects()
        
    # 다음 씬으로 변경
    def next_scene(self):
        if self.scene_name is None:
            self.scene_name = self.scene_list[0]
        else:
            self.scene_name = self.scene_list[(self.scene_list.index(self.scene_name) + 1) % len(self.scene_list)]
        self.CT.reset(self.scene_name)
        print('scene name : ' + self.scene_name)
        self.event = self.CT.step(dict(action='Initialize', gridSize=0.25))
        self.get_visible_objects()
        
    # 모든 씬 이름 반환
    def get_scenes(self):
        return self.scene_list
        
    # 현재 이벤트 반환 (직접 event 호출해도 됨)
    def get_event(self):
        return self.event
        
    # MoveAhead를 repeat만큼 반복
    def go(self, repeat=1):
        for _ in range(repeat):
            self.event = self.CT.step(dict(action='MoveAhead'))
            if not self.check_success('MoveAhead'): # 실패 시 중지
                return self
        self.get_visible_objects()
        return self # (결과값을 통해, 연속적인 액션 가능 ex ct.go().left() ..)
        
    # MoveBack을 repeat만큼 반복
    def back(self, repeat=1):
        for _ in range(repeat):
            self.event = self.CT.step(dict(action='MoveBack'))
            if not self.check_success('MoveBack'):
                return self
        self.get_visible_objects()
        return self
        
    # RotateRight을 repeat만큼 반복
    def right(self, repeat=1):
        for _ in range(repeat):
            self.event = self.CT.step(dict(action='RotateRight'))
            if not self.check_success('RotateRight'):
                return self
        self.get_visible_objects()
        return self
        
    # RotateLeft을 repeat만큼 반복
    def left(self, repeat=1):
        for _ in range(repeat):
            self.event = self.CT.step(dict(action='RotateLeft'))
            if not self.check_success('RotateLeft'):
                return self
        self.get_visible_objects()
        return self
        
    # MoveRight을 repeat만큼 반복
    def go_right(self, repeat=1):
        for _ in range(repeat):
            self.event = self.CT.step(dict(action='MoveRight'))
            if not self.check_success('MoveRight'):
                return self
        self.get_visible_objects()
        return self
        
    # MoveLeft을 repeat만큼 반복
    def go_left(self, repeat=1):
        for _ in range(repeat):
            self.event = self.CT.step(dict(action='MoveLeft'))
            if not self.check_success('MoveLeft'):
                return self
        self.get_visible_objects()
        return self
        
    # LookUp을 repeat만큼 반복
    def up(self, repeat=1):
        for _ in range(repeat):
            self.event = self.CT.step(dict(action='LookUp'))
            if not self.check_success('LookUp'):
                return self
        self.get_visible_objects()
        return self
        
    # LookDown을 repeat만큼 반복
    def down(self, repeat=1):
        for _ in range(repeat):
            self.event = self.CT.step(dict(action='LookDown'))
            if not self.check_success('LookDown'):
                return self
        self.get_visible_objects()
        return self
    
    def teleport(self, x, z, y=1): # y는 위아래. 1이 기본임
        self.event = self.CT.step(dict(action='Teleport', x=x, y=y, z=z))
        self.check_success('Teleport')
        self.get_visible_objects()
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
        if type(object) == unicode or type(object) == str:
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
        if self.check_success('OpenObject'):
            self.get_visible_objects()
        return self
        
    # CloseObject 수행
    def close(self, object):
        # object : close할 물체(현재물체번호 or 물체딕션 or 물체 아이디) [openable 해야함] # 공백처리는 미구현
        #                   현재물체번호 : objects() 결과의 번호
        if type(object) == unicode or type(object) == str:
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
        if self.check_success('CloseObject'):
            self.get_visible_objects()
        return self
        
    # PickupObject 수행
    def pickup(self, object):
        # object : pickup할 물체(현재물체번호 or 물체딕션 or 물체 아이디) [pickupable 해야함] # 공백처리는 미구현
        #                   현재물체번호 : objects() 결과의 번호
        if type(object) == unicode or type(object) == str:
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
        self.event = self.CT.step(dict(action='PickupObject', objectId=object))
        if self.check_success('PickupObject'):
            self.get_visible_objects()
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
            
        if type(object) == unicode or type(object) == str:
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
        self.event = self.CT.step(dict(action='PutObject', objectId=inv_obj, receptacleObjectId=object))
        if self.check_success('PutObject'):
            self.get_visible_objects()
        return self
        
    def show_inventory(self):
        print(self.event.metadata['inventoryObjects'])
        
    # 로봇이 가진(pickup된) 물체 아이디를 반환
    def get_inventory(self):
        try:
            return self.event.metadata['inventoryObjects'][0]['objectId']
        except:
            return None
        
    # 보이는 물체들을 인식함
    def get_visible_objects(self):
        # output visible objects
        vis_objs = []
        for obj in self.event.metadata['objects']:
            if obj['visible']:
                vis_objs.append(obj)
        self.vis_objs = vis_objs
        return vis_objs
        
    # 보이는 물체들 리스트를 번호별로 출력함 (번호는 액션에 사용가능)
    def objects(self):
        for idx, obj in enumerate(self.vis_objs):
            print('{}: {}'.format(idx, obj['name']))
        
    def all_objects(self):
        return self.event.metadata['objects']
    
    def get_receptacle(self):
        # output visible objects
        rec_objs = []
        for obj in self.event.metadata['objects']:
            if obj['visible'] and obj['receptacle']:
                rec_objs.append(obj)
                print('{}: {}'.format(len(rec_objs)-1, obj['name']))
        return rec_objs
        
    def check_success(self, action=''):
        if self.event.metadata['lastActionSuccess']:
            return True
        else:
            print('Action Fail [{}]'.format(action))
            print('\tmsg:'+self.event.metadata['errorMessage'])
            return False

