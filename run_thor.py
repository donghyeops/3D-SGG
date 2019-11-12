# -*- coding:utf-8 -*-

from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QTextEdit, QComboBox, QLineEdit, QHBoxLayout, \
    QVBoxLayout, QLabel, QWidget, QScrollArea, QCheckBox, QTextBrowser, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import uic
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot, Qt, QProcess
import sys
from thor_controller import ThorCtrl
from scene_graph_manager import DynamicGlobalSceneGraphManager

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from thor_utils import draw_util as du, annotation_util as au, owl_util as ou
from dataset import ThorDB
import graphviz as gv
import random
import threading
import time
import json

CHECK_TIME = False


class Ai2Thor_GUI(QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.ui = uic.loadUi("ai2thor_gui.ui", self)
        self.setWindowTitle('AI2THOR Controller')
        self.ui.show()
        self.thorCtrl = ThorCtrl(positive_objects=au.object_label)
        # self.thorCtrl = None
        self.scene_type2name = {}
        self.init_init_scene_combobox()  # 콤보박스 초기화
        self.init_teleport()  # 체크박스 초기화
        self.init_init_detection_view_checkbox() # 시각화 체크박스에 이벤트 달기

        self.recorder_init_setting()
        self.thorDB = ThorDB()

        self.visual_minimize = False # True 시, GUI 갱신 안함 (auto record 시, 최적화를 위해 사용)

        # 두 시각화 창 생성 (초기엔 hide)
        self.SgDialog = SubDialog(self, title='Scene Graph', x=500, y=0)  # 현재 frame에 해당하는 scene graph
        self.OdDialog = SubDialog(self, title='Objects', x=0, y=0)  # 현재 frame에 해당하는 object detection 결과
        self.OMDialog = SubDialog(self, title='Object Memory', x=800, y=0)  # 현재 frame에 해당하는 object detection 결과

        self.record_action = self.ui.check_record_action.isChecked()  # action만 재현을 목적으로 따로 기록 [일단 이동만]
        if self.record_action:
            self.action_history = []

        self.GSGMgr = DynamicGlobalSceneGraphManager(
            thorCtrl=self.thorCtrl,
            use_dnn=self.ui.check_use_dnn.isChecked(),
            dnn_check_dict=self.get_dnn_check_dict()
        )

        self.SAVE_IMAGE = False
        print('SAVE_IMAGE:', self.SAVE_IMAGE)
        self.GSGMgr.set_use_history(True) ##

        self.action_fail_mode = False
        success_rate = 0.8
        print('action_fail_mode:', self.action_fail_mode, success_rate)
        if self.action_fail_mode:
            self.roulette = [True] * round(success_rate * 10) + [False] * round((1 - success_rate) * 10)


    def init_init_scene_combobox(self):
        scene_types = self.thorCtrl.get_scene_types()
        for scene_type in scene_types:
            self.scene_type2name[scene_type] = []
        self.ui.cb_scene_type.addItems(scene_types)  # 타입 추가
        self.ui.cb_scene_type.currentTextChanged.connect(self.init_update_scene_name_combobox)

        scene_names = self.thorCtrl.get_scene_names()
        #print(scene_names)
        for scene_name in scene_names:
            #scene_name = scene_name.split('_')[0]
            # print(scene_name.split('FloorPlan'))
            floor = int(scene_name.split('FloorPlan')[1]) // 100
            if floor < 2:
                self.scene_type2name[scene_types[0]].append(scene_name)
            else:
                self.scene_type2name[scene_types[floor - 1]].append(scene_name)
        self.init_update_scene_name_combobox()

    def init_update_scene_name_combobox(self):
        # 씬 이름 콤보박스 업데이트
        self.ui.cb_scene_name.clear()
        self.ui.cb_scene_name.addItems(self.scene_type2name[self.ui.cb_scene_type.currentText()])

    def init_init_detection_view_checkbox(self):
        # object detection / scene graph / global scene graph 시각화 체크박스에 이벤트 달기
        def show_od_viewer(self):
            # object detection 체크 시, 뷰어 열기
            def event():
                if self.ui.check_show_bbox.isChecked():
                    self.OdDialog.show()
                    # 부모 창을 활성화 (얘가 없으면 클릭해서 다시 활성화해야함)
                    # 활성화 안시키면 단축키가 안먹힘
                    self.activateWindow()
                else:
                    self.OdDialog.hide()
            return event
        def show_sg_viewer(self):
            # scene graph 체크 시, 뷰어 열기
            def event():
                if self.ui.check_show_graph.isChecked():
                    self.SgDialog.show()
                    self.activateWindow()
                else:
                    self.SgDialog.hide()
            return event
        def show_om_viewer(self):
            # OM(object memory) 체크 시, 뷰어 열기
            def event():
                if self.ui.check_show_om.isChecked():
                    self.OMDialog.show()
                    self.activateWindow()
                else:
                    self.OMDialog.hide()
            return event
        def use_dnn(self):
            # Use DNN 체크 시, DNN 불러오기
            def event():
                self.ui.check_dnn_od.setChecked(True)
                self.ui.check_dnn_att.setChecked(True)
                self.ui.check_dnn_transfer.setChecked(True)
                self.ui.check_dnn_rel.setChecked(True)

                self.GSGMgr.set_use_dnn(self.ui.check_use_dnn.isChecked(), self.get_dnn_check_dict())
                self.GSGMgr.reset()  # dnn 누르면 그때부터 새로 시작

            return event
        def check_dnn_usage(self):
            # ObjNet 등 체크 시, 매니저 업데이트
            def event():
                self.GSGMgr.set_use_dnn(self.ui.check_use_dnn.isChecked(), self.get_dnn_check_dict())

            return event
        def record_action(self):
            def event():
                if self.ui.check_record_action.isChecked():
                    self.action_history = []
                    self.record_action = True
                else:
                    if hasattr(self, 'action_history'):
                        del self.action_history
                    self.record_action = False
            return event
        self.ui.check_show_bbox.stateChanged.connect(show_od_viewer(self))  # Object Detection 결과
        self.ui.check_show_graph.stateChanged.connect(show_sg_viewer(self))  # Scene Graph
        self.ui.check_show_om.stateChanged.connect(show_om_viewer(self))  # Object Memory 3D boxes

        self.ui.check_use_dnn.stateChanged.connect(use_dnn(self))  # DNN 사용할 지 말지
        self.ui.check_dnn_od.stateChanged.connect(check_dnn_usage(self))  # DNN 체크 업데이트
        self.ui.check_dnn_att.stateChanged.connect(check_dnn_usage(self))  # DNN 체크 업데이트
        self.ui.check_dnn_transfer.stateChanged.connect(check_dnn_usage(self))  # DNN 체크 업데이트
        self.ui.check_dnn_rel.stateChanged.connect(check_dnn_usage(self))  # DNN 체크 업데이트

        self.ui.check_record_action.stateChanged.connect(record_action(self))  # action들 기록할지 말지 [나중에 재현하기위해 기록]

    def recorder_init_setting(self):
        self.is_recording = False
        self.is_image_dir_opened = False
        self.is_object_file_opened = False
        self.image_dirName = None
        self.object_fileName = None
        self.object_log = []
        self.record_step_count = 0 # record step을 나타냄. thorCtrl이랑 독립적임
        self.auto_recording = False

    def get_dnn_check_dict(self):
        dnn_check_dict = {
            'od': self.ui.check_dnn_od.isChecked(),
            'att': self.ui.check_dnn_att.isChecked(),
            'transfer': self.ui.check_dnn_transfer.isChecked(),
            'rel': self.ui.check_dnn_rel.isChecked()
        }
        return dnn_check_dict

    @pyqtSlot()
    def init_start(self):
        self.ui.bt_start.setEnabled(False)  # 구동 중일 때, 클릭 불가
        try:
            window_w = float(self.tx_window_w.text())
            window_h = float(self.tx_window_h.text())
        except:
            print('window size is not float')
            return
        if not self.thorCtrl.is_powered():
            self.thorCtrl.power_on(w=window_w, h=window_h)
        try:
            grid_size = float(self.tx_grid_size.text())
        except:
            print('grid size is not float')
            return

        self.thorCtrl.start(self.ui.cb_scene_name.currentText(), grid_size)
        self.log_clear()
        if self.ui.check_teleport.isChecked():
            try:
                x = float(self.tx_teleport_x.text())
                y = float(self.tx_teleport_y.text())
                z = float(self.tx_teleport_z.text())
            except:
                print('(teleport) x, y, z is not float')
                return
            self.action_teleport(x, y, z)()  #
        self.SgDialog.set_size(800, 600) # 시각화 창 크기 조절
        self.OdDialog.set_size(600, 600) # 시각화 창 크기 조절
        self.OMDialog.set_size(1400, 400)  # 시각화 창 크기 조절

        self.action_show_objects()
        self.ui.bt_start.setText('exit')
        self.ui.bt_start.clicked.disconnect()
        self.ui.bt_start.clicked.connect(self.init_end)
        self.ui.bt_start.setEnabled(True)  # 구동 끝나면 다시 활성화
        self.activateWindow()

    def init_end(self):
        self.ui.bt_start.setEnabled(False)  # 구동 중일 때, 클릭 불가
        self.thorCtrl.power_off()
        self.ui.bt_start.setText('start')
        if hasattr(self, 'GSGMgr'):
            self.GSGMgr.reset()
        self.ui.bt_start.clicked.disconnect()
        self.ui.bt_start.clicked.connect(self.init_start)
        self.ui.bt_start.setEnabled(True)  # 구동 끝나면 다시 활성화

    @pyqtSlot()
    def init_reset(self, seed=None):
        if not self.thorCtrl.is_powered():
            return
        self.thorCtrl.set_scene(scene_name=self.ui.cb_scene_name.currentText(), seed=seed)
        self.log_clear()
        if self.ui.check_teleport.isChecked():
            try:
                x = float(self.tx_teleport_x.text())
                y = float(self.tx_teleport_y.text())
                z = float(self.tx_teleport_z.text())
            except:
                print('(teleport) x, y, z is not float')
                return
            self.action_teleport(x, y, z)()
        self.action_show_objects()
        self.action_show_inventory()
        self.action_show_result()
        if hasattr(self, 'GSGMgr'):
            self.GSGMgr.reset()
        self.SgDialog.clear()
        self.OdDialog.clear()
        self.OMDialog.clear()

        self.activateWindow()
        if self.ui.check_record_action.isChecked():  # action 저장 시, history 초기화
            self.action_history = []
        if hasattr(self, 'draw_answer'):  # OM에서 정답 그림 그린 여부
            del self.draw_answer

    def init_teleport(self):
        self.ui.check_teleport.stateChanged.connect(self.init_check_teleport)  # 이벤트 등록
        self.init_check_teleport()  # 최초 갱신

    @pyqtSlot()
    def init_check_teleport(self):
        isChecked = self.ui.check_teleport.isChecked()
        self.ui.tx_teleport_x.setEnabled(isChecked)
        self.ui.tx_teleport_y.setEnabled(isChecked)
        self.ui.tx_teleport_z.setEnabled(isChecked)

    @pyqtSlot()
    def action_go(self, **kwargs):
        output_dict = {'action': 'go'}

        if self.action_fail_mode:
            success = random.choice(self.roulette)
            kwargs['success'] = success

        if 'success' in kwargs:
            output_dict['success'] = kwargs['success']
            if kwargs['success']:
                self.thorCtrl.go()
        else:
            self.thorCtrl.go()

        self.action_postProcessing(**output_dict)

    @pyqtSlot()
    def action_back(self, **kwargs):
        output_dict = {'action': 'back'}

        if self.action_fail_mode:
            success = random.choice(self.roulette)
            kwargs['success'] = success

        if 'success' in kwargs:
            output_dict['success'] = kwargs['success']
            if kwargs['success']:
                self.thorCtrl.back()
        else:
            self.thorCtrl.back()
        self.action_postProcessing(**output_dict)

    @pyqtSlot()
    def action_go_left(self, **kwargs):
        output_dict = {'action': 'go_left'}

        if self.action_fail_mode:
            success = random.choice(self.roulette)
            kwargs['success'] = success

        if 'success' in kwargs:
            output_dict['success'] = kwargs['success']
            if kwargs['success']:
                self.thorCtrl.go_left()
        else:
            self.thorCtrl.go_left()
        self.action_postProcessing(**output_dict)

    @pyqtSlot()
    def action_go_right(self, **kwargs):
        output_dict = {'action': 'go_right'}

        if self.action_fail_mode:
            success = random.choice(self.roulette)
            kwargs['success'] = success

        if 'success' in kwargs:
            output_dict['success'] = kwargs['success']
            if kwargs['success']:
                self.thorCtrl.go_right()
        else:
            self.thorCtrl.go_right()
        self.action_postProcessing(**output_dict)

    @pyqtSlot()
    def action_left(self, **kwargs):
        output_dict = {'action': 'left'}

        if self.action_fail_mode:
            success = random.choice(self.roulette)
            kwargs['success'] = success

        if 'success' in kwargs:
            output_dict['success'] = kwargs['success']
            if kwargs['success']:
                self.thorCtrl.left()
        else:
            self.thorCtrl.left()
        self.action_postProcessing(**output_dict)

    @pyqtSlot()
    def action_right(self, **kwargs):
        output_dict = {'action': 'right'}

        if self.action_fail_mode:
            success = random.choice(self.roulette)
            kwargs['success'] = success

        if 'success' in kwargs:
            output_dict['success'] = kwargs['success']
            if kwargs['success']:
                self.thorCtrl.right()
        else:
            self.thorCtrl.right()
        self.action_postProcessing(**output_dict)

    @pyqtSlot()
    def action_up(self, **kwargs):
        output_dict = {'action': 'up'}

        if self.action_fail_mode:
            success = random.choice(self.roulette)
            kwargs['success'] = success

        if 'success' in kwargs:
            output_dict['success'] = kwargs['success']
            if kwargs['success']:
                self.thorCtrl.up()
        else:
            self.thorCtrl.up()
        self.action_postProcessing(**output_dict)

    @pyqtSlot()
    def action_down(self, **kwargs):
        output_dict = {'action': 'down'}

        if self.action_fail_mode:
            success = random.choice(self.roulette)
            kwargs['success'] = success

        if 'success' in kwargs:
            output_dict['success'] = kwargs['success']
            if kwargs['success']:
                self.thorCtrl.down()
        else:
            self.thorCtrl.down()
        self.action_postProcessing(**output_dict)

    def action_open(self, object_number, **kwargs):
        def open():
            output_dict = {'action': 'open', 'target_object':object_number}

            if self.action_fail_mode:
                success = random.choice(self.roulette)
                kwargs['success'] = success

            if 'success' in kwargs:
                output_dict['success'] = kwargs['success']
                if kwargs['success']:
                    self.thorCtrl.open(object_number)
            else:
                self.thorCtrl.open(object_number)
            self.action_postProcessing(**output_dict)

        return open

    def action_close(self, object_number, **kwargs):
        def close():
            output_dict = {'action': 'close', 'target_object':object_number}

            if self.action_fail_mode:
                success = random.choice(self.roulette)
                kwargs['success'] = success

            if 'success' in kwargs:
                output_dict['success'] = kwargs['success']
                if kwargs['success']:
                    self.thorCtrl.close(object_number)
            else:
                self.thorCtrl.close(object_number)
            self.action_postProcessing(**output_dict)

        return close

    def action_pickup(self, object_number, **kwargs):
        def pickup():
            output_dict = {'action': 'pickup', 'target_object':object_number}

            if self.action_fail_mode:
                success = random.choice(self.roulette)
                kwargs['success'] = success

            if 'success' in kwargs:
                output_dict['success'] = kwargs['success']
                if kwargs['success']:
                    self.thorCtrl.pickup(object_number)
            else:
                self.thorCtrl.pickup(object_number)
            self.action_postProcessing(**output_dict)

        return pickup

    def action_put(self, object2_number, **kwargs):
        # object2_number는 장소 물체임, 자동으로 손에 든 물체가 들어감
        def put():
            output_dict = {'action': 'put', 'target_object':object2_number}

            if self.action_fail_mode:
                success = random.choice(self.roulette)
                kwargs['success'] = success

            if 'success' in kwargs:
                output_dict['success'] = kwargs['success']
                if kwargs['success']:
                    self.thorCtrl.put(object2_number)
            else:
                self.thorCtrl.put(object2_number)
            self.action_postProcessing(**output_dict)

        return put

    def action_teleport(self, x, y, z, **kwargs):
        def teleport():
            output_dict = {'action': 'teleport', 'target_position': [x, y, z]}
            self.thorCtrl.teleport(x, y, z)
            self.action_postProcessing(**output_dict)

        return teleport

    @pyqtSlot()
    def capture(self, image_path=None, depth_path=None):
        if image_path is None:
            image_name = 'screenshot_' + self.thorCtrl.scene_name + '.jpg'
            image_path = os.path.join('.', image_name)
        opencv_frame = self.thorCtrl.get_image('opencv')  # BGR
        cv2.imwrite(image_path, opencv_frame) # 이미지 저장

        if depth_path is None:
            depth_name = 'screenshot_d_' + self.thorCtrl.scene_name + '.jpg'
            depth_path = os.path.join('.', depth_name)
        opencv_frame = self.thorCtrl.get_depth_image()  # depth 이미지
        cv2.imwrite(depth_path, opencv_frame)  # 이미지 저장


    def action_postProcessing(self, **inputs):
        if not self.visual_minimize:
            # 각 액션이 끝나면 수행
            self.action_show_objects() # visible object list 화면 업데이트
            self.action_show_inventory() # inventory 화면 업데이트
            self.action_show_result() # action 결과 화면 업데이트
            self.action_update_log() # scene log (view) 업데이트
            if CHECK_TIME:
                sub_st2 = time.time()
            isSuccess, previous_action, errorMessage = self.thorCtrl.get_previous_action_result()
            if isSuccess:
                if 'target_object' in inputs:
                    inputs['target_obj'] = self.__previous_objs[int(inputs['target_object'])]
                    #print(inputs['target'])
                    #print(inputs['target_obj'])
                self.__previous_objs = self.thorCtrl.get_visible_objects()
                self.GSGMgr.apply_action_model(inputs)  # 행동 모델에 따라 인식 기능 수행 (renew SceneMap)
                #ou.write_owl_from_gsg(self.GSGMgr.sceneMap.get_gsg_history_only_now(),
                #                      file_name=f'{self.thorCtrl.scene_name}_T{self.GSGMgr.sceneMap.get_time()}.owl',
                #                      dir_path=f'./dynamic_gsg_owls/{self.thorCtrl.scene_name}') # only_now 안쓰면 한 파일에 모두 씀
                self.action_visualize_sceneMap()  # 인식 결과 출력 (read SceneMap)
                if self.SAVE_IMAGE:
                    image_dir = '/media/ailab/D/ai2thor/thorDBv2/images'
                    depth_dir = '/media/ailab/D/ai2thor/thorDBv2/depth_images'
                    if not os.path.isdir(image_dir):
                        os.makedirs(image_dir, exist_ok=True)
                    if not os.path.isdir(depth_dir):
                        os.makedirs(depth_dir, exist_ok=True)
                    t=self.GSGMgr.sceneMap.get_time()
                    sn = self.thorCtrl.scene_name
                    image_path = os.path.join(image_dir, f'{sn}_{str(t)}.jpg')
                    depth_path = os.path.join(depth_dir, f'{sn}_d_{str(t)}.jpg')
                    self.capture(image_path=image_path, depth_path=depth_path)
            if CHECK_TIME:
                sub_time = time.time() - sub_st2
                print('[TIME] action_visualize_detection() : {}s'.format(str(sub_time)[:4]))
                print('*************')
            # self.action_bb_test() # 바운딩 박스 시각화 테스트
            # if len(inputs) > 0:
            #    print(inputs)
        if self.is_recording:
            self.action_record()  # 레코드 체크 및 이미지 저장/로그 추가
        if self.record_action:
            self.action_record_action(inputs)  # action만 재현을 위해 따로 저장


    def action_record_action(self, inputs):
        # action = [inputs['action']]
        # if 'target' in inputs:
        #     action += [inputs['target']]
        # self.action_history.append(action)
        if 'target_obj' in inputs:
            inputs.pop('target_obj')
        self.action_history.append(inputs)

    def action_show_objects(self):
        # object 액션 리스트 갱신
        vis_objs = self.thorCtrl.get_visible_objects()

        for i in reversed(range(self.ui.layout_objects.count())):  # 스크롤 영역 clear
            layout = self.ui.layout_objects.itemAt(i).layout()
            for j in reversed(range(layout.count())):
                layout.itemAt(j).widget().deleteLater()
            layout.deleteLater()
        for idx, obj in enumerate(vis_objs):
            container = QHBoxLayout()
            label, open_btn, pickup_btn = QLabel(), QPushButton(), QPushButton()
            container.addWidget(label)
            container.addWidget(open_btn)
            container.addWidget(pickup_btn)

            self.ui.layout_objects.addLayout(container)  # 스크롤 영역에 추가

            label.setText(obj['objectType']) # name으로 변경 가능
            if obj['openable']:  # open/close 체크
                if obj['isopen']:
                    open_btn.setText('close')
                    open_btn.clicked.connect(self.action_close(idx))
                else:
                    open_btn.setText('open')
                    open_btn.clicked.connect(self.action_open(idx))
            else:
                open_btn.setEnabled(False)
            if obj['receptacle']:  # put/pickup 체크
                if self.thorCtrl.get_inventory() is None:
                    pickup_btn.setText('put(no item)')
                    pickup_btn.setEnabled(False)
                elif obj['openable'] and not obj['isopen']:
                    pickup_btn.setText('put(closed)')
                    pickup_btn.setEnabled(False)
                elif obj['receptacleCount'] == len(obj['receptacleObjectIds']):
                    pickup_btn.setText('put(full)')
                    pickup_btn.setEnabled(False)
                else:
                    pickup_btn.setText('put')
                    pickup_btn.clicked.connect(self.action_put(idx))
            elif obj['pickupable']:
                if self.thorCtrl.get_inventory() is not None:
                    pickup_btn.setText('pickup(full)')
                    pickup_btn.setEnabled(False)
                else:
                    pickup_btn.setText('pickup')
                    pickup_btn.clicked.connect(self.action_pickup(idx))
            else:
                pickup_btn.setEnabled(False)

    def action_show_inventory(self):
        item = self.thorCtrl.get_inventory()
        if item is None:
            self.ui.tx_inventory.setText('')
        else:
            self.ui.tx_inventory.setText(item['objectType'])

    def action_show_result(self):
        isSuccess, previous_action, errorMessage = self.thorCtrl.get_previous_action_result()
        if isSuccess:
            self.ui.la_success.setText('success')
            self.ui.la_error_msg.setText('')
        else:
            self.ui.la_success.setText('fail')
            self.ui.la_error_msg.setText(errorMessage)
        self.ui.la_action.setText(previous_action)

    def action_update_log(self):
        vis_objs = self.thorCtrl.get_visible_objects()
        isSuccess, previous_action, errorMessage = self.thorCtrl.get_previous_action_result()
        self.ui.tb_scene_log.append('[step {}]'.format(self.thorCtrl.step_count))

        success_text = 'success' if isSuccess else 'fail'

        self.ui.tb_scene_log.append('action : {} ({})'.format(previous_action, success_text))
        if not isSuccess:
            self.ui.tb_scene_log.append('   error msg : {}'.format(errorMessage))
        self.ui.tb_scene_log.append('visible object :')
        if len(vis_objs) == 0:
            self.ui.tb_scene_log.append('  nothing')
        for idx, vis_obj in enumerate(vis_objs):
            self.ui.tb_scene_log.append('  {}. n:{} id:{}'.format(idx+1, vis_obj['name'], vis_obj['objectId']))
        self.ui.tb_scene_log.append('-'*50)

    def action_record(self):
        # 레코딩 중이 아니거나, 일시정지(pause) 중이면 기록 안함, (액션 끝날 때마다 호출됨)
        if not self.is_recording:
            return
        isSuccess, _, _ = self.thorCtrl.get_previous_action_result()
        if not isSuccess:
            return

        if hasattr(self, 'record_count'):
            self.record_count += 1
        else:
            self.record_count = 1  # record 한 번 하면서 시작하므로 1부터 시작

        scene_info = self.recorder_get_scene_annotation(save_image=True) # scene annotation 생성 & 영상 저장
        self.thorDB.add(scene_info) # DB에 현재 frame annotation 추가
        if self.record_count % 100 == 0 and self.record_count > 0:
            self.thorDB.save_json(self.object_fileName) # 축적된 annotation을 json으로 저장

        self.ui.la_record_count.setText(str(self.record_count))
        self.ui.la_db_len.setText(str(self.thorDB.next_data_id))

    def action_visualize_sceneMap(self):
        # 그래프 그릴 때 쓸 색상 생성
        colors_plt = plt.cm.Set3(np.linspace(0, 1, len(au.object_label)))
        rgb = (colors_plt[:, :3] * 255).astype('int32')
        colors_gv = []
        for row in rgb:
            out = '#'
            for value in row:
                if len(hex(value)[2:]) == 1:
                    out += '0' + hex(value)[2:]
                else:
                    out += hex(value)[2:]
            colors_gv.append(out)

        # bbox 시각화
        if self.ui.check_show_bbox.isChecked():
            if CHECK_TIME:
                sub_st = time.time()
            frame = self.thorCtrl.get_image()  # RGB
            if not 'bbox_fig' in dir(self):
                self.bbox_fig = plt.figure(figsize=(6.0, 6.0))
                ax = plt.Axes(self.bbox_fig, [0., 0., 1., 1.])
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                self.bbox_fig.add_axes(ax)
                self.ax = ax

            self.ax.cla()
            self.ax.imshow(frame)

            od_results = self.GSGMgr.get_od_results() # sceneMap으로부터 OD 결과 가져옴
            boxes = od_results['boxes']
            classes = od_results['classes']

            # DNN 안써도 동일하게 동작 (위 주석 코드는 원본)
            for i in range(len(boxes)):
                rect = patches.Rectangle((boxes[i][0], boxes[i][1]), boxes[i][2] - boxes[i][0],
                                         boxes[i][3] - boxes[i][1],
                                         linewidth=6, edgecolor=colors_plt[au.obj_s2i[classes[i]]],
                                         facecolor='none')
                self.ax.add_patch(rect)
                # self.ax.text(boxes[i][0], boxes[i][1] - 9, classes[i],
                #              style='italic',
                #              bbox={'facecolor': colors_plt[au.obj_s2i[classes[i]]], 'alpha': 0.5}, fontsize=15)
                self.ax.text(boxes[i][0], boxes[i][1] - 15, classes[i],
                             style='italic',
                             bbox={'facecolor': colors_plt[au.obj_s2i[classes[i]]], 'alpha': 0.5}, fontsize=25)

            # plt.show()
            plt.savefig(fname='temp_od.jpg', bbox_inches='tight', pad_inches=0)
            self.OdDialog.show_image(filePath='temp_od.jpg')
            if CHECK_TIME:
                sub_time = time.time() - sub_st
                print('[TIME] draw OD results : {}s'.format(str(sub_time)[:4]))

        gsg = self.GSGMgr.get_gsg() # sceneMap으로부터 gsg 가져옴

        # 3D 위치 시각화
        if self.ui.check_show_om.isChecked():
            if not hasattr(self, 'draw_answer'):
                if CHECK_TIME:
                    sub_st = time.time()
                aws_objects = self.thorCtrl.get_all_objects()
                aws_poses = []
                aws_labals = []
                for obj in aws_objects:
                    pos = []
                    pos += [obj['position']['x'], obj['position']['y'], obj['position']['z']]
                    b_3d = obj['bounds3D']
                    size_3d = [b_3d[3] - b_3d[0], b_3d[4] - b_3d[1], b_3d[5] - b_3d[2]]
                    pos += size_3d
                    aws_poses.append(pos)
                    aws_labals.append(au.obj_s2i[obj['objectType']])
                map_img_path = du.draw_object_3d_map(pos=np.array(aws_poses), labels=np.array(aws_labals),
                                                     colors=colors_gv,
                                                     draw_answer=True)
                self.draw_answer = True
                if CHECK_TIME:
                    sub_time = time.time() - sub_st
                    print('[TIME] Aws draw_object_3d_map() : {}s'.format(str(sub_time)[:4]))

            if CHECK_TIME:
                sub_st = time.time()
            if len(gsg['objects']) > 0:
                gsg_objects = gsg['objects']
                gsg_pos = np.array([obj['box3d'] for obj in gsg_objects.values()])
                gsg_labels = np.array([obj['label'] for obj in gsg_objects.values()])
                map_img_path = du.draw_object_3d_map(pos=gsg_pos, labels=gsg_labels, colors=colors_gv)

                # 에이전트 위치 그리기
                # agent_info = self.thorCtrl.get_event().metadata['agent']
                # agent_pos = [agent_info['position']['x'], agent_info['position']['y'],
                #              agent_info['position']['z']]
                # #print('agent pos:', agent_pos)  ###
                #
                # map_img_path = du.draw_agent(agent_pos, draw_answer=False)

                self.OMDialog.show_image(filePath=map_img_path) # 에러 없으면 if문 밖에 둬야함 ###
            if CHECK_TIME:
                sub_time = time.time() - sub_st
                print('[TIME] Predict DRAW & SHOW draw_object_3d_map() : {}s'.format(str(sub_time)[:4]))

        # Scene Graph 시각화
        if self.ui.check_show_graph.isChecked():
            sg = gv.Digraph('structs', format='png')  # 그래프 생성
            # sg = gv.Digraph('structs', format='pdf')  # 그래프 생성
            # graphviz 색상표
            # https://www.graphviz.org/doc/info/colors.html#brewer

            if CHECK_TIME:
                sub_st = time.time()
            for oid, obj in gsg['objects'].items():  # object node, 속성 node/edge 추가
                # for idx in range(len(boxes)):
                if obj['detection']:
                    penwidth = '2'
                    pencolor = 'red'
                else:
                    penwidth = '0'
                    pencolor = 'blue'
                with sg.subgraph(name=str(oid)) as obj_g:
                    # 물체 추가
                    obj_g.node(str(oid), label=au.obj_i2s[int(obj['label'])]+f'_{oid}', shape='box',
                               style='filled', fillcolor=colors_gv[int(obj['label'])], fontsize='20',
                               penwidth=penwidth,
                               color=pencolor)

                    if int(obj['open_state']) != au.openState_s2i['unable']:
                        # is_open 추가
                        obj_g.node(str(oid) + '_isOpen', label=au.openState_i2s[int(obj['open_state'])],
                                   shape='ellipse', style='filled', color='lightseagreen', fontsize='15')
                        obj_g.edge(str(oid) + '_isOpen', str(oid), dir='none')

                    # color 추가
                    obj_g.node(str(oid) + '_color', label=au.color_i2s[int(obj['color'])],
                               shape='ellipse',
                               style='filled', color='lightskyblue1', fontsize='15')
                    obj_g.edge(str(oid) + '_color', str(oid), dir='none')

            for key, relation in gsg['relations'].items():  # 관계 edge 추가
                if au.rel_i2s[relation['rel_class']] == 'background':
                    continue
                sg.edge(str(relation['subject_id']), str(relation['object_id']),
                        label=' ' + au.rel_i2s[relation['rel_class']], fontsize='16')

            # agent 정보 추가 (agent, inventory_object)
            with sg.subgraph(name=str(999)) as obj_g:
                obj_g.node(str(999), label='Agent', shape='box',
                           style='filled', fontsize='25',
                           color='gray')
                owned_obj_id = self.GSGMgr.sceneMap.get_agent_object_id()
                if owned_obj_id is not None:
                    sg.edge(str(999), str(owned_obj_id),
                            label=' has', fontsize='16')


            if CHECK_TIME:
                sub_time = time.time() - sub_st
                print('[TIME] draw Graph (DNN results) : {}s'.format(str(sub_time)[:4]))

            # sg.render('./gv_temp.gv', view=True, cleanup=True) # 랜더링
            if CHECK_TIME:
                sub_st = time.time()
            # 랜더링이 상당히 오래걸림
            filePath = sg.render('./temp', view=False, cleanup=False)  # 랜더링
            self.SgDialog.show_image(filePath=filePath)
            if CHECK_TIME:
                sub_time = time.time() - sub_st
                print('[TIME] show scene graph : {}s'.format(str(sub_time)[:4]))

    @pyqtSlot()
    def save_results(self, seed=None):
        # GSG 결과 => json, owl
        # action history => json
        if self.ui.check_use_dnn.isChecked():
            dir_path = './datasets/190514 gsg_pred_OATR_only_recog'
        else:
            dir_path = './datasets/190514 gsg_gt'
        os.makedirs(dir_path, exist_ok=True)

        scene_name = self.thorCtrl.scene_name

        #### action history to json ####
        if self.record_action:
            action_path = os.path.join(dir_path, 'action_history')
            os.makedirs(action_path, exist_ok=True)
            action_db = dict()
            action_db['actions'] = self.action_history
            action_db['scene_name'] = scene_name
            with open(os.path.join(action_path, scene_name + '.json'), 'w') as f:
                json.dump(action_db, f, indent='\t')
            print('save action_history ({})'.format(os.path.join(action_path, scene_name+'.json')))
        ###############################

        ######## GSG to OWL ########
        # sceneMap의 use_history가 켜져있어야 작동됨 (action history load시 자동으로 켜짐)
        owl_path = os.path.join(dir_path, 'owl', scene_name)
        if seed is not None:
            owl_path = os.path.join(owl_path, f'S{seed}')
            file_prefix = f'{scene_name}_S{seed}'
        else:
            owl_path = os.path.join(owl_path, 'NS')
            file_prefix = f'{scene_name}_NS'
        os.makedirs(owl_path, exist_ok=True)
        for k, gsg in self.GSGMgr.sceneMap.get_gsg_history().items():
            ou.write_owl_from_gsg({k:gsg},
                                  file_name=f'{file_prefix}_T{k}.owl',
                                  dir_path=owl_path)  # only_now 안쓰면 한 파일에 모두 씀
        ###########################

        ####### GSG to json #######
        gsg_path = os.path.join(dir_path, 'gsg', scene_name)
        os.makedirs(gsg_path, exist_ok=True)
        if seed is not None:
            file_prefix = f'{scene_name}_S{seed}'
        else:
            file_prefix = f'{scene_name}_NS'
        with open(os.path.join(gsg_path, f'{file_prefix}.json'), 'w') as f:
            json_db = {}
            json_db['mode'] = 'dynamic'
            json_db['seed'] = seed
            json_db['scene_name'] = scene_name
            json_db['gsg_history'] = self.GSGMgr.sceneMap.get_gsg_history()
            #self.GSGMgr.sceneMap.print_dict(json_db) # numpy는 저장이 안되서, numpy 있나 확인하는 용임
            # if self.record_action:
            #     json_db['actions'] = self.action_history
            # else:
            #     json_db['actions'] = []
            json.dump(json_db, f, indent='\t')
        print('save GSG file ({})'.format(os.path.join(gsg_path, f'{file_prefix}.json')))
        ###########################

    @pyqtSlot()
    def run_action_history_BT(self):
        if self.ui.check_record_action.isChecked():
            self.action_history = []
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        ah_path, _ = QFileDialog.getOpenFileName(self, "Select Action History File (Open)",
                                                              "./datasets/action_history",
                                                              "Json Files (*.json)", options=options)
        if ah_path == '':  # 취소할 경우
            return

        # with open(ah_path, 'r') as f:
        #     action_history = json.load(f)
        # self.init_reset()
        # self.thorCtrl.set_scene(action_history['scene_name'])
        #
        # for action_list in action_history['actions']:
        #     action = action_list[0]
        #     if action in ['pickup', 'put', 'open', 'close', 'teleport']:
        #         target = action_list[1]
        #         eval('self.action_{}({})()'.format(action, target))
        #     else:
        #         eval('self.action_{}()'.format(action))
        print('run action history')
        self.run_action_history(ah_path)

    @pyqtSlot()
    def run_action_histories_BT(self, use_seed=True):

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        ah_path = QFileDialog.getExistingDirectory(self, "Select Action Histories Folder (Open)",
                                                 "./datasets",
                                                 QFileDialog.ShowDirsOnly)
        if ah_path == '':  # 취소할 경우
            return
        print('start run actions...')
        st = time.time()
        action_histories = os.listdir(ah_path)
        action_histories = [ah for ah in action_histories if ah.split('.')[-1] == 'json'] # 확장자가 json인 파일만 추림
        action_histories.sort()

        self.GSGMgr.set_use_history(True)
        for i, file_name in enumerate(action_histories):
            print('run action histories... \'{}\' ({}/{}) '.format(file_name, i+1, len(action_histories)))
            file_path = os.path.join(ah_path, file_name)
            if use_seed:
                for seed in range(10):
                    self.run_action_history(file_path, seed)
                    self.save_results(seed)
            else:
                self.run_action_history(file_path)
                self.save_results()

        self.GSGMgr.set_use_history(False)
        sp = time.time() - st
        print(f'\ndone [{int(sp//3600)}h {int(sp%3600//60)}m {int(sp%60)}s]\n')

    def run_action_history(self, path, seed=None):
        with open(path, 'r') as f:
            action_history = json.load(f)
        if not self.thorCtrl.is_powered():
            self.init_start()
        self.action_fail_mode=False
        self.init_reset()
        self.thorCtrl.set_scene(action_history['scene_name'], seed=seed)

        for actions in action_history['actions']:
            if isinstance(actions, dict):
                action = actions['action']
                success = actions['success']
            else:
                action = actions[0]
                success = True

            if action in ['pickup', 'put', 'open', 'close', 'teleport']:
                if isinstance(actions, dict):
                    target = actions['target_object']
                else:
                    target = actions[1]

                # pickup/put 행동 가능한지 체크하고, 안되면 대체 혹은 넘김
                if action == 'pickup':
                    vis_objs = self.thorCtrl.get_visible_objects()
                    possible_idx = [i for i, obj in enumerate(vis_objs) if obj['pickupable']]
                    if len(possible_idx) == 0: # action 못하면 그냥 넘어감
                        continue

                    if target not in possible_idx:
                        target = possible_idx[0] # random.choice(possible_idx)
                    #print(vis_objs[target])
                    #print([obj for i, obj in enumerate(vis_objs) if obj['pickupable']])
                elif action == 'put':
                    if self.GSGMgr.sceneMap.get_agent_object() is None:
                        continue

                    vis_objs = self.thorCtrl.get_visible_objects()
                    possible_idx = [i for i, obj in enumerate(vis_objs) if obj['receptacle']]
                    if len(possible_idx) == 0: # action 못하면 그냥 넘어감
                        continue
                    if target not in possible_idx:
                        target = possible_idx[0] # random.choice(possible_idx)
                try:
                    eval(f'self.action_{action}({target}, success={success})()')
                except Exception as e:
                    print('error:', action, target, e)
                    pass
            else:
                eval(f'self.action_{action}(success={success})')

    def log_add_log(self, text): # 로그에 추가
        self.ui.tb_scene_log.append(text)

    def log_clear(self): # 로그 지우기
        self.ui.tb_scene_log.clear()

    @pyqtSlot()
    def log_reset(self):
        self.log_clear()

    @pyqtSlot()
    def log_save(self):
        if not os.path.isdir('log'):
            os.mkdir('log')
        log_file = open('log/ai2thor_log.txt', 'w')
        log_file.write(self.ui.tb_scene_log.toPlainText())
        log_file.close()

    @pyqtSlot()
    def recorder_dbFile_open(self): # select 버튼 클릭 시, 이벤트 함수
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.object_fileName, _ = QFileDialog.getSaveFileName(self, "Select Object Log File (Create/Open)", "thor_DB.json",
                                                  "Json Files (*.json)", options=options)
        if self.object_fileName:
            self.is_object_file_opened = True
            name_token = self.object_fileName.split('/')
            if len(name_token) >= 2:
                self.ui.dbFile_name.setText(os.path.join('...', name_token[-2], name_token[-1]))
            else:
                self.ui.dbFile_name.setText(self.object_fileName)

    @pyqtSlot()
    def recorder_imageDir_open(self): # select 버튼 클릭 시, 이벤트 함수
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.image_dirName = QFileDialog.getExistingDirectory(self, "Select Image Log Directory", "image", options=options)
        if self.image_dirName:
            self.is_image_dir_opened = True
            name_token = self.image_dirName.split('/')
            if len(name_token) >= 2:
                self.ui.imageDir_name.setText(os.path.join('...', name_token[-2], name_token[-1]))
            else:
                self.ui.imageDir_name.setText(self.image_dirName)

    @pyqtSlot()
    def recorder_record(self): # 레코드 버튼 눌렸을 때, 이벤트 함수
        #self.visual_minimize = True # GUI 결과 보여주기 멈추기 (최적화) ##
        # 이미지 폴더 자동 설정 (설정 안되있을 경우만)
        if self.image_dirName is None:
            #self.image_dirName = './images'
            self.image_dirName = '/media/ailab/D/ai2thor/images'
            self.depth_image_dirName = '/media/ailab/D/ai2thor/depth_images'
        if not os.path.isdir(self.image_dirName):
            os.mkdir(self.image_dirName)
        if not os.path.isdir(self.depth_image_dirName):
            os.mkdir(self.depth_image_dirName)
        # json 파일 패스 자동 설정
        if self.object_fileName is None:
            #self.object_fileName = './thor_DB.json'
            self.object_fileName = '/media/ailab/D/ai2thor/thor_DB.json'
        self.thorDB.load_json(self.object_fileName)


        self.is_recording = True
        self.ui.bt_record.setEnabled(False)  # 제거요망 (stop으로 바꾸기)

        self.ui.bt_pause.setEnabled(True)  # 제거요망
        self.ui.bt_save.setEnabled(True)  # 제거요망

        # 저장될 공간 선택 버튼 비활성화
        self.ui.bt_imageOpen.setEnabled(False)
        self.ui.bt_dbOpen.setEnabled(False)

        self.ui.bt_record.setText('recording...')

    @pyqtSlot()
    def recorder_pause(self): # 제거요망 (ui 파일에서 종속성을 없애야 지울 수 있음)
        if self.ui.bt_pause.isChecked():
            self.is_recording = True
        else:
            self.is_recording = False

    @pyqtSlot()
    def recorder_save(self): # 제거요망 (ui 파일에서 종속성을 없애야 지울 수 있음)
        self.is_recording = False
        self.visual_minimize = False  # GUI 결과 보여주기 (최적화 해제) ##

        # 이미지는 동적으로 저장되므로, 따로 처리 없음
        '''
        if self.is_object_file_opened and len(self.object_log) > 0:
            # object 로그 저장
            with open(self.object_fileName, 'w') as file:
                json.dump(self.object_log, file, indent='\t')

        if self.is_action_file_opened and len(self.action_log) > 0:
            # action 로그 저장
            with open(self.action_fileName, 'w') as file:
                json.dump(self.action_log, file, indent='\t')
        '''

        # 파일/폴더 open 버튼 초기화 (현재의 파일/폴더 명은 그대로 기억)
        self.ui.bt_dbOpen.setEnabled(True)
        self.ui.bt_imageOpen.setEnabled(True)

        # pause 상태 초기화 # 제거요망
        if self.ui.bt_pause.isChecked(): # 제거요망
            self.ui.bt_pause.setChecked(False) # 제거요망
        self.ui.bt_pause.setEnabled(False) # 제거요망

        # record 상태 초기화
        self.ui.bt_record.setEnabled(True)
        self.ui.bt_record.setText('record')

        # save 상태 초기화 # 제거요망
        self.ui.bt_save.setEnabled(False) # 제거요망

        # log 초기화 # 제거요망
        #self.object_log.clear() # 제거요망

        self.thorDB.save_json(self.object_fileName)

    @pyqtSlot()
    def recorder_autorecord(self): # Auto Record 버튼 눌렀을 때
        # 바로 버튼 눌러서 시작 가능
        if not self.thorCtrl.is_powered():
            self.init_start()
        self.thorCtrl.set_scene('FloorPlan2') # 1에는 tabletop이 있는데, annotation이 안되어있음.

        # 이미지 폴더 자동 설정 (설정 안되있을 경우만)
        if self.image_dirName is None:
            #self.image_dirName = './images'
            self.image_dirName = '/media/ailab/D/ai2thor/images'
        if not os.path.isdir(self.image_dirName):
            os.mkdir(self.image_dirName)


        if self.object_fileName is None:
            #self.object_fileName = './thor_DB.json'
            self.object_fileName = '/media/ailab/D/ai2thor/thor_DB.json'
        self.thorDB.load_json(self.object_fileName)

        self.auto_recording = True
        self.ui.bt_auto_record.clicked.disconnect()
        self.ui.bt_auto_record.clicked.connect(self.recorder_stop_autorecord)
        self.ui.bt_auto_record.setText('Stop Record')

        self.ui.bt_record.setEnabled(False)  # record 버튼 비활성화
        self.ui.bt_imageOpen.setEnabled(False) # 이미지 폴더 선택 비활성화
        self.ui.bt_dbOpen.setEnabled(False)  # json 파일 선택 비활성화

        self.visual_minimize = True # GUI 결과 보여주기 멈추기 (최적화)
        threading.Thread(target=self.recorder_run_autorecord, args=(self.thorDB.get_next_id()-1,)).start()

    def recorder_run_autorecord(self, count=0):
        #move_func = [self.action_go, self.action_go, self.action_go,
        #             self.action_go_left, self.action_go_right,
        #             self.action_left, self.action_left, self.action_right] # 너무 벽에서만 맴돌아서 left하나 추가함
        go_side_func = [self.action_go_left, self.action_go_right]
        rotate_func = [self.action_left, self.action_right]
        look_func = [self.action_up, self.action_down]
        record_term = 3 # 5번 action : 1번 저장
        term_count = 0 # %5 사이클
        goal_instance_number = 40000

        target_scene_number = list(range(2, 12)) # 2번 방부터 10번 방까지
        current_scene_idx = 0  # target_scene_number에서 인덱스 번호
        process_all_time = 0

        store_step = 1000
        log_step = 100
        NO_OBJECTS_THAN_NO_STORE = True # 물체 없으면 저장 안함
        PRINT_ACTION = False # action 출력
        look_value = 0 # -2 최하단, -1 하단, 0 중단, 1 상단

        while True:
            if not self.auto_recording: # stop 버튼 누르면 탈출함
                break
            st = time.time()

            # 가능한 행동 리스트 불러오기
            vis_objects = self.thorCtrl.get_visible_objects()
            inv_obj = self.thorCtrl.get_inventory()

            openable_objects, closeable_objects = [], []
            pickupable_objects, receptacle_objects = [], []
            # onable_objects, offable_objects = [], [] # on/off 미구현

            for idx, obj in enumerate(vis_objects):
                if obj['openable']:
                    if obj['isopen']:
                        closeable_objects.append(idx)
                    else:
                        openable_objects.append(idx)
                elif obj['pickupable'] and inv_obj is None:
                    pickupable_objects.append(idx)
                elif obj['receptacle'] and inv_obj is not None:
                    if len(obj['receptacleObjectIds']) < obj['receptacleCount']:
                        receptacle_objects.append(idx)
            vote = ['GO'] * 4 + ['GO_SIDE'] + ['ROTATE'] + ['LOOK']
            if len(openable_objects) > 0:
                vote += ['OPEN'] * 2
            if len(closeable_objects) > 0:
                vote += ['CLOSE'] * 2
            if len(pickupable_objects) > 0:
                vote += ['PICKUP']
            if len(receptacle_objects) > 0:
                vote += ['PUT']

            if look_value == 1: # agent가 위를 보면 강제로 내림
                self.action_down()
                look_value -= 1
            elif look_value == -2: # agnet가 땅을 보면 강제로 올림
                self.action_up()
                look_value += 1
            else:
                # 행동 랜덤으로 선택
                win = random.choice(vote)
                if win == 'GO':
                    self.action_go()
                    isSuccess, _, _ = self.thorCtrl.get_previous_action_result()
                    if not isSuccess:
                        random.choice(rotate_func * 2 + go_side_func)()
                elif win == 'GO_SIDE':
                    random.choice(go_side_func)()
                elif win == 'ROTATE':
                    random.choice(rotate_func)()
                elif win == 'LOOK':
                    isSuccess = False
                    while not isSuccess: # 만약에 맨 위를 보고있으면, 아래를 보는 행동을 시도. 반대도 마찬가지.
                        selected_func = random.choice(look_func)
                        if selected_func == self.action_up:
                            look_value += 1
                        elif selected_func == self.action_down:
                            look_value -= 1
                        else:
                            raise Exception("long func in LOOK vote")
                        selected_func()
                        isSuccess, _, _ = self.thorCtrl.get_previous_action_result()
                elif win == 'OPEN':
                    self.action_open(random.choice(openable_objects))()
                elif win == 'CLOSE':
                    self.action_close(random.choice(closeable_objects))()
                elif win == 'PICKUP':
                    self.action_pickup(random.choice(pickupable_objects))()
                elif win == 'PUT':
                    self.action_put(random.choice(receptacle_objects))()
                if PRINT_ACTION:
                    print("step: {}, data_id: {}, action: {}".format(count, self.thorDB.get_next_id(), win))

            isSuccess, _, _ = self.thorCtrl.get_previous_action_result()
            if not isSuccess: # 행동 실패면 저장 안함, count도 안셈
                process_all_time += time.time() - st
                continue
            vis_objects = self.thorCtrl.get_visible_objects() # 움직인 후의 frame 정보

            term_count = (term_count + 1) % record_term
            if term_count != 0: # record_term 마다 저장함. 나머진 저장 안함
                process_all_time += time.time() - st
                continue
            if NO_OBJECTS_THAN_NO_STORE and len(vis_objects) == 0: # 물체 없으면 저장 안함
                process_all_time += time.time() - st
                continue

            # scene 상태 저장
            scene_info = self.recorder_get_scene_annotation(save_image=True)  # scene annotation 부여 및 이미지 저장
            self.thorDB.add(scene_info)

            # 속도 및 남은 시간 출력
            if count % log_step == 0:
                if count != 0:
                    process_all_time /= log_step
                else:
                    process_all_time += time.time() - st
                process_time = str(process_all_time).split('.')
                process_time = float(process_time[0] + '.' + process_time[1][:3])
                left_time = process_time * (goal_instance_number - count)
                print('[{}/{}] {}s per oneshot, {}h {}m {}s left.'.format(count, goal_instance_number, process_time,
                                                                          int(left_time//3600), int((left_time%3600)//60), int(left_time%60)))
                process_all_time = 0.
            # 일정 스텝 이상이면 scene 랜덤 변경
            #if count % 200 == 0 and count > 0:
            #    self.thorCtrl.set_scene(random.choice(['FloorPlan'+str(i) for i in range(2, 21)])) # 1은 table missing
            #    print('\nscene transition\n')
            # 일정 스텝 이상이면 다음 scene으로 넘김
            if count % 1000 == 0 and count > 0:
                current_scene_idx = (current_scene_idx + 1) % len(target_scene_number)
                print('\nscene transition')
                self.thorCtrl.set_scene('FloorPlan' + str(target_scene_number[current_scene_idx])) # scene name 출력됨
                print('')

            # 일정 스텝마다 json 저장
            if count % store_step == 0:
                self.thorDB.save_json(self.object_fileName)
                print('== save DB ==')


            # 종료 조건
            if count >= goal_instance_number - 1:
                self.thorDB.save_json(self.object_fileName)
                print('== save DB ==')
                break

            # step count
            count += 1

            self.ui.la_record_count.setText(str(count))
            self.ui.la_db_len.setText(str(self.thorDB.next_data_id))
            process_all_time += time.time()-st

    def recorder_stop_autorecord(self):  # Auto Record 취소 버튼 눌렀을 때
        self.auto_recording = False # thread 종료 조건 활성화
        time.sleep(0.5)  # thread 종료를 기다림 [segmantation falut (core dumped) 방지]

        self.ui.bt_auto_record.clicked.disconnect()
        self.ui.bt_auto_record.clicked.connect(self.recorder_autorecord)
        self.ui.bt_auto_record.setText('Auto Record')

        self.ui.bt_record.setEnabled(True)  # record 버튼 활성화
        self.ui.bt_imageOpen.setEnabled(True) # 이미지 폴더 선택 활성화
        self.ui.bt_dbOpen.setEnabled(True)  # json 파일 선택 활성화

        self.visual_minimize = False  # GUI 최적화 종료
        self.thorDB.save_json(self.object_fileName)
        print('== save DB ==')

    def recorder_get_scene_annotation(self, save_image=True):
        next_data_id = self.thorDB.get_next_id()  # .다음 데이터 아이디 부여

        image_name = str(next_data_id) + '_' + self.thorCtrl.scene_name + '.jpg'
        depth_image_name = str(next_data_id) + '_' + self.thorCtrl.scene_name + '_depth.jpg'
        opencv_frame = self.thorCtrl.get_image('opencv')  # BGR
        depth_frame = self.thorCtrl.get_depth_image()

        if save_image:
            # 이미지 저장
            cv2.imwrite(os.path.join(self.image_dirName, image_name), opencv_frame)
            cv2.imwrite(os.path.join(self.depth_image_dirName, depth_image_name), depth_frame)

        vis_objs = self.thorCtrl.get_visible_objects()
        isSuccess, previous_action, errorMessage = self.thorCtrl.get_previous_action_result()
        relations, global_relations = au.get_relations(objects=vis_objs,
                                    objects_id=self.thorCtrl.object_global_ids,
                                    agent_info=self.thorCtrl.event.metadata['agent'], get_global=True)

        #attributes = au.get_attributes(objects=vis_objs, objects_id=self.thorCtrl.object_global_ids,
        #                                            image=opencv_frame)  # BGR 영상을 줌
        attributes = au.get_attributes(objects=vis_objs, objects_id=self.thorCtrl.object_global_ids,
                                       image=opencv_frame, instance_masks=self.thorCtrl.get_event().instance_masks)  # BGR 영상을 줌
        agent_info = {}
        agent_data = self.thorCtrl.get_event().metadata['agent']
        agent_info['global_position'] = agent_data['position']
        agent_info['global_rotation'] = agent_data['rotation']

        scene_info = {}
        scene_info['data_id'] = next_data_id
        scene_info['image_file'] = image_name
        scene_info['depth_file'] = depth_image_name
        scene_info['scene'] = self.thorCtrl.scene_name
        scene_info['action'] = previous_action
        scene_info['is_success'] = 1 if isSuccess else 0
        scene_info['error_msg'] = errorMessage
        scene_info['visible_object'] = []
        scene_info['relation'] = relations
        scene_info['global_relation'] = global_relations
        scene_info['inventory'] = {}
        scene_info['agent'] = agent_info

        for idx, vis_obj in enumerate(vis_objs):
            info = {}
            info['id'] = self.thorCtrl.object_global_ids[vis_obj['objectId']]
            info['obj_class'] = au.obj_s2i[vis_obj['objectType']]
            info['name'] = vis_obj['name']
            info['objectId'] = vis_obj['objectId']
            info['global_position'] = vis_obj['position']
            info['global_rotation'] = vis_obj['rotation']
            info['global_bounds3D'] = vis_obj['bounds3D'] # 값 6개인 리스트
            info['distance'] = vis_obj['distance'] # value
            if 'bb' in vis_obj:
                info['bounding_box'] = list(map(int, vis_obj['bb']))
            else:
                info['bounding_box'] = [-1, -1, -1, -1]  # 정의 안되어있을 때
            if vis_obj['openable']:
                info['open_state'] = 1 if vis_obj['isopen'] else 2
            else:
                info['open_state'] = 0
            info['color'] = attributes[info['id']]['color']
            info['color_hsv'] = attributes[info['id']]['hsv']

            scene_info['visible_object'].append(info)
        inventory = self.thorCtrl.get_inventory()
        if not inventory is None:
            scene_info['inventory']['obj_class'] = inventory['objectType']
            scene_info['inventory']['objectId'] = inventory['objectId']
        else:
            scene_info['inventory']['obj_class'] = 'NONE'
            scene_info['inventory']['objectId'] = 'NONE'

        return scene_info

    def clear_views(self):
        if self.use_logDialog:
            self.LogDialog.clear_log()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Up or e.key() == Qt.Key_8:
            self.action_go()
        elif e.key() == Qt.Key_Down or e.key() == Qt.Key_5:
            self.action_back()
        elif e.key() == Qt.Key_Left or e.key() == Qt.Key_4:
            self.action_left()
        elif e.key() == Qt.Key_Right or e.key() == Qt.Key_6:
            self.action_right()
        elif e.key() == Qt.Key_7:
            self.action_go_left()
        elif e.key() == Qt.Key_9:
            self.action_go_right()
        elif e.key() == Qt.Key_Minus:
            self.action_up()
        elif e.key() == Qt.Key_Plus:
            self.action_down()
        elif e.key() == Qt.Key_R:
            self.init_reset()
        elif e.key() == Qt.Key_Escape:
            self.close()
        elif e.key() == Qt.Key_Z:
            self.ui.check_show_bbox.nextCheckState()
        elif e.key() == Qt.Key_X:
            self.ui.check_show_graph.nextCheckState()
        elif e.key() == Qt.Key_C:
            self.ui.check_use_gsg.nextCheckState()
        elif e.key() == Qt.Key_V:
            if self.ui.check_use_gsg.isChecked():
                self.ui.check_show_gsg.nextCheckState()
        elif e.key() == Qt.Key_S:
            if self.thorCtrl.is_powered():
                self.init_end()
            else:
                self.init_start()
        elif e.key() == Qt.Key_C:
            self.clear_views()


class SubDialog(QDialog):
    def __init__(self, parent, title=None, width=500, height=300, x=0, y=0):
        super(SubDialog, self).__init__(parent)
        self.ui = uic.loadUi("sub_dialog.ui", self)

        #self.setBaseSize(width, height)
        self.setFixedSize(width, height)
        self.move(x, y)
        self.imageLabel = self.ui.imageLabel
        self.imageLabel.setFixedSize(width, height)
        self.width = width # 안씀..
        self.height = height # 안씀..
        if title is not None:
            self.set_title(title)

        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtCore.Qt.white)
        self.setPalette(p)


    def set_title(self, title):
        self.setWindowTitle(title)

    def show_image(self, filePath):
        # 영상 파일을 읽어서, 다이얼로그에 출력
        pixxap = QPixmap(filePath)
        if pixxap.width() > self.imageLabel.width()*0.7 \
            or pixxap.height() > self.imageLabel.height()*0.7:
            pixmap = QPixmap(filePath).scaled(self.imageLabel.width(), self.imageLabel.height(),
                                              QtCore.Qt.KeepAspectRatio)
        else:
            pixmap = QPixmap(filePath).scaled(self.imageLabel.width()*0.7, self.imageLabel.height()*0.7,
                                              QtCore.Qt.KeepAspectRatio)

        self.imageLabel.setPixmap(pixmap)

    def set_size(self, width, height):
        self.setFixedSize(width, height)
        self.imageLabel.setFixedSize(width, height)

    def clear(self):
        self.imageLabel.clear()
        #self.clearMask()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Ai2Thor_GUI()
    sys.exit(app.exec())