# -*- coding:utf-8 -*-

import numpy as np
import torch
import json

import sys
import os

# sys.path.append(os.path.join(os.getcwd(), 'pytorch_yolo_v3'))
# from pytorch_yolo_v3.darknet import Darknet
# from pytorch_yolo_v3 import util_yolo

sys.path.append(os.path.join(os.getcwd(), 'PyTorch_YOLOv3'))
from PyTorch_YOLOv3.models import *
from PyTorch_YOLOv3.yolo_utils.utils import *

from iqa_util import bb_util
from iqa_util import py_util
from thor_utils import annotation_util as au

sys.path.append(os.path.join(os.getcwd(), 'ARNet_ai2thor/faster_rcnn'))
sys.path.append(os.path.join(os.getcwd(), 'ARNet_ai2thor'))
sys.path.append(os.path.join(os.getcwd(), 'ARNet_ai2thor/faster_rcnn/fast_rcnn'))
sys.path.append(os.path.join(os.getcwd(), 'ARNet_ai2thor/faster_rcnn/roi_pooling'))
sys.path.append(os.path.join(os.getcwd(), 'ARNet_ai2thor/faster_rcnn/utils'))
sys.path.append(os.path.join(os.getcwd(), 'ARNet_ai2thor/faster_rcnn/roi_pooling/_ext/roi_pooling'))

from ARNet_ai2thor.faster_rcnn.AttNet4 import AttNet  # AttNet4 사용
from ARNet_ai2thor.faster_rcnn.TransferNet_angle import TransferNet  # TransferNet6 사용 [1: 논문, 3: 이미지, prior 사용, 4: depth도 사용, 6:상대값]
from ARNet_ai2thor.faster_rcnn import network
from ARNet_ai2thor.faster_rcnn.fast_rcnn.config import cfg

from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy.random as npr
import cv2
import copy
from matplotlib import pyplot as plt

import time
CHECK_TIME = True

class DNNManager():
    def __init__(self):
        # 학습된 모델들을 불러오고 적용시킴
        pass

    def load_models(self):
        # object detector (yolo)
        self.obj_net = None
        self.resolution = 600

        '''
        # old version
        self.obj_net = Darknet('./models/yolov3-thor.cfg')
        self.obj_net.cuda()
        self.obj_net.eval()
        self.obj_net.load_weights('./models/yolov3-thor_final.weights')
        self.reso = 416
        self.obj_net.net_info["height"] = self.reso
        with open('./models/iqa.names', 'r') as f:
            self.iqa_obj_label = f.read().split('\n')
        '''
        # new version
        self.obj_net = Darknet('./models/yolov3-thor.cfg')
        self.obj_net.load_weights('./models/yolov3-thor_final.weights')
        self.obj_net.cuda()
        self.obj_net.eval()
        self.reso = 608
        with open('./models/iqa.names', 'r') as f:
            self.iqa_obj_label = f.read().split('\n')

        # attribute predictor
        self.att_net = AttNet()
        self.att_net.cuda()
        self.att_net.eval()
        self.att_net.load_state_dict(torch.load('ARNet_ai2thor/output/A4_AttNet_best.state')) # ROI는 pickle 형태로 저장 불가하다함

        #normalize = transforms.Normalize(mean=[0.352, 0.418, 0.455], std=[0.155, 0.16, 0.162])
        #normalize = transforms.Normalize(mean=[0.319, 0.396, 0.452],
        #                                 std=[0.149, 0.155, 0.155])  # 새 DB 3507개 (181106)에 바꿈
        self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

        # transfer
        #self.trans_net = torch.load('ARNet_ai2thor/output/TransferNet.pt')
        #self.trans_net.cuda()
        #self.trans_net.eval()
        prior_knowledge = json.load(open('prior_knowledge.json'))
        self.prior_knowledge = prior_knowledge['objects']
        self.trans_net = TransferNet(use_angle=True)
        self.trans_net.cuda()
        self.trans_net.eval()
        self.trans_net.load_state_dict(torch.load('ARNet_ai2thor/output/TransferNet_angle_db_angle.state')) # ROI는 pickle 형태로 저장 불가하다함
        self.change_agent_rotation_scale = {0: 0., 90: 0.33, 180: 0.66, 270: 0.99,
                                            0: 0., 1: 0.33, 2: 0.66, 3: 0.99}  # 0~270을 0~1로 변환
        self.change_agent_angle = {60:0., 30:1., 0:2., -30:3.}
        caa = copy.copy(self.change_agent_angle)
        for k, v in caa.items(): # 가끔 59 등의 값이 입력됨.
            self.change_agent_angle[k - 1] = v
            self.change_agent_angle[k + 1] = v


        # relationship predictor
        self.rel_net = torch.load('ARNet_ai2thor/output/(R2_thorDBv2) RelNet.pt')
        self.rel_net.cuda()
        self.rel_net.eval()

        print('models loaded')

    def _load_yolo_from_darknet(self):
        import darknet as dn
        # object detector (yolo)
        dn.set_gpu(0)
        self.obj_net = dn.load_net(py_util.encode('./models/yolov3-thor.cfg'), \
                               py_util.encode('./models/yolov3-thor_final.weights'), 0)
        self.obj_net_meta = dn.load_meta(py_util.encode('./models/thor.data'))

    # YOLOv3 darknet 버전 (다크넷과 pytorch를 동시에 사용하면 충돌 문제 발생)
    def detect_objects_from_darknet(self, image):
        import darknet as dn
        confidence_threshold = 0.7

        results = dn.detect_numpy(self.obj_net, self.obj_net_meta, image, confidence_threshold)

        if len(results) > 0:
            classes, scores, boxes = zip(*results)
        else:
            classes = []
            scores = []
            boxes = np.zeros((0, 4))
        boxes = np.array(boxes)
        scores = np.array(scores)
        classes = np.array([py_util.decode(cls) for cls in classes])
        inds = np.where(np.logical_and(scores > confidence_threshold,
                                       np.min(boxes[:, [2, 3]], axis=1) > .01 * image.shape[0]))[0]
        used_inds = []
        for ind in inds:
            if classes[ind] in set(au.object_label):
                used_inds.append(ind)
        inds = np.array(used_inds)
        if len(inds) > 0:
            classes = np.array(classes[inds])
            boxes = boxes[inds]
            if len(boxes) > 0:
                boxes = bb_util.xywh_to_xyxy(boxes.T).T
            boxes *= np.array([self.resolution * 1.0 / image.shape[1],
                               self.resolution * 1.0 / image.shape[0]])[[0, 1, 0, 1]]
            boxes = np.clip(np.round(boxes), 0, np.array([self.resolution,
                                                          self.resolution])[[0, 1, 0, 1]]).astype(np.int32)
            scores = scores[inds]
        else:
            boxes = np.zeros((0, 4))
            classes = np.zeros(0)
            scores = np.zeros(0)

        '''
        # 결과 예시
        boxes.shape : (5, 4)
        scores.shape : (5,)
        classes.shape : (5,)
        boxes : [[  5 202 128 600]
         [139 469 195 598]
         [269 404 283 430]
         [298 446 314 472]
         [240 443 249 457]]
        scores : [0.99959159 0.9975611  0.98745704 0.98426211 0.95225739]
        classes : ['Fridge' 'Bread' 'Mug' 'Potato' 'Tomato']
        '''
        return boxes, scores, classes

    # pytorch 버전으로 변형한 YOLOv3 사용
    def detect_objects(self, image, confidence_threshold=0.8, nms_thesh=0.4):
        if CHECK_TIME:
            st = time.time()
        input_shape = image.shape
        #img = np.array(image)  #
        img = cv2.resize(image, (self.reso, self.reso))
        #Image.fromarray(np.uint8(image)).resize((self.reso, self.reso)).save('pil.jpg')
        #img = np.asarray(Image.open('pil.jpg'))

        #image = img[:,:,::-1].transpose((2,0,1))  #
        image = img.transpose((2, 0, 1))  #
        image = image[np.newaxis,:,:,:]/255.0  # 원래 -1이 붙어서, RGB 순서를 바꿨는데 이게 치명적인 문제점이었음...
        image = network.np_to_variable(np.array(image), is_cuda=True)
        #print('image size:',image.size())
        if CHECK_TIME:
            sub_st = time.time()
        #print(image)
        results = self.obj_net(image)
        results = non_max_suppression(results, 80, confidence_threshold, nms_thesh)[0] # 영상 하나
        if CHECK_TIME:
            sub_time = time.time()-sub_st
            print('[TIME] 1. run obj_net() : {}s'.format(str(sub_time)[:7]))
        if results is None:
            classes = np.array([])
            scores = np.array([])
            boxes = np.zeros((0, 4))
            return boxes, scores, classes
        results = results.cpu().data.numpy()
        classes = np.array([self.iqa_obj_label[int(obj[-1])] for obj in results])

        used_inds = [] # 실제 타겟 class만 필터링 (yolo가 다양한 물체들을 잡음)
        for ind in range(len(classes)):
            if classes[ind] in set(au.object_label):
                used_inds.append(ind)
        inds = np.array(used_inds)
        if len(used_inds) > 0:
            boxes = np.array(results[inds, 0:4])
            scores = np.array(results[inds, -2])
            classes = classes[inds]

            boxes *= np.array([input_shape[1] * 1.0 / image.shape[3],
                               input_shape[0] * 1.0 / image.shape[2]])[[0, 1, 0, 1]]
            boxes = np.clip(np.round(boxes), 0, np.array([input_shape[1],
                                                          input_shape[0]])[[0, 1, 0, 1]]).astype(np.int32)
        else:
            classes = np.array([])
            scores = np.array([])
            boxes = np.zeros((0, 4))
        # for i in range(len(classes)):
        #    print('idx:{}, class:{}, bbox:{}, score:{}'.format(i, classes[i], boxes[i], scores[i]))
        '''
        # 결과 예시
        boxes.shape : (5, 4)
        scores.shape : (5,)
        classes.shape : (5,)
        boxes : [[  5 202 128 600]
         [139 469 195 598]
         [269 404 283 430]
         [298 446 314 472]
         [240 443 249 457]]
        scores : [0.99959159 0.9975611  0.98745704 0.98426211 0.95225739]
        classes : ['Fridge' 'Bread' 'Mug' 'Potato' 'Tomato']
        '''
        if CHECK_TIME:
            total_time = time.time()-st
            print('[TIME] 1. run detect_objects() : {}s'.format(str(total_time)[:7]))
            print('[TIME] \tpre/pos processing time : {}s'.format(str(total_time-sub_time)[:7]))
        return boxes, scores, classes

    # 검토완료 # label은 안씀 (label이 잘못 붙여진 경우에 오류날 것 같아서)
    # 봐줄만한 성능임 (왠만해선 맞음)
    def predict_attributes(self, image, boxes, labels):
        if CHECK_TIME:
            st = time.time()
        ## model input
        # forward(self, im_data, im_info, gt_objects, gt_colors=None, gt_open_states=None, gt_relationships=None,
        #        use_beam_search=False, graph_generation=False):
        ## model output
        #return (color_prob, open_state_prob, object_rois), None

        input_objects = np.zeros((len(boxes), 5))
        input_objects[:, :4] = np.array(boxes)
        input_objects[:, 4] = np.array(labels)  # background는 고려안함
        #input_boxes = [obj['bounding_box'] + [obj['obj_class']+1] for obj in objects] # +1은 background를 위해

        # image processing
        target_scale = cfg.TRAIN.SCALES[npr.randint(0, high=len(cfg.TRAIN.SCALES))]
        #cv2.imwrite('./1.jpg', image)
        #image = cv2.imread('./1.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(type(image))
        #image = image.astype(float) / 255.
        img, im_scale = self._image_resize(image, target_scale, cfg.TRAIN.MAX_SIZE)
        # bbox 크기 조정 (resize된 이미지에 따라.)
        input_objects[:, :4] *= im_scale
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        img = img.unsqueeze(0)
        #im_data = Variable(img.cuda(), volatile=True)  ## 추가 !!
        im_data = Variable(img.cuda())

        if CHECK_TIME:
            sub_st = time.time()
        try:
            preds = self.att_net(im_data, None, input_objects)  # None은 img_info 자리
        except:
            print(input_objects)
            raise Exception('attnet error')
        if CHECK_TIME:
            sub_time = time.time()-sub_st
            print('[TIME] 2. run att_net() : {}s'.format(str(sub_time)[:7]))

        color_prob, open_state_prob = preds
        color_prob = color_prob.data.cpu().numpy()
        open_state_prob = open_state_prob.data.cpu().numpy()

        colors = color_prob.argmax(1)
        open_states = open_state_prob.argmax(1)
        #print('output[color_prob]:', color_prob)
        #print('output[color]:', colors)
        #print('output[os_prob]:', open_state_prob)
        #print('output[os]:', open_states)
        if CHECK_TIME:
            total_time = time.time() - st
            print('[TIME] 2. run predict_attributes() : {}s'.format(str(total_time)[:7]))
            print('[TIME] \tpre/pos processing time : {}s'.format(str(total_time - sub_time)[:7]))
        return colors, open_states, color_prob, open_state_prob

    def transfer_to_global_map(self, image, depth, boxes, labels, agent):
        # TransferNet_angle version
        roi_images = torch.zeros((len(boxes), 3, 64, 64))
        roi_depths = torch.zeros((len(boxes), 1, 64, 64))

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        default_boxes = np.zeros((len(boxes), 6))

        for i, box in enumerate(boxes):
            r_image = copy.copy(image[box[1]:box[3], box[0]:box[2]])

            r_image = cv2.cvtColor(r_image, cv2.COLOR_BGR2RGB)
            r_image = cv2.resize(r_image, (64, 64))
            r_image = Image.fromarray(r_image)  # 0~1
            r_image = transform(r_image)  # convert to Tensor
            roi_images[i] = r_image

            r_depth = copy.copy(depth[box[1]:box[3], box[0]:box[2]])

            # 0~255 값이어야함. 확인할 것.
            r_depth = cv2.resize(r_depth, (64, 64))
            r_depth = r_depth.astype(np.uint8)
            default_boxes[i] = self._get_default_box(box[:4], r_depth)
            r_depth = Image.fromarray(r_depth)  # 0~1
            r_depth = transform(r_depth)  # convert to Tensor
            roi_depths[i] = r_depth

        input_boxes = torch.zeros((len(boxes), 5))
        input_boxes[:, 0:4] = torch.FloatTensor(boxes[:, :4])
        input_boxes[:, 4] = torch.FloatTensor(labels)

        input_agent = torch.zeros(5)  # (x, y, z, rotation, angle)
        input_agent[:3] = torch.FloatTensor(list(agent['position'].values()))
        input_agent[3] = torch.FloatTensor([self.change_agent_rotation_scale[agent['rotation']['y']]])
        input_agent[4] = torch.FloatTensor([self.change_agent_angle[round(agent['cameraHorizon'])]])
        input_agent = input_agent.view(1, -1)
        input_agent = input_agent.repeat(len(boxes), 1)

        roi_images = Variable(roi_images.cuda())  ## 추가 !!
        roi_depths = Variable(roi_depths.cuda())

        output, _, _ = self.trans_net(roi_images, roi_depths, input_boxes.numpy(), input_agent)

        output = output.data.cpu().numpy()
        default_box = default_boxes
        agent = input_agent.data.cpu().numpy()

        output2 = output + default_box
        # print(output.shape)

        results = np.zeros((len(output2), 6))
        for j in range(len(results)):
            rotate = str(agent[j][3])[:4]
            if rotate == '0.00' or rotate == '0.0':
                results[j, 0] = output2[j, 0] + agent[j][0]
                results[j, 2] = output2[j, 2] + agent[j][2]
            elif rotate == '0.33':
                results[j, 0] = output2[j, 2] + agent[j][0]
                results[j, 2] = -output2[j, 0] + agent[j][2]
            elif rotate == '0.66':
                results[j, 0] = -output2[j, 0] + agent[j][0]
                results[j, 2] = -output2[j, 2] + agent[j][2]
            elif rotate == '0.99':
                results[j, 0] = -output2[j, 2] + agent[j][0]
                results[j, 2] = output2[j, 0] + agent[j][2]
            else:
                assert False, f'error {rotate}'
            results[j, 1] = output2[j, 1] * 2 #+ agent[j][1]
            results[j, 3:] = output2[j, 3:]
        #print('labels', len(labels), labels)
        #print('results', len(results), results)
        return results


    def _get_default_box(self, box, depth):
        # x, y, z 값을 현재 에이전트의 중심과 얼마나 차이나는 지를 답으로함 # wd, h는 그대로.
        default_box = torch.zeros(6)

        default_box[0] = (((box[2] + box[0]) / 2) - self.resolution/2) / self.resolution  # x
        default_box[1] = (self.resolution - ((box[3] + box[1]) / 2)) / self.resolution  # y # 300은 가운데인데, agent 위치가 아님, 그래서 바닥을 기준

        default_box[2] = self._get_center_depth(depth)  # z: depth

        default_box[3] = (box[2] - box[0]) / self.resolution  # * math.log2(default_box[2]+1) # w
        default_box[4] = (box[3] - box[1]) / self.resolution  # * math.log2(default_box[2]+1) # h
        default_box[5] = (box[2] - box[0]) / self.resolution  # * math.log2(default_box[2]+1) # d: w와 동등

        default_box[:2] *= default_box[2]  # 거리에 따라 좁게 보이는 현상을 고려.

        return default_box

    def _get_center_depth(self, depth, range=0.6):
        # TransferNet8에서 사용. 센터(range)에 해당하는 부분의 평균 depth값을 추출
        w = depth.shape[1]
        h = depth.shape[0]
        min = (1-range)/2
        max = (1-range)/2 + range
        depth_value = depth[int(w*min):int(w*max), int(h*min):int(h*max)]

        return depth_value.mean()/50

    def transfer_to_global_map_back_(self, image, depth_map, boxes, labels, agent):
        if CHECK_TIME:
            st = time.time()
        ## model input (TransferNet6)
        # forward(self, objects, agent, gt_boxes_3d=None):
        # 1. boxes : (object#, 8[x, y, x2, y2, label, depth, size_w[PK], size_h[PK], size_z[PK]]) PK는 사전지식
        # 2. agent (x, y, z, rotation) #rotation은 0, 0.33, 0.66, 0.99 값

        ## model output
        # return fc3_feature, total_loss, losses

        # image processing
        target_scale = cfg.TRAIN.SCALES[npr.randint(0, high=len(cfg.TRAIN.SCALES))]
        #cv2.imwrite('./2.jpg', image)
        #image = cv2.imread('./2.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(type(image))
        # image = image.astype(float) / 255.
        img, im_scale = self._image_resize(image, target_scale, cfg.TRAIN.MAX_SIZE)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        img = img.unsqueeze(0)
        im_data = Variable(img.cuda())

        depth_map, depth_scale = self._image_resize(depth_map, target_scale, cfg.TRAIN.MAX_SIZE)
        #depth_map = Image.fromarray(depth_map, 'L')
        #depth_map = Image.fromarray(depth_map)
        #depth_map = self.transform(depth_map)

        # 2D bounding box (TransferNet6 버전)
        input_boxes = np.zeros((len(boxes), 9))
        input_boxes[:, :4] = np.array(boxes)  # TransferNet3 부터는 foward에서 600으로 나눔
        input_boxes[:, :4] *= im_scale  # bbox 크기 조정 (resize된 이미지에 따라.)
        input_boxes[:, 4] = np.array(labels)  # background가 0임

        print('depth_map')
        print(depth_map)
        depth_values = get_center_depth(depth_map, input_boxes[:, :4], range=0.6) # depth 계산
        input_boxes[:, 5] = depth_values
        #depth_map = depth_map.unsqueeze(0)
        #depth_map = Variable(depth_map.cuda())

        prior = np.array([self.prior_knowledge[au.obj_n2c[label]]['size_3d'] for label in labels])
        input_boxes[:, 6:] = prior
        print('input_boxes')
        print(input_boxes)
        #input_boxes = network.np_to_variable(np.array(input_boxes), is_cuda=True)

        # agent position, orientation
        agent_rotation = self.change_agent_rotation_scale[agent['rotation']['y']]
        input_agent = [agent['position']['x'], agent['position']['y'],
                       agent['position']['z'], agent_rotation]
        input_agent = network.np_to_variable(np.array(input_agent), is_cuda=True)
        input_agent = input_agent[np.newaxis, :]

        if CHECK_TIME:
            sub_st = time.time()
        # input_boxes : numpy, input_agent : Tensor
        # global_coor, _, _ = self.trans_net(input_boxes, input_agent)
        # im_data: Tensor, input_boxes: Tensor, input_agent: Tensor
        outputs, _, _ = self.trans_net(im_data, depth_map, input_boxes, input_agent)  # TransferNet3 버전용
        if CHECK_TIME:
            sub_time = time.time() - sub_st
            print('[TIME] 3. run trans_net() : {}s'.format(str(sub_time)[:7]))
        outputs = outputs.data.cpu().numpy() # [xz, y, wd]
        print('outputs')
        print(outputs)
        global_coor = np.zeros((len(outputs), 6))

        # 좌표 계산 (상대값 + 에이전트 위치) TransferNet6 전용
        if agent['rotation']['y'] == 0:
            global_coor[:, 0] = agent['position']['x'] + outputs[:, 0]
            global_coor[:, 2] = agent['position']['z'] + depth_values
        elif agent['rotation']['y'] == 90:
            global_coor[:, 0] = agent['position']['x'] + depth_values
            global_coor[:, 2] = agent['position']['z'] - outputs[:, 0]
        elif agent['rotation']['y'] == 180:
            global_coor[:, 0] = agent['position']['x'] - outputs[:, 0]
            global_coor[:, 2] = agent['position']['z'] - depth_values
        elif agent['rotation']['y'] == 270:
            global_coor[:, 0] = agent['position']['x'] - depth_values
            global_coor[:, 2] = agent['position']['z'] + outputs[:, 0]
        global_coor[:, 1] = agent['position']['y'] + outputs[:, 1]

        # prior 크기로 대체
        if agent['rotation']['y'] == 90 or agent['rotation']['y'] == 270: # z 축 기준이면 크기 돌림
            prior = prior[:, [2, 1, 0]]
        for i in range(len(global_coor)):  # prior 크기로 대체
            predict_size = outputs[i][2]  # 너비만 예측
            prior_size_1 = prior[i]
            prior_size_2 = prior_size_1[[2, 1, 0]]
            if abs(predict_size-prior_size_1[0]) < abs(predict_size-prior_size_2[0]):
                global_coor[i][3:] = prior_size_1
            else:
                global_coor[i][3:] = prior_size_2

        if CHECK_TIME:
            total_time = time.time() - st
            print('[TIME] 3. run transfer_to_global_map() : {}s'.format(str(total_time)[:7]))
            print('[TIME] \tpre/pos processing time : {}s'.format(str(total_time - sub_time)[:7]))
        print('global_coor')
        print(global_coor)
        return global_coor


    def transfer_to_global_map_back2(self, image, depth_map, boxes, labels, agent):
        if CHECK_TIME:
            st = time.time()
        ## model input
        # forward(self, objects, agent, gt_boxes_3d=None):
        # 1. bbox (x, y, x2, y2, distance, label) # OD 결과값 (학습 시에는 OD의 정답 bbox가 들어감)
        # 2. agent (x, y, z, rotation) #rotation은 0~3 값

        ## model output
        # return fc3_feature, total_loss, losses

        # image processing
        target_scale = cfg.TRAIN.SCALES[npr.randint(0, high=len(cfg.TRAIN.SCALES))]
        cv2.imwrite('./2.jpg', image)
        image = cv2.imread('./2.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(type(image))
        #image = image.astype(float) / 255.
        img, im_scale = self._image_resize(image, target_scale, cfg.TRAIN.MAX_SIZE)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        img = img.unsqueeze(0)
        #im_data = Variable(img.cuda(), volatile=True)  ## 추가 !!
        im_data = Variable(img.cuda())


        depth_map, depth_scale = self._image_resize(depth_map, target_scale, cfg.TRAIN.MAX_SIZE)
        depth_map = Image.fromarray(depth_map, 'L')
        if self.transform is not None:
            depth_map = self.transform(depth_map)
        depth_map = depth_map.unsqueeze(0)
        #im_data = Variable(img.cuda(), volatile=True)  ## 추가 !!
        depth_map = Variable(depth_map.cuda())
        '''
        # 2D bounding box (TransferNet3 이전 버전)
        input_boxes = np.zeros((len(boxes), 6))
        input_boxes[:, :4] = np.array(boxes)/600.
        input_boxes[:, 4] = np.array(labels)  # background가 0임
        '''

        # 2D bounding box (TransferNet3 버전)
        input_boxes = np.zeros((len(boxes), 8))
        input_boxes[:, :4] = np.array(boxes)  # TransferNet3는 foward에서 600으로 나눔
        input_boxes[:, :4] *= im_scale  # bbox 크기 조정 (resize된 이미지에 따라.)
        input_boxes[:, 4] = np.array(labels)  # background가 0임
        box_sizes = np.array([self.prior_knowledge[au.obj_n2c[label]]['size_3d'] for label in labels])
        input_boxes[:, 5:] = box_sizes
        input_boxes = network.np_to_variable(np.array(input_boxes), is_cuda=True)

        # agent position, orientation
        if agent['rotation']['y'] == 0:
            agent_rotation = 0  # int 형태
        elif agent['rotation']['y'] == 90 or agent['rotation']['y'] == -270:
            agent_rotation = 1
        elif agent['rotation']['y'] == 180 or agent['rotation']['y'] == -180:
            agent_rotation = 2
        elif agent['rotation']['y'] == 270 or agent['rotation']['y'] == -90:
            agent_rotation = 3
        input_agent = [agent['position']['x'],  agent['position']['y'],
                       agent['position']['z'], agent_rotation]
        input_agent = network.np_to_variable(np.array(input_agent), is_cuda=True)
        input_agent = input_agent[np.newaxis, :]

        if CHECK_TIME:
            sub_st = time.time()
        # input_boxes : numpy, input_agent : Tensor
        #global_coor, _, _ = self.trans_net(input_boxes, input_agent)
        # im_data: Tensor, input_boxes: Tensor, input_agent: Tensor
        global_coor, _, _ = self.trans_net(im_data, depth_map, input_boxes, input_agent) # TransferNet3 버전용
        if CHECK_TIME:
            sub_time = time.time()-sub_st
            print('[TIME] 3. run trans_net() : {}s'.format(str(sub_time)[:7]))
        global_coor = global_coor.data.cpu().numpy()
        #global_coor = np.concatenate((global_coor, box_sizes), 1) # TransferNet3 버전용

        for i in range(len(global_coor)):  # prior 크기로 대체
            predict_size = global_coor[i][3:]
            prior_size_1 = box_sizes[i]
            prior_size_2 = [prior_size_1[2], prior_size_1[1], prior_size_1[0]]
            if self._get_l2(predict_size, prior_size_1) < self._get_l2(predict_size, prior_size_2):
                global_coor[i][3:] = np.array(prior_size_1)
            else:
                global_coor[i][3:] = np.array(prior_size_2)

        if CHECK_TIME:
            total_time = time.time() - st
            print('[TIME] 3. run transfer_to_global_map() : {}s'.format(str(total_time)[:7]))
            print('[TIME] \tpre/pos processing time : {}s'.format(str(total_time - sub_time)[:7]))
        return global_coor

    def _get_l2(self, a, b):
        distance = 0.
        for i in range(len(a)):
            distance += np.power(a[i] - b[i], 2)
        return distance
    # 검토중
    def transfer_to_global_map_back(self, boxes, labels, agent):
        if CHECK_TIME:
            st = time.time()
        ## model input
        # forward(self, objects, agent, gt_boxes_3d=None):
        # 1. bbox (x, y, x2, y2, distance, label) # OD 결과값 (학습 시에는 OD의 정답 bbox가 들어감)
        # 2. agent (x, y, z, rotation) #rotation은 0~3 값

        ## model output
        # return fc3_feature, total_loss, losses

        input_boxes = np.zeros((len(boxes), 6))
        input_boxes[:, :4] = np.array(boxes)/self.resolution
        input_boxes[:, 4] = np.array(labels) + 1 # +1은 background를 위해

        if agent['rotation']['y'] == 0:
            agent_rotation = 0  # int 형태
        elif agent['rotation']['y'] == 90 or agent['rotation']['y'] == -270:
            agent_rotation = 1
        elif agent['rotation']['y'] == 180 or agent['rotation']['y'] == -180:
            agent_rotation = 2
        elif agent['rotation']['y'] == 270 or agent['rotation']['y'] == -90:
            agent_rotation = 3
        input_agent = [agent['position']['x'],  agent['position']['y'],
                       agent['position']['z'], agent_rotation]

        input_agent = network.np_to_variable(np.array(input_agent), is_cuda=True)
        input_agent = input_agent[np.newaxis, :]

        if CHECK_TIME:
            sub_st = time.time()
        # input_boxes : numpy, input_agent : Tensor
        global_coor, _, _ = self.trans_net(input_boxes, input_agent)
        if CHECK_TIME:
            sub_time = time.time()-sub_st
            print('[TIME] 3. run trans_net() : {}s'.format(str(sub_time)[:7]))
        global_coor = global_coor.data.cpu().numpy()

        if CHECK_TIME:
            total_time = time.time() - st
            print('[TIME] 3. run transfer_to_global_map() : {}s'.format(str(total_time)[:7]))
            print('[TIME] \tpre/pos processing time : {}s'.format(str(total_time - sub_time)[:7]))
        return global_coor


    def predict_relationships(self, global_coors, labels, exist_box_index=[]):
        # exist_box_index에 해당하는 box(in global_coors/labels)끼리의 relationship은 제거함
        # return (새 박스 간의 관계 + 새 박스와 기존 박스간의 관계) 나머진 background
        # 동적 환경에서 새로운 관계만을 추가하기위해 구현. (기존 정적 환경에선 그냥 안쓰면됨)
        # RelNet3에서는 label 안씀

        if CHECK_TIME:
            st = time.time()
        ## model input
        # forward(self, boxes_3d, gt_relationships=None):
        # 1. 3D bbox (x, y, z, w, h, d, label)

        ## model output
        # return boxes_3d[:, :6], rel_prob, target_relationship, mat_phrase

        #input_data = global_coors + [labels]  # list 일 때
        input_data = np.concatenate((global_coors, np.expand_dims(labels, 1)), -1)  # numpy 일 때
        input_data = input_data.astype(float)
        input_data = torch.FloatTensor(input_data)

        if CHECK_TIME:
            sub_st = time.time()
        obj, rel_prob, _, _ = self.rel_net(input_data)
        if CHECK_TIME:
            sub_time = time.time()-sub_st
            print('[TIME] 4. run rel_net() : {}s'.format(str(sub_time)[:7]))

        output_rels = rel_prob.data.cpu().numpy().argmax(1)

        n_obj = len(labels)
        relationships = [] # [{sub_id, obj_id, relation}, ...] # 유효한 relationship만 추출

        for idx, rel in enumerate(output_rels):
            if rel == 0: # background
                continue
            if idx // n_obj == idx % n_obj: # sub와 obj가 같음
                continue
            if idx // n_obj in exist_box_index and idx % n_obj in exist_box_index: # 기존 물체간의 관계
                continue
            triple = {}
            triple['subject_id'] = idx // n_obj
            triple['object_id'] = idx % n_obj
            triple['rel_class'] = rel
            relationships.append(triple)
        if CHECK_TIME:
            total_time = time.time() - st
            print('[TIME] 4. run predict_relationships() : {}s'.format(str(total_time)[:7]))
            print('[TIME] \tpre/pos processing time : {}s'.format(str(total_time - sub_time)[:7]))
        #print('r', relationships)
        return np.array(relationships)

    def _image_resize(self, im, target_size, max_size):
        """Builds an input blob from the images in the roidb at the specified
        scales.
        """
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)

        return im, im_scale


def get_center_depth(depth, coordinates, range=0.6, show_plot=False, plot_title=""):
    # TransferNet6에서 사용. 각 box의 센터(range)에 해당하는 부분의 평균 depth값을 추출
    depth_values = np.zeros((len(coordinates)))
    for idx, (x1, y1, x2, y2) in enumerate(coordinates):
        w_alpha = ((x1 + x2) / 2 - x1) * (1 - range)
        h_alpha = ((y1 + y2) / 2 - y1) * (1 - range)
        #print('w_alpha, h_alpha:', w_alpha, h_alpha)
        if int(x2 - w_alpha) - int(x1 + w_alpha) == 0 or int(y2 - h_alpha) - int(y1 + h_alpha) == 0:
            depth_values[idx] = depth[int(x1):int(x2), int(y1):int(y2)].mean() / 50
            #print('mean1:', depth[int(x1):int(x2), int(y1):int(y2)].mean())
        else:
            depth_values[idx] = depth[int(x1 + w_alpha):int(x2 - w_alpha), int(y1 + h_alpha):int(y2 - h_alpha)].mean() / 50
            #print('mean2:los:', depth[int(x1 + w_alpha):int(x2 - w_alpha), int(y1 + h_alpha):int(y2 - h_alpha)])
            #print('mean2:', depth[int(x1 + w_alpha):int(x2 - w_alpha), int(y1 + h_alpha):int(y2 - h_alpha)].mean())

    return depth_values
