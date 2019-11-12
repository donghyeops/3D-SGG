# -*- coding: utf-8 -*-

from PIL import Image
import os
import os.path as osp
import errno
import numpy as np
import numpy.random as npr
import sys
import json
import cv2
import random as rd
import pdb
import math

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

# 본 클래스는 transferNet7부터 사용함.
class ai2thor_transfer_dataset(data.Dataset):
    def __init__(self, set_option, image_set, use_default_box):
        self.use_default_box = use_default_box
        self._name = 'ai2thor_' + set_option + '_' + image_set
        IMAGE_DATA_DIR = '/media/ailab/D/ai2thor/thorDBv2'#'/media/ailab/D/ai2thor'
        DATA_DIR = './data/thorDBv2/roi_db'

        self._image_set = image_set
        self._image_path = IMAGE_DATA_DIR
        self._data_path = osp.join(IMAGE_DATA_DIR, 'roi_images')
        self._depth_path = osp.join(IMAGE_DATA_DIR, 'roi_depth_images')

        self._set_option = set_option
        # load category names and annotations
        annotation_dir = DATA_DIR
        cats = json.load(open(osp.join(annotation_dir, 'categories.json')))
        prior_knowledge = json.load(open(osp.join(annotation_dir, 'prior_knowledge.json')))
        self.prior_knowledge = prior_knowledge['objects']

        self._object_classes = tuple(['__background__'] + cats['object'])

        self._object_class_to_ind = dict(list(zip(self.object_classes, list(range(self.num_object_classes)))))
        self._object_ind_to_class = dict(list(zip(list(range(self.num_object_classes)), self.object_classes)))

        ann_file_name = {
            'ai2thor_normal_train': 'merge_roi_train.json',
            'ai2thor_normal_test': 'merge_roi_test.json'
        }

        ann_file_path = osp.join(annotation_dir, ann_file_name[self.name])
        self.annotations = json.load(open(ann_file_path))

        ## 데이터 필터링
        #if image_set == 'train':
        #    self.remove_small_objects(area_limit=300) # box 너비가 400 미만이면 학습 안함


        print(f'{image_set} data#: {len(self.annotations)}')
        ##
        self.tokenize_annotations()

        # image transformation
        # normalize = transforms.Normalize(mean=[0.352, 0.418, 0.455], std=[0.155, 0.16, 0.162]) # 기존 DB
        normalize = transforms.Normalize(mean=[0.319, 0.396, 0.452],
                                         std=[0.149, 0.155, 0.155])  # 새 DB 3507개 (181106)에 바꿈
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # self.transform = transforms.Compose([
        #    transforms.ToTensor(),
        #    normalize,
        # ])

        self.use_distance = False  # distance 데이터 사용
        self.only_position = False  # position만 예측 => ann 데이터는 pos만 받음, 아니면 size까지 받음
        self.use_depth_image = True
        self.use_add_noise = True  # box 좌표에 노이즈 추가 [버전에 무관하게 사용 가능]
        self.change_agent_rotation_scale = {0: 0., 1: 0.33, 2: 0.66, 3: 0.99}  # 0~3을 0~1로 변환
        self.use_relative_margin_for_answer = True  # 절대값이 아닌 차이값을 예측 (TransferNet6)
        #self.log = open('/home/ailab/DH/ai2thor/ARNet_ai2thor/data/log.txt', 'wt')

    def remove_small_objects(self, area_limit):
        remove_target_idx = []
        for i, ann in enumerate(self.annotations):
            box = ann['object_info']['box']
            if (box[3]-box[1]) * (box[2]-box[0]) < area_limit:
                remove_target_idx.append(i)
        remove_target_idx = sorted(remove_target_idx)[::-1]
        for i in remove_target_idx:
            del self.annotations[i]
        print(f'\nremove small objects [#:{len(remove_target_idx)}]')
                

    def __getitem__(self, index):
        img = cv2.imread(osp.join(self._image_path, f'roi_images_{self.annotations[index]["source"]}', self.annotations[index]['roi_image_path']))  # 0~255
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            print(osp.join(self._image_path, self.annotations[index]['roi_image_path']))
            assert False, 'there is no file'

        img = cv2.resize(img, (64,64))
        img = Image.fromarray(img)  # 0~1

        if self.transform is not None:
            img = self.transform(img)  # convert to Tensor

        depth = cv2.imread(osp.join(self._image_path, f'roi_depth_images_{self.annotations[index]["source"]}', self.annotations[index]['roi_depth_path']),
                           cv2.IMREAD_GRAYSCALE)  # 0~255
        #print(osp.join(self._image_path, f'roi_depth_images_{self.annotations[index]["source"]}', self.annotations[index]['roi_depth_path']))
        depth = cv2.resize(depth, (64, 64))
        #print(depth.shape,type(depth), depth[0,:5])
        depth = Image.fromarray(depth)  # 0~1
        depth_temp = np.array(depth)
        #print(depth.size, type(depth))
        depth = self.transform(depth)  # convert to Tensor

        _annotation = self.annotations[index]

        # input data
        # 1. bbox (x, y, x2, y2, distance, label) # OD 결과값 (학습 시에는 OD의 정답 bbox가 들어감)
        # 2. agent (x, y, z, rotation) #rotation은 0~3 값

        # boxes : (object#, 5[x, y, x2, y2, label)
        boxes = torch.zeros(5)

        # boxes[:, :4] = torch.FloatTensor([obj['box'] for obj in _annotation['objects']])
        # boxes[:, :4] = torch.FloatTensor([[v/600. for v in obj['box']] for obj in _annotation['objects']])
        obj = _annotation['object_info']
        coordinates = obj['box']

        # coordinates = self.add_noise(coordinates, noise=10)
        boxes[0:4] = torch.FloatTensor(coordinates)
        # print(boxes[0])
        boxes[4] = torch.FloatTensor([obj['class']])

        agent = torch.zeros(5)  # (x, y, z, rotation)
        agent[:3] = torch.FloatTensor(_annotation['agent']['global_position'])
        agent[3] = torch.FloatTensor([self.change_agent_rotation_scale[_annotation['agent']['global_rotation']]])
        agent[4] = torch.FloatTensor([float(_annotation['agent']['cameraHorizon'])])
        # agent[3] = torch.FloatTensor([_annotation['agent']['global_rotation']])  # 튜닝 미적용

        # output data (answer)
        # 1. 3D bbox (x, y, z, wd, h)
        target, default_box = self.get_target_v2(_annotation, depth_temp)
        if not self.use_default_box:
            target = target + default_box
        # print('img:', _annotation['path'])
        # print('roi:', _annotation['roi_image_path'])
        # print('cls:', _annotation['object_info']['class'])
        # print('') #
        # print('') #
        
        # 잘 되는지 출력 값 확인
        # print(_annotation['path'])
        # for i in range(len(targets)):
        #    print('\t', self._object_ind_to_class[_annotation['objects'][i]['class']], targets[i])
        # self.get_center_depth(depth, coordinates, range=0.6, show_plot=True, plot_title=_annotation['path'])
        # assert 1==2
        gt_boxes_3d = torch.zeros((6))
        gt_boxes_3d[:3] = torch.FloatTensor(obj['global_position'])
        gt_boxes_3d[3:] = torch.FloatTensor(obj['size_3d'])

        # for factor in [img, depth, boxes, agent, target, default_box, gt_boxes_3d]:
        #     print(factor.shape)

        return img, depth, boxes, agent, target, default_box, gt_boxes_3d

    def __len__(self):
        return len(self.annotations)

    def get_target_v2(self, _annotation, depth):
        # x, y, z 값을 현재 에이전트의 중심과 얼마나 차이나는 지를 답으로함 # wd, h는 그대로.
        
        obj = _annotation['object_info']

        relative_box = torch.zeros(6) # 현재 에이전트 관점에서 (x, y, z, w, h, d)
        # 현재 에이전트 각도에 따라 다름
        if _annotation['agent']['global_rotation'] == 0:  # 앞을 볼 때, x축 사용, -|+
            relative_box[0] = torch.FloatTensor( \
                [+ obj['global_position'][0] - _annotation['agent']['global_position'][0]])
            relative_box[2] = torch.FloatTensor( \
                [+ obj['global_position'][2] - _annotation['agent']['global_position'][2]])
        elif _annotation['agent']['global_rotation'] == 1:  # 오른쪽 볼 때, z축 사용, +|-
            relative_box[0] = torch.FloatTensor( \
                [- obj['global_position'][2] + _annotation['agent']['global_position'][2]])
            relative_box[2] = torch.FloatTensor( \
                [+ obj['global_position'][0] - _annotation['agent']['global_position'][0]])
        elif _annotation['agent']['global_rotation'] == 2:  # 뒤를 볼 때, x축 사용, -|+
            relative_box[0] = torch.FloatTensor( \
                [- obj['global_position'][0] + _annotation['agent']['global_position'][0]])
            relative_box[2] = torch.FloatTensor( \
                [- obj['global_position'][2] + _annotation['agent']['global_position'][2]])
        else:  # 왼쪽 볼 때, z축 사용, -|+
            relative_box[0] = torch.FloatTensor( \
                [+ obj['global_position'][2] - _annotation['agent']['global_position'][2]])
            relative_box[2] = torch.FloatTensor( \
                [- obj['global_position'][0] + _annotation['agent']['global_position'][0]])
        # y축 값은 안뺌 (9에서만..) (default box로 차를 구하는데, y는 기준값이 바닥이기 때문에 안하는게 나음)
        # 2.는 높이 최대값.
        relative_box[1] = torch.FloatTensor([obj['global_position'][1]])/2. # /2하면 성능 조금 올라감 (값 범위를 0~1로 낮춤)
        # print(targets.size())
        # print(torch.FloatTensor([obj['size_3d'] for obj in _annotation['objects']]).size())
        relative_box[3:] = torch.FloatTensor(obj['size_3d'])
        
        default_box = torch.zeros(6)
        
        default_box[0] = (((obj['box'][2] + obj['box'][0]) / 2)-300) / 600 # x
        default_box[1] = (600-((obj['box'][3] + obj['box'][1]) / 2)) / 600# y # 300은 가운데인데, agent 위치가 아님, 그래서 바닥을 기준
        # 그리고 위가 0이고, 아래가 600이라서 반전.
        default_box[2] = self.get_center_depth(depth) # z: depth

        #default_box[:2] *= default_box[2]  # 거리에 따라 좁게 보이는 현상을 고려.

        default_box[3] = (obj['box'][2] - obj['box'][0]) / 600# * math.log2(default_box[2]+1) # w
        default_box[4] = (obj['box'][3] - obj['box'][1]) / 600# * math.log2(default_box[2]+1) # h
        default_box[5] = (obj['box'][2] - obj['box'][0]) / 600# * math.log2(default_box[2]+1) # d: w와 동등
        
        # target = relative_box - default_box
        # print('r:', relative_box)
        # print('d1:', default_box)
        # print('t1:', target)
        default_box[:2] *= default_box[2]  # 거리에 따라 좁게 보이는 현상을 고려.
        #default_box[1] *= math.log(default_box[2], 1.4)  # 거리에 따라 좁게 보이는 현상을 고려.
        target = relative_box - default_box
        #print('agent:', _annotation['agent']['global_position'])
        # print('d2:', default_box)
        # print('t2:', target)
        # print('')
        # print('')


        return target, default_box
    
    
    def get_target_v1(self, _annotation):
        # x, y, z 값을 현재 에이전트의 중심과 얼마나 차이나는 지를 답으로함 # wd, h는 그대로.
        obj = _annotation['object_info']

        targets = torch.zeros(6)
        # 현재 에이전트 각도에 따라 다름
        if _annotation['agent']['global_rotation'] == 0:  # 앞을 볼 때, x축 사용, -|+
            targets[0] = torch.FloatTensor( \
                [+ obj['global_position'][0] - _annotation['agent']['global_position'][0]])
            targets[2] = torch.FloatTensor( \
                [+ obj['global_position'][2] - _annotation['agent']['global_position'][2]])
        elif _annotation['agent']['global_rotation'] == 1:  # 오른쪽 볼 때, z축 사용, +|-
            targets[0] = torch.FloatTensor( \
                [- obj['global_position'][2] + _annotation['agent']['global_position'][2]])
            targets[2] = torch.FloatTensor( \
                [+ obj['global_position'][0] - _annotation['agent']['global_position'][0]])
        elif _annotation['agent']['global_rotation'] == 2:  # 뒤를 볼 때, x축 사용, -|+
            targets[0] = torch.FloatTensor( \
                [- obj['global_position'][0] + _annotation['agent']['global_position'][0]])
            targets[2] = torch.FloatTensor( \
                [- obj['global_position'][2] + _annotation['agent']['global_position'][2]])
        else:  # 왼쪽 볼 때, z축 사용, -|+
            targets[0] = torch.FloatTensor( \
                [+ obj['global_position'][2] - _annotation['agent']['global_position'][2]])
            targets[2] = torch.FloatTensor( \
                [- obj['global_position'][0] + _annotation['agent']['global_position'][0]])
        # y축 값은 공통
        targets[1] = torch.FloatTensor( \
            [+ obj['global_position'][1] - _annotation['agent']['global_position'][1]])
        # print(targets.size())
        # print(torch.FloatTensor([obj['size_3d'] for obj in _annotation['objects']]).size())
        targets[3:] = torch.FloatTensor(obj['size_3d'])
        
        return targets
        
        
    def add_noise(self, coordinates, noise=40):
        for i, coor in enumerate(coordinates):
            for j, v in enumerate(coor):
                coordinates[i][j] += rd.randint(max(-noise + v, 1), min(noise + v, 599))
        return coordinates

    def get_center_depth(self, depth, range=0.6):
        # TransferNet8에서 사용. 센터(range)에 해당하는 부분의 평균 depth값을 추출
        w = depth.shape[1]
        h = depth.shape[0]
        min = (1-range)/2
        max = (1-range)/2 + range
        depth_value = depth[int(w*min):int(w*max), int(h*min):int(h*max)]

        return depth_value.mean()/50

    @property
    def voc_size(self):
        return len(self.idx2word)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(i)

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # Example image path for index=119993:
        #   images/train2014/COCO_train2014_000000119993.jpg
        file_name = self.annotations[index]['path']
        image_path = osp.join(self._data_path, file_name)
        assert osp.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def tokenize_annotations(self):

        counter = 0
        # print 'Tokenizing annotations...'
        for im in self.annotations:
            im['object_info']['class'] = self._object_class_to_ind[im['object_info']['class']]

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

    @property
    def name(self):
        return self._name

    @property
    def num_object_classes(self):
        return len(self._object_classes)

    @property
    def object_classes(self):
        return self._object_classes
