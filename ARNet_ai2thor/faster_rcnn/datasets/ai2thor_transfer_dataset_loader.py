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

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


class ai2thor_transfer_dataset(data.Dataset):
    def __init__(self, set_option, image_set):
        self._name = 'ai2thor_' + set_option + '_' + image_set
        IMAGE_DATA_DIR = '/media/ailab/D/ai2thor'
        DATA_DIR = './data'

        self._image_set = image_set
        self._data_path = osp.join(IMAGE_DATA_DIR, 'images')
        self._depth_path = osp.join(IMAGE_DATA_DIR, 'depth_images')

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
            'ai2thor_normal_train': 'train.json',
            'ai2thor_normal_test': 'test.json'
        }

        ann_file_path = osp.join(annotation_dir, ann_file_name[self.name])
        self.annotations = json.load(open(ann_file_path))

        ##############################
        target_room = None#'FloorPlan29'
        print('tartget room: {}'.format(target_room))
        if target_room is not None:
            new_ann = []
            for ann in self.annotations:
                if ann['scene_name'] == target_room:
                    new_ann.append(ann)
            self.annotations = new_ann
        ##############################

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
        self.use_depth_value = True  # TransferNet6에서만 사용
        self.use_add_noise = True  # box 좌표에 노이즈 추가 [버전에 무관하게 사용 가능]
        self.change_agent_rotation_scale = {0: 0., 1: 0.33, 2: 0.66, 3: 0.99}  # 0~3을 0~1로 변환
        self.use_relative_margin_for_answer = True  # 절대값이 아닌 차이값을 예측 (TransferNet6)

    def __getitem__(self, index):
        # Sample random scales to use for each image in this batch
        target_scale = cfg.TRAIN.SCALES[npr.randint(0, high=len(cfg.TRAIN.SCALES))]
        img = cv2.imread(osp.join(self._data_path, self.annotations[index]['path']))  # 0~255

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img, im_scale = self._image_resize(img, target_scale, cfg.TRAIN.MAX_SIZE)  # 0~255
        im_info = np.array([img.shape[0], img.shape[1], im_scale], dtype=np.float32)
        img = Image.fromarray(img)  # 0~1

        if self.transform is not None:
            img = self.transform(img)  # convert to Tensor

        if self.use_depth_image:
            depth = cv2.imread(osp.join(self._depth_path, self.annotations[index]['depth_path']),
                               cv2.IMREAD_GRAYSCALE)  # 0~255

            depth, depth_scale = self._image_resize(depth, target_scale, cfg.TRAIN.MAX_SIZE)  # 0~255
            depth_info = np.array([depth.shape[0], depth.shape[1], depth_scale], dtype=np.float32)

            if not self.use_depth_value:
                depth = Image.fromarray(depth)  # 0~1
                depth = self.transform(depth)  # convert to Tensor

        _annotation = self.annotations[index]

        # input data
        # 1. bbox (x, y, x2, y2, distance, label) # OD 결과값 (학습 시에는 OD의 정답 bbox가 들어감)
        # 2. agent (x, y, z, rotation) #rotation은 0~3 값

        if self.use_depth_value:
            # boxes : (object#, 8[x, y, x2, y2, label, depth, size_w[PK], size_h[PK], size_z[PK]]) PK는 사전지식
            # depth value 추가된 버전 (TransferNet6)
            boxes = torch.zeros((len(_annotation['objects']), 9))
        else:
            # boxes : (object#, 8[x, y, x2, y2, label, size_w[PK], size_h[PK], size_z[PK]]) PK는 사전지식
            boxes = torch.zeros((len(_annotation['objects']), 8))

        # boxes[:, :4] = torch.FloatTensor([obj['box'] for obj in _annotation['objects']])
        # boxes[:, :4] = torch.FloatTensor([[v/600. for v in obj['box']] for obj in _annotation['objects']])
        coordinates = [obj['box'] for obj in _annotation['objects']]
        # coordinates = self.add_noise(coordinates, noise=10)
        boxes[:, 0:4] = torch.FloatTensor(coordinates) * im_scale
        # print(boxes[0])
        boxes[:, 4] = torch.FloatTensor([obj['class'] for obj in _annotation['objects']])

        if self.use_depth_value:
            boxes[:, 5] = torch.FloatTensor(self.get_center_depth(depth, coordinates, range=0.6))
            boxes[:, 6:] = torch.FloatTensor([self.prior_knowledge[self._object_ind_to_class[obj['class']]]['size_3d']
                                              for obj in _annotation['objects']])
        else:
            boxes[:, 5:] = torch.FloatTensor([self.prior_knowledge[self._object_ind_to_class[obj['class']]]['size_3d']
                                              for obj in _annotation['objects']])

        agent = torch.zeros(4)  # (x, y, z, rotation)
        agent[:3] = torch.FloatTensor(_annotation['agent']['global_position'])
        agent[3] = torch.FloatTensor([self.change_agent_rotation_scale[_annotation['agent']['global_rotation']]])
        # agent[3] = torch.FloatTensor([_annotation['agent']['global_rotation']])  # 튜닝 미적용

        # output data (answer)
        # 1. 3D bbox (x, y, z, w, h, d)

        if self.use_relative_margin_for_answer:
            # xz, y 값을 현재 에이전트의 중심과 얼마나 차이나는 지를 답으로함 # wd는 그대로.
            # TransferNet6에서만 적용
            targets = torch.zeros((len(_annotation['objects']), 3))
            # 현재 에이전트 각도에 따라 다름
            if _annotation['agent']['global_rotation'] == 0:  # 앞을 볼 때, x축 사용, -|+
                targets[:, 0] = torch.FloatTensor(
                    [+ obj['global_position'][0] - _annotation['agent']['global_position'][0] for obj in
                     _annotation['objects']])
                targets[:, 2] = torch.FloatTensor([obj['size_3d'][0] for obj in _annotation['objects']])
            elif _annotation['agent']['global_rotation'] == 1:  # 오른쪽 볼 때, z축 사용, +|-
                targets[:, 0] = torch.FloatTensor(
                    [- obj['global_position'][2] + _annotation['agent']['global_position'][2] for obj in
                     _annotation['objects']])
                targets[:, 2] = torch.FloatTensor([obj['size_3d'][2] for obj in _annotation['objects']])
            elif _annotation['agent']['global_rotation'] == 2:  # 뒤를 볼 때, x축 사용, -|+
                targets[:, 0] = torch.FloatTensor(
                    [- obj['global_position'][0] + _annotation['agent']['global_position'][0] for obj in
                     _annotation['objects']])
                targets[:, 2] = torch.FloatTensor([obj['size_3d'][0] for obj in _annotation['objects']])
            else:  # 왼쪽 볼 때, z축 사용, -|+
                targets[:, 0] = torch.FloatTensor(
                    [+ obj['global_position'][2] - _annotation['agent']['global_position'][2] for obj in
                     _annotation['objects']])
                targets[:, 2] = torch.FloatTensor([obj['size_3d'][2] for obj in _annotation['objects']])
            # y축 값은 공통
            targets[:, 1] = torch.FloatTensor(
                [+ obj['global_position'][1] - _annotation['agent']['global_position'][1] for obj in
                 _annotation['objects']])

            # 잘 되는지 출력 값 확인
            # print(_annotation['path'])
            # for i in range(len(targets)):
            #    print('\t', self._object_ind_to_class[_annotation['objects'][i]['class']], targets[i])
            # self.get_center_depth(depth, coordinates, range=0.6, show_plot=True, plot_title=_annotation['path'])
            # assert 1==2
            return img, depth, boxes, agent, targets

        # gt_boxes_3d_object : (object#, 6[x, y, z, w, h, d])
        if self.only_position:
            gt_boxes_3d = torch.zeros((len(_annotation['objects']), 3))
            gt_boxes_3d[:] = torch.FloatTensor([obj['global_position'] for obj in _annotation['objects']])
        else:
            gt_boxes_3d = torch.zeros((len(_annotation['objects']), 6))
            gt_boxes_3d[:, :3] = torch.FloatTensor([obj['global_position'] for obj in _annotation['objects']])
            gt_boxes_3d[:, 3:] = torch.FloatTensor([obj['size_3d'] for obj in _annotation['objects']])

        if self.use_depth_image or self.use_depth_value:
            return img, depth, boxes, agent, gt_boxes_3d
        return img, im_info, boxes, agent, gt_boxes_3d

    def __len__(self):
        return len(self.annotations)

    def add_noise(self, coordinates, noise=40):
        for i, coor in enumerate(coordinates):
            for j, v in enumerate(coor):
                coordinates[i][j] += rd.randint(max(-noise + v, 1), min(noise + v, 599))
        return coordinates

    def get_center_depth(self, depth, coordinates, range=0.6, show_plot=False, plot_title=""):
        # TransferNet6에서 사용. 각 box의 센터(range)에 해당하는 부분의 평균 depth값을 추출
        depth_values = np.zeros((len(coordinates)))
        for idx, (x1, y1, x2, y2) in enumerate(coordinates):
            w_alpha = ((x1 + x2) / 2 - x1) * (1 - range)
            h_alpha = ((y1 + y2) / 2 - y1) * (1 - range)
            if int(x2 - w_alpha) - int(x1 + w_alpha) == 0 or int(y2 - h_alpha) - int(y1 + h_alpha) == 0:
                depth_values[idx] = depth[int(x1):int(x2), int(y1):int(y2)].mean() / 50
            else:
                depth_values[idx] = depth[int(x1 + w_alpha):int(x2 - w_alpha),
                                    int(y1 + h_alpha):int(y2 - h_alpha)].mean() / 50

            if show_plot:
                plt.suptitle(str(plot_title))
                a = plt.subplot(221)
                b = plt.subplot(222)
                c = plt.subplot(223)
                d = plt.subplot(224)
                a.imshow(depth, cmap='gray')
                a.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=None, alpha=1))
                b.imshow(depth, cmap='gray')
                b.add_patch(Rectangle((x1 + w_alpha, y1 + h_alpha), (x2 - w_alpha) - (x1 + w_alpha),
                                      (y2 - h_alpha) - (y1 + h_alpha), fill=None, alpha=1))
                c.imshow((depth[int(y1):int(y2), int(x1):int(x2)] / 20),
                         cmap='gray')
                d.imshow((depth[int(y1 + h_alpha):int(y2 - h_alpha), int(x1 + w_alpha):int(x2 - w_alpha)] / 20),
                         cmap='gray')
                plt.show()

        return depth_values

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
            for obj in im['objects']:
                obj['class'] = self._object_class_to_ind[obj['class']]

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
