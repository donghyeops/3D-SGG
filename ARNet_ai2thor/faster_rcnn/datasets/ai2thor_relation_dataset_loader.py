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

import pdb

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob


class ai2thor_relation_dataset(data.Dataset):
    def __init__(self, set_option, image_set, db_version=2):
        self.db_version = db_version  #1.0은 오리지널, 2.0은 thorDBv2

        self._name = 'ai2thor_' + set_option + '_' + image_set
        if db_version == 1:
            DATA_DIR = './data'
        elif db_version == 2:
            DATA_DIR = './data/thorDBv2'
        print('DATA_DIR:', DATA_DIR)

        self._set_option = set_option
        # load category names and annotations
        annotation_dir = DATA_DIR
        cats = json.load(open(osp.join(annotation_dir, 'categories.json')))
        class_weight = json.load(open(osp.join(annotation_dir, 'class_weights.json')))

        self._object_classes = tuple(['__background__'] + cats['object'])
        self._predicate_classes = tuple(['background'] + cats['predicate'])

        self._object_class_to_ind = dict(list(zip(self.object_classes, list(range(self.num_object_classes)))))
        self._predicate_class_to_ind = dict(list(zip(self.predicate_classes, list(range(self.num_predicate_classes)))))


        ann_file_name = {'ai2thor_small_train': 'small_train.json',
                         'ai2thor_small_test': 'small_test.json',
                         'ai2thor_normal_train': 'train.json',
                         'ai2thor_normal_test': 'test.json'
                         }

        self.class_weight_relationship = torch.ones(self.num_predicate_classes)
        for idx in range(1, self.num_predicate_classes):
            if self._predicate_classes[idx] in class_weight['relationship']:  # 만약에 해당 클래스의 weight가 없으면 1로 만듦
                self.class_weight_relationship[idx] = class_weight['relationship'][self._predicate_classes[idx]]  # 이건 train.json에 정의해둔 class가 없어서 그럼
                print(self._predicate_classes[idx])
                continue
            else:  #
                self.class_weight_relationship[idx] = 0.
        print('class_weight_relationship:', self.class_weight_relationship) ##

        ann_file_path = osp.join(annotation_dir, ann_file_name[self.name])
        self.annotations = json.load(open(ann_file_path))
        self.tokenize_annotations()

    def __getitem__(self, index):
        # input data (global coordinate of objects)
        # 1. 3D bbox (x, y, z, w, h, d, label)

        # output data
        # 1. relations (box#, box#)

        _annotation = self.annotations[index]


        boxes_3d = torch.zeros((len(_annotation['objects']), 7))
        try:
            boxes_3d[:, :3] = torch.FloatTensor([obj['global_position'] for obj in _annotation['objects']])
        except:
            print(len(_annotation['objects']), _annotation['id'])
        boxes_3d[:, 3:6] = torch.FloatTensor([obj['size_3d'] for obj in _annotation['objects']])
        boxes_3d[:, 6] = torch.FloatTensor([obj['class'] for obj in _annotation['objects']])

        gt_relationships = torch.zeros(len(_annotation['objects']), (len(_annotation['objects']))).type(
            torch.LongTensor)
        try:
            for rel in _annotation['global_relationships']:
                gt_relationships[rel['sub_id'], rel['obj_id']] = rel['predicate']
        except:
            print(len(_annotation['objects']), _annotation['id'])
            print(rel['sub_id'], rel['obj_id'])

        return boxes_3d, gt_relationships

    def __len__(self):
        return len(self.annotations)

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
        del_list = []
        for im in self.annotations:
            if len(im['objects']) < 2:
                del_list.append(im)
                continue
            for obj in im['objects']:
                obj['class'] = self._object_class_to_ind[obj['class']]
            for rel in im['global_relationships']:
                rel['predicate'] = self._predicate_class_to_ind[rel['predicate']]
        for d in del_list:
            self.annotations.remove(d)

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
    def num_predicate_classes(self):
        return len(self._predicate_classes)

    @property
    def object_classes(self):
        return self._object_classes

    @property
    def predicate_classes(self):
        return self._predicate_classes
#    @property
#    def off_state_classes(self): #
#        return self._off_state_classes