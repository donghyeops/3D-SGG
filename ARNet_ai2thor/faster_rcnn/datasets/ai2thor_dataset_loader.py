#-*- coding: utf-8 -*-

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

class ai2thor_dataset(data.Dataset):
    def __init__(self, set_option, image_set, annotation_folder='annotation'):
        self._name = 'ai2thor_' + set_option + '_' + image_set
        IMAGE_DATA_DIR = '/media/ailab/D/ai2thor'
        DATA_DIR = './data'
        
        self._set_option = set_option
        self._image_set = image_set
        self._data_path = osp.join(IMAGE_DATA_DIR, 'images')
        # load category names and annotations
        annotation_dir = DATA_DIR
        cats = json.load(open(osp.join(annotation_dir, 'categories.json')))
        #inverse_weight = json.load(open(osp.join(annotation_dir, 'inverse_weight.json')))

        self._object_classes = tuple(['__background__'] + cats['object'])
        self._predicate_classes = tuple(['__background__'] + cats['predicate'])
        self._color_classes = tuple(cats['color']) # background 따로 없음
        self._open_state_classes = tuple(cats['open_state']) # background 따로 없음
        #self._off_state_classes = tuple(cats['off_state']) # background 따로 없음
        
        self._object_class_to_ind = dict(list(zip(self.object_classes, list(range(self.num_object_classes)))))
        self._color_class_to_ind = dict(list(zip(self.color_classes, list(range(self.num_color_classes))))) #
        self._open_state_class_to_ind = dict(list(zip(self.open_state_classes, list(range(self.num_open_state_classes))))) #
        #self._off_state_class_to_ind = dict(zip(self.off_state_classes, xrange(self.num_off_state_classes))) #
        self._predicate_class_to_ind = dict(list(zip(self.predicate_classes, list(range(self.num_predicate_classes)))))
        #self.inverse_weight_object = torch.ones(self.num_object_classes)
        #for idx in xrange(1, self.num_object_classes):
        #    self.inverse_weight_object[idx] = inverse_weight['object'][self._object_classes[idx]]
        #self.inverse_weight_object = self.inverse_weight_object / self.inverse_weight_object.min()
        # print self.inverse_weight_object
        #self.inverse_weight_predicate = torch.ones(self.num_predicate_classes)
        #for idx in xrange(1, self.num_predicate_classes):
        #    self.inverse_weight_predicate[idx] = inverse_weight['predicate'][self._predicate_classes[idx]]
        #self.inverse_weight_predicate = self.inverse_weight_predicate / self.inverse_weight_predicate.min()
        # print self.inverse_weight_predicate
        ann_file_name = {'ai2thor_normal_train': 'train.json', 
                           'ai2thor_normal_test': 'test.json'}

        ann_file_path = osp.join(annotation_dir, ann_file_name[self.name])
        self.annotations = json.load(open(ann_file_path))
        self.tokenize_annotations()
        
        # image transformation
        normalize = transforms.Normalize(mean=[0.352, 0.418, 0.455], std=[0.155, 0.16, 0.162])
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

    def __getitem__(self, index):
        # Sample random scales to use for each image in this batch
        target_scale = cfg.TRAIN.SCALES[npr.randint(0, high=len(cfg.TRAIN.SCALES))]
        #img = cv2.imread(osp.join('/media/ailab/D/ai2thor/images', self.annotations[index]['path']))
        img = cv2.imread(osp.join(self._data_path, self.annotations[index]['path']))

        img, im_scale = self._image_resize(img, target_scale, cfg.TRAIN.MAX_SIZE)
        im_info = np.array([img.shape[0], img.shape[1], im_scale], dtype=np.float32)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
            
        _annotation = self.annotations[index]
        gt_boxes_object = torch.zeros((len(_annotation['objects']), 5))
        gt_boxes_object[:, 0:4] = torch.FloatTensor([obj['box'] for obj in _annotation['objects']]) * im_scale
        gt_boxes_object[:, 4]   = torch.FloatTensor([obj['class'] for obj in _annotation['objects']])

        gt_colors = torch.LongTensor([obj['color'] for obj in _annotation['objects']])
        #print(gt_colors.size()) ##
        gt_open_states = torch.LongTensor([obj['open_state'] for obj in _annotation['objects']])
        #print(gt_open_states.size()) ##

        gt_relationships = torch.zeros(len(_annotation['objects']), (len(_annotation['objects']))).type(torch.LongTensor)
        for rel in _annotation['relationships']:
            gt_relationships[rel['sub_id'], rel['obj_id']] = rel['predicate']

        return img, im_info, gt_boxes_object, gt_colors, gt_open_states, gt_relationships

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
        for im in self.annotations:
            for obj in im['objects']:
                obj['class'] = self._object_class_to_ind[obj['class']]
                obj['color'] = self._color_class_to_ind[obj['color']]
                obj['open_state'] = self._open_state_class_to_ind[obj['open_state']]
                #obj['off_state'] = self._off_state_class_to_ind[obj['off_state']]
            for rel in im['relationships']:
                rel['predicate'] = self._predicate_class_to_ind[rel['predicate']]


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
    def num_color_classes(self): #
        return len(self._color_classes)
    @property
    def num_open_state_classes(self): #
        return len(self._open_state_classes)
#    @property
#    def num_off_state_classes(self): #
#        return len(self._off_state_classes)
        
    @property
    def object_classes(self):
        return self._object_classes

    @property
    def predicate_classes(self):
        return self._predicate_classes
        
    @property
    def color_classes(self): #
        return self._color_classes
    @property
    def open_state_classes(self): #
        return self._open_state_classes
#    @property
#    def off_state_classes(self): #
#        return self._off_state_classes
