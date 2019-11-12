# -*- coding: utf-8 -*-
# code for ARNet

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from fast_rcnn.nms_wrapper import nms
import network
from vgg16 import VGG16
import torchvision.models as models

DEBUG = False


def nms_detections(pred_boxes, scores, nms_thresh, inds=None):
    dets = np.hstack((pred_boxes,
                      scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, nms_thresh)
    if inds is None:
        return pred_boxes[keep], scores[keep]
    return pred_boxes[keep], scores[keep], inds[keep]


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        # self.features = VGG16(bn=False)
        self.features = models.vgg16(pretrained=True).features
        self.features.__delattr__('30')  # to delete the max pooling
        # by default, fix the first four layers
        network.set_trainable_param(list(self.features.parameters())[:8], requires_grad=False)

    def forward(self, im_data, im_info, gt_objects=None, dontcare_areas=None):
        im_data = Variable(im_data.cuda())

        features = self.features(im_data)
        # print 'features.std()', features.data.std()
        return features

    @staticmethod
    def reshape_layer(x, d):
        input_shape = x.size()
        # x = x.permute(0, 3, 1, 2)
        # b c w h
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        # x = x.permute(0, 2, 3, 1)
        return x

    def load_from_npz(self, params):
        # params = np.load(npz_file)
        self.features.load_from_npz(params)

        pairs = {'conv1.conv': 'rpn_conv/3x3', 'score_conv.conv': 'rpn_cls_score', 'bbox_conv.conv': 'rpn_bbox_pred'}
        own_dict = self.state_dict()
        for k, v in list(pairs.items()):
            key = '{}.weight'.format(k)
            param = torch.from_numpy(params['{}/weights:0'.format(v)]).permute(3, 2, 0, 1)
            own_dict[key].copy_(param)

            key = '{}.bias'.format(k)
            param = torch.from_numpy(params['{}/biases:0'.format(v)])
            own_dict[key].copy_(param)
