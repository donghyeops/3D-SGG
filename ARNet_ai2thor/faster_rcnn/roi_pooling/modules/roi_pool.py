#-*- coding:utf-8 -*-
from torch.nn.modules.module import Module
from functions.roi_pool import RoIPoolFunction


class RoIPool(Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(RoIPool, self).__init__()

        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.module = RoIPoolFunction(self.pooled_height, self.pooled_width, self.spatial_scale) ##

    def forward(self, features, rois):
        return self.module(features, rois)
        #return RoIPoolFunction(self.pooled_height, self.pooled_width, self.spatial_scale)(features, rois)
