# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'faster_rcnn/fast_rcnn'))

import config
import nms_wrapper
# from nms_wrapper import nms