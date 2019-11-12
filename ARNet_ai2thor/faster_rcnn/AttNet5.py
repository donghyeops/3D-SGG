# -*- coding: utf-8 -*-
# code for AttNet


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import network
from fast_rcnn.nms_wrapper import nms
from network import Conv2d, FC, my_FC
from roi_pooling.modules.roi_pool import RoIPool
from matplotlib import pyplot as plt
import matplotlib.patches as patches

PRINT_MODEL_PROCESS = False
PRINT_FEATURE_VALUES = False

def nms_detections(pred_boxes, scores, nms_thresh, inds=None):
    dets = np.hstack((pred_boxes,
                      scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, nms_thresh)
    if inds is None:
        return pred_boxes[keep], scores[keep], keep
    return pred_boxes[keep], scores[keep], inds[keep], keep


class AttNet(nn.Module):
    def __init__(self,
                 nlabel=25, ncolor=10, nopen_state=3,
                 class_weight_color=None,
                 class_weight_os=None,
                 use_class=True,
                 use_vis=True):
        super(AttNet, self).__init__()
        self.class_weight_color = class_weight_color
        self.class_weight_os = class_weight_os
        print('**** AttNet5.py config ****')

        self.use_class = use_class
        self.use_vis = use_vis
        self.dropout = False
        nembedding = 20
        print('# use_label:', self.use_class)
        print('# visual_feature:', self.use_vis)
        print('nembedding: ', nembedding)
        print('nlabel: ', nlabel)
        print('ncolor: ', ncolor)
        print('nopen_state: ', nopen_state)
        print('dropout: ', self.dropout)
        print('************************')
        
        combine_feature_size = 0
        
        if self.use_class:
            self.word_embedding = nn.Embedding(nlabel, nembedding)
            self.wf = FC(20, 32)
            combine_feature_size += 32
            
        if self.use_vis:
            self.roi_pool_object = RoIPool(32, 32, 1.0)
            # cnn
            bn = False
            self.conv1 = []
            self.conv2_top = []
            self.conv2_bot = []
            self.conv3 = []
            self.conv4_top = []
            self.conv4_bot = []
            self.conv5_top = []
            self.conv5_bot = []
    
            [nn.Sequential(Conv2d(3, 48, 3, same_padding=True, bn=bn), nn.MaxPool2d(2)).cuda() for i in range(2)]
            for i in range(2):
                self.conv1.append(nn.Sequential(Conv2d(3, 48, 3, same_padding=True, bn=bn),
                                         nn.MaxPool2d(2)).cuda())  # 32x32x3 -> 16x16x48
                self.conv2_top.append(nn.Sequential(Conv2d(24, 64, 3, same_padding=True, bn=bn),
                                         nn.MaxPool2d(2)).cuda())  # 16x16x24 -> 8x8x64
                self.conv2_bot.append(nn.Sequential(Conv2d(24, 64, 3, same_padding=True, bn=bn),
                                         nn.MaxPool2d(2)).cuda())  # 16x16x24 -> 8x8x64
                self.conv3.append(Conv2d(128, 192, 3, same_padding=True, bn=False).cuda())  # 8x8x128 -> 8x8x192
                self.conv4_top.append(Conv2d(96, 96, 3, same_padding=True, bn=False).cuda())  # 8x8x96 -> 8x8x96
                self.conv4_bot.append(Conv2d(96, 96, 3, same_padding=True, bn=False).cuda())  # 8x8x96 -> 8x8x96
                self.conv5_top.append(nn.Sequential(Conv2d(96, 64, 3, same_padding=True, bn=False),
                                         nn.MaxPool2d(2)).cuda())  # 8x8x96 -> 4x4x64
                self.conv5_bot.append(nn.Sequential(Conv2d(96, 64, 3, same_padding=True, bn=False),
                                         nn.MaxPool2d(2)).cuda())  # 8x8x96 -> 4x4x64
    
            self.m_conv1 = nn.ModuleList(self.conv1)
            self.m_conv2_top = nn.ModuleList(self.conv2_top)
            self.m_conv2_bot = nn.ModuleList(self.conv2_bot)
            self.m_conv3 = nn.ModuleList(self.conv3)
            self.m_conv4_top = nn.ModuleList(self.conv4_top)
            self.m_conv4_bot = nn.ModuleList(self.conv4_bot)
            self.m_conv5_top = nn.ModuleList(self.conv5_top)
            self.m_conv5_bot = nn.ModuleList(self.conv5_bot)

            combine_feature_size += 4*4*64*4

        self.fc1_color = FC(combine_feature_size, 4096)
        
        self.fc2_color = FC(4096, 4096)
        self.pred_color = FC(4096, ncolor, relu=False)

        self.fc1_open_state = FC(combine_feature_size, 300)
        self.fc2_open_state = FC(300, 300)
        self.pred_open_state = FC(300, nopen_state, relu=False)

        network.weights_normal_init(self.pred_color, 0.01)
        network.weights_normal_init(self.pred_open_state, 0.01)

    def forward(self, im_data, im_info, objects, gt_colors=None, gt_open_states=None):
        # objects.size() : (obj#, 5[xmin, ymin, xmax, ymax, obj_class]) t:numpy
        #print('attnet4 img:', im_data)
        #print('attnet4 objs:', objects)
        if PRINT_FEATURE_VALUES:
            print('im_data.sum():', im_data.sum())
        if PRINT_MODEL_PROCESS:
            print('im_data.shape:', im_data.shape)
            print('gt_objects.shape:', objects.shape)
        bboxes = np.zeros((len(objects), 5))
        # bboxes : (obj#, 5[0, xmin, ymin, xmax, ymax])
        bboxes[:, 1:] = objects[:, :4] # roi pooling 함수를 위해 0을 넣음
        bboxes = network.np_to_variable(bboxes, is_cuda=True)
        labels = network.np_to_variable(np.expand_dims(objects[:, 4], -1), is_cuda=True)
        labels = labels.long()
        
        features = []

        ### common img ###
        # im_data : (1, 3, 600, 600)
        # roi_pool_object에 들어가는 좌표 데이터는 (obj#, 5[0, xmin, ymin, xmax, ymax]), cuda_tensor 여야함
        if self.use_vis:
            obj_vis = self.roi_pool_object(im_data, bboxes)
            #print('obj_vis:', obj_vis)
            if PRINT_FEATURE_VALUES:
                print('obj_vis.sum():', obj_vis.sum())
                print(obj_vis.size())
            if PRINT_MODEL_PROCESS:
                print('obj_vis', obj_vis.size())

            vf = []
            for i in range(2):
                # print('obj_vis:', obj_vis.size())
                c1 = self.m_conv1[i](obj_vis)
                if PRINT_FEATURE_VALUES:
                    print('c1.sum():', c1.sum())
                    print('c1:', c1.size())
                c1_top, c1_bot = c1[:, 24:], c1[:, :24]
                c2_top = self.m_conv2_top[i](c1_top)
                if PRINT_FEATURE_VALUES:
                    print('c2_top.sum():', c2_top.sum())
                c2_bot = self.m_conv2_bot[i](c1_bot)
                if PRINT_FEATURE_VALUES:
                    print('c2_bot.sum():', c2_bot.sum())
                c2 = torch.cat((c2_top, c2_bot), 1)
                if PRINT_FEATURE_VALUES:
                    print('c2.sum():', c2.sum())
                    print('c2:', c2.size())
                c3 = self.m_conv3[i](c2)
                if PRINT_FEATURE_VALUES:
                    print('c3.sum():', c3.sum())

                c3_top, c3_bot = c3[:, 96:], c3[:, :96]
                c4_top = self.m_conv4_top[i](c3_top)
                if PRINT_FEATURE_VALUES:
                    print('c4_top.sum():', c4_top.sum())
                c4_bot = self.m_conv4_bot[i](c3_bot)
                if PRINT_FEATURE_VALUES:
                    print('c4_bot.sum():', c4_bot.sum())
                c5_top = self.m_conv5_top[i](c4_top)
                if PRINT_FEATURE_VALUES:
                    print('c5_top.sum():', c5_top.sum())
                c5_bot = self.m_conv5_bot[i](c4_bot)
                if PRINT_FEATURE_VALUES:
                    print('c5_bot.sum():', c5_bot.sum())
                vf.append(c5_top)
                vf.append(c5_bot)
                if PRINT_FEATURE_VALUES:
                    print('c5_top:', c5_top.size())
                    print('c5_bot:', c5_bot.size())

            vf = torch.cat(vf, 1)
            vf = vf.view(obj_vis.size()[0], -1)
            features.append(vf)
            
            
        ### word ###
        if self.use_class:
            word_vector = self.word_embedding(labels)
            word_vector = word_vector.view(labels.size()[0], -1)
            wf = self.wf(word_vector)
            features.append(wf)

        #### color ####
        if len(features) > 1:
            c_feature = torch.cat(features, 1)
        else:
            c_feature = features[0]
            
        if PRINT_FEATURE_VALUES:
            print('c_feature.sum():', c_feature.sum())

        color_fc1 = self.fc1_color(c_feature)
        if PRINT_FEATURE_VALUES:
            print('color_fc1:', color_fc1.sum())
        color_fc1 = F.dropout(color_fc1, training=self.training)
        if PRINT_FEATURE_VALUES:
            print('color_fc1_dropout:', color_fc1.sum())
        color_fc2 = self.fc2_color(color_fc1)
        if PRINT_FEATURE_VALUES:
            print('color_fc2:', color_fc1.sum())
        color_fc2 = F.dropout(color_fc2, training=self.training)
        if PRINT_FEATURE_VALUES:
            print('color_fc2_dropout:', color_fc1.sum())

        color_score = self.pred_color(color_fc2)
        color_prob = F.softmax(color_score, -1)

        #### open_state ####

        open_state_fc1 = self.fc1_open_state(c_feature)
        open_state_fc1 = F.dropout(open_state_fc1, training=self.training)
        open_state_fc2 = self.fc2_open_state(open_state_fc1)
        open_state_fc2 = F.dropout(open_state_fc2, training=self.training)

        open_state_score = self.pred_open_state(open_state_fc2)
        open_state_prob = F.softmax(open_state_score, -1)

        if self.training:
            self.att_loss, (self.ce_color, self.ce_open_state) = self.build_loss_att(color_score, open_state_score,
                                                                    gt_colors.cuda(), gt_open_states.cuda())
        return color_prob, open_state_prob


    def evaluate_acc(self, im_data, im_info, objects, gt_colors, gt_open_states):
        # original code
        color_prob, open_state_prob = self(im_data, im_info, objects, gt_colors, gt_open_states)

        color_cnt, color_correct_cnt = self.get_recall(color_prob, gt_colors)
        os_cnt, os_correct_cnt = self.get_recall(open_state_prob, gt_open_states)

        return (color_cnt, color_correct_cnt), (os_cnt, os_correct_cnt)
    
    def evaluate(self, im_data, im_info, objects, gt_colors, gt_open_states):
        # original code
        color_prob, open_state_prob = self(im_data, im_info, objects, gt_colors, gt_open_states)
        def get_result(pred_prob, gt):
            pred_prob = pred_prob.data.cpu().numpy()
            gt = gt.data.cpu().numpy()
        
            pred = pred_prob.argmax(1)
            return pred, gt
        color_pred, color_gt = get_result(color_prob, gt_colors)
        os_pred, os_gt = get_result(open_state_prob, gt_open_states)
        
        return (color_pred, color_gt), (os_pred, os_gt)

    def visualize(self, im_data, im_info, objects, gt_colors, gt_open_states):
        print('im_data:', im_data)
        print('im_data.dtype:', im_data.dtype)
        print('im_data.shape:', im_data.shape)
        print('objects:', objects)
        empty_color = torch.LongTensor([0 for _ in objects])  # eval하면  성능이 애꾸되는 버그때문에 넣어줌
        empty_os = torch.LongTensor([0 for _ in objects])

        color_prob, open_state_prob = self(im_data, im_info, objects, empty_color, empty_os) #######

        color_cnt, color_correct_cnt = self.get_recall(color_prob, gt_colors)
        os_cnt, os_correct_cnt = self.get_recall(open_state_prob, gt_open_states)
        print('color_recall:', float(color_correct_cnt) / color_cnt * 100)
        print(' os_recall:', float(os_correct_cnt) / os_cnt * 100)
        frame = np.transpose(im_data[0], (1, 2, 0))

        bbox_fig = plt.figure(figsize=(6.0, 6.0))
        ax = plt.Axes(bbox_fig, [0., 0., 1., 1.])
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        bbox_fig.add_axes(ax)

        ax.cla()
        ax.imshow(frame)

        color_prob = color_prob.data.cpu().numpy()
        open_state_prob = open_state_prob.data.cpu().numpy()

        color = color_prob.argmax(1)
        os = open_state_prob.argmax(1)


        print('output[color_prob]:', color_prob)
        print('output[color]:', color)
        print('output[os_prob]:', open_state_prob)
        print('output[os]:', os)
        for i in range(len(objects)):
            rect = patches.Rectangle((objects[i][0], objects[i][1]), objects[i][2] - objects[i][0],
                                     objects[i][3] - objects[i][1], linewidth=6, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(objects[i][0], objects[i][1] - 9, '{}, {}, {}'.format(int(objects[i][4]), color[i], os[i]),
                    style='italic',
                    bbox={'alpha': 0.5}, fontsize=15)


        plt.show()
        #plt.savefig(fname='temp_od.jpg', bbox_inches='tight', pad_inches=0)
        #self.OdDialog.show_image(filePath='temp_od.jpg')



    @staticmethod
    def get_recall(pred_prob, gt):
        pred_prob = pred_prob.data.cpu().numpy()
        gt = gt.data.cpu().numpy()

        gt_cnt = len(gt)

        pred = pred_prob.argmax(1)
        rel_correct_cnt = np.sum(np.equal(pred, gt))

        return gt_cnt, rel_correct_cnt

    def build_loss_att(self, color_score, open_state_score, gt_colors, gt_open_states):
        if self.class_weight_color is not None:
            #color_weights = np.sqrt(self.class_weight_color)
            color_weights = self.class_weight_color.cuda()  # 값 너무 큰거는 0으로 대체해버림
            ce_color = F.cross_entropy(color_score, gt_colors, weight=color_weights)
        else:
            ce_color = F.cross_entropy(color_score, gt_colors)
        if self.class_weight_os is not None:
            #os_weights = np.sqrt(self.class_weight_os)
            os_weights = self.class_weight_os.cuda()
            ce_open_state = F.cross_entropy(open_state_score, gt_open_states, weight=os_weights)
        else:
            ce_open_state = F.cross_entropy(open_state_score, gt_open_states)
        total_loss = 1.5 * ce_color + ce_open_state

        return total_loss, (ce_color, ce_open_state)

