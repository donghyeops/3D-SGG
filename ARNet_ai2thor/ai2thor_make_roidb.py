# -*- coding: utf-8 -*-

import os
import os.path as osp
import json
import cv2
import glob
import copy
from matplotlib import pyplot as plt

print('os.getcwd():', os.getcwd())

import sys
sys.path.append('/home/ailab/DH/ai2thor')
temp_path = os.getcwd()
os.chdir('/home/ailab/DH/ai2thor')
from thor_utils import annotation_util as au
os.chdir(temp_path)

class roi_maker:
    def __init__(self, ann_path, image_path):
        self.ann_path = ann_path
        self.image_path = osp.join(image_path, 'images')
        self.depth_path = osp.join(image_path, 'depth_images')

        self.roi_img_path = osp.join(image_path, 'roi_images')
        self.roi_depth_path = osp.join(image_path, 'roi_depth_images')

        os.makedirs(self.roi_img_path, exist_ok=True)
        os.makedirs(self.roi_depth_path, exist_ok=True)

        # load category names and annotations

        self.ann_file_names = ['train.json', 'test.json']
        self.new_json_prefix = 'roi_' # new_train.json

    def make_roi_db(self, only_json=False):
        roi_id = 0
        for idx, split in enumerate(self.ann_file_names):
            ann_file_path = osp.join(self.ann_path, split)
            anns = json.load(open(ann_file_path))
            roi_anns = []
            for i, ann in enumerate(anns):
                if not only_json:
                    img = cv2.imread(osp.join(self.image_path, ann['path']))  # 0~255
                    depth = cv2.imread(osp.join(self.depth_path, ann['depth_path']),
                                       cv2.IMREAD_GRAYSCALE)  # 0~255
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                common_roi_ann = {}
                common_roi_ann['scene_name'] = ann['scene_name']
                common_roi_ann['image_id'] = ann['id']
                common_roi_ann['path'] = ann['path']
                common_roi_ann['depth_path'] = ann['depth_path']
                common_roi_ann['agent'] = ann['agent']


                for j, obj in enumerate(ann['objects']):
                    roi_ann = copy.copy(common_roi_ann)
                    roi_ann['object_info'] = obj
                    roi_ann['object_info']['obj_id'] = j

                    roi_img_file = 'roi_img_' + roi_ann['scene_name'] + '_' + str(roi_id) + '.jpg'
                    roi_depth_file = 'roi_depth_' + roi_ann['scene_name'] + '_' + str(roi_id) + '.jpg'
                    if not only_json:
                        box = obj['box']
                        roi_img = img[box[1]:box[3], box[0]:box[2]]
                        roi_depth = depth[box[1]:box[3], box[0]:box[2]]

                        cv2.imwrite(osp.join(self.roi_img_path, roi_img_file), roi_img)
                        cv2.imwrite(osp.join(self.roi_depth_path, roi_depth_file), roi_depth)

                    roi_ann['roi_image_path'] = roi_img_file
                    roi_ann['roi_depth_path'] = roi_depth_file
                    roi_ann['roi_id'] = roi_id
                    roi_anns.append(roi_ann)
                    roi_id += 1
                if i % 500 == 0:
                    print(f'processing ... [{split.split(".")[0]}][{i}/{len(anns)}]')
                    
            #print_dict(roi_anns)
            with open(osp.join(self.ann_path, self.new_json_prefix+split), 'w') as f:
                json.dump(roi_anns, f, indent='\t')
            print(f'{split}({len(roi_anns)}) done')

def print_dict(data, key=''):
    if isinstance(data, dict):
        for k, v in data.items():
            print_dict(v, key=k)
    elif isinstance(data, list):
        for v in data:
            #print(f'K:{key}, T:{type(data)}, V:...')
            print_dict(v)
    else:
        if not (data is None \
            or isinstance(data, int) \
            or isinstance(data, float) \
            or isinstance(data, str)):
            print(f'K:{key}, T:{type(data)}, V:{data}')


class base_maker:
    # 학습 데이터와 평가데이터를 나눔
    # 학습 및 평가용 데이터로 변환해줌
    def __init__(self, gsg_path, output_path='./data'):
        self.gsg_path = gsg_path
        self.output_path = output_path
        if not os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)

        self.test_set = ['FloorPlan10', 'FloorPlan20']

    def make_base_db(self):
        json_paths = glob.glob(f'{self.gsg_path}/**/*.json')
        train_db = []
        test_db = []

        segment_id = 0
        for json_path in json_paths:

            with open(json_path, 'r') as f:
                data = json.load(f)
            common_segment = {}
            common_segment['height'] = 600
            common_segment['width'] = 600
            common_segment['scene_name'] = data['scene_name']
            for key, gsg in data['gsg_history'].items():
                segment = copy.copy(common_segment)
                segment['id'] = segment_id
                segment['path'] = f'{segment["scene_name"]}_{key}.jpg'
                segment['depth_path'] = f'{segment["scene_name"]}_d_{key}.jpg'

                segment['agent'] = self.__get_agent_info(gsg['agent'])
                segment['objects'], obj_gid2lid = self.__get_object_info(gsg['objects'])
                segment['global_relationships'] = self.__get_relation_info(gsg['relations'], gsg['objects'], obj_gid2lid)
                if segment['scene_name'] in self.test_set:
                    test_db.append(segment)
                else:
                    train_db.append(segment)
                segment_id += 1

        with open(osp.join(self.output_path, 'train.json'), 'w') as f:
            json.dump(train_db, f, indent='\t')
        print(f'generate train.json (#{len(train_db)}) path:{open(osp.join(self.output_path, "train.json"))}')
        with open(osp.join(self.output_path, 'test.json'), 'w') as f:
            json.dump(test_db, f, indent='\t')
        print(f'generate test.json (#{len(test_db)}) path:{open(osp.join(self.output_path, "test.json"))}')

    def __get_agent_info(self, low_agent):
        agent = {}
        agent['global_rotation'] = int(low_agent['rotation']['y']/90)
        agent['global_position'] = list(low_agent['position'].values())
        cameraHorizon_dict = {60:0, 30:1, 0:2, -30:3} # 0은 최하단, 2는 정면, 3은 상단
        if 'cameraHorizon' in low_agent:
            agent['cameraHorizon'] = cameraHorizon_dict[int(low_agent['cameraHorizon'])]
        return agent

    def __get_object_info(self, low_objects):
        objects = []
        obj_gid2lid = {}
        lid=-1
        for low_obj in low_objects.values():
            if not low_obj['detection']:
                continue
            lid += 1
            obj_gid2lid[low_obj['id']] = lid
            obj = {}
            # au.obj_i2s, au.rel_i2s, au.color_i2s, openState_i2s
            obj['class'] = au.obj_i2s[low_obj['label']]
            obj['box'] = list(map(int, low_obj['box2d']))
            obj['global_position'] = low_obj['box3d'][:3]
            obj['size_3d'] = low_obj['box3d'][3:]
            obj['distance'] = -1.
            obj['color'] = au.color_i2s[low_obj['color']]
            #obj['detection'] = low_obj['detection']
            if 'open_state' in obj:
                obj['open_state'] = au.color_i2s[low_obj['open_state']]
            else:
                obj['open_state'] = 'unable'
            objects.append(obj)
        return objects, obj_gid2lid

    def __get_relation_info(self, low_relation, low_objects, obj_gid2lid):
        relations = []
        for k, v in low_relation.items():
            rel = {}
            if not (low_objects[str(v['subject_id'])]['detection'] and low_objects[str(v['object_id'])]['detection']):
                continue
            rel['sub_id'] = obj_gid2lid[v['subject_id']]
            rel['predicate'] = au.rel_i2s[v['rel_class']]
            rel['obj_id'] = obj_gid2lid[v['object_id']]
            relations.append(rel)
        return relations


def merge_two_roi_db(gt_roi_path, pred_roi_path, output_path):
    gt_roi_train = json.load(open(osp.join(gt_roi_path, 'gt_roi_train.json')))
    gt_roi_test = json.load(open(osp.join(gt_roi_path, 'gt_roi_test.json')))
    pred_roi_train = json.load(open(osp.join(pred_roi_path, 'pred_roi_train.json')))
    pred_roi_test = json.load(open(osp.join(pred_roi_path, 'pred_roi_test.json')))

    def merge_db(gt_roi, pred_roi):
        for roi in gt_roi:
            roi['source'] = 'gt'
        for roi in pred_roi:
            roi['source'] = 'pred'

        return gt_roi + pred_roi
    merge_roi_train = merge_db(gt_roi_train, pred_roi_train)
    merge_roi_test = merge_db(gt_roi_test, pred_roi_test)

    with open(osp.join(output_path, 'merge_roi_train.json'), 'w') as f:
        json.dump(merge_roi_train, f, indent='\t')
    print(f'generate train.json (#{len(merge_roi_train)}) path:{open(osp.join(output_path, "merge_roi_train.json"))}')
    with open(osp.join(output_path, 'merge_roi_test.json'), 'w') as f:
        json.dump(merge_roi_test, f, indent='\t')
    print(f'generate test.json (#{len(merge_roi_test)}) path:{open(osp.join(output_path, "merge_roi_test.json"))}')


if __name__ == '__main__':
    Base_Maker = base_maker(gsg_path='/home/ailab/DH/ai2thor/datasets/thorDBv2_gsg_gt/gsg',
                             output_path='/home/ailab/DH/ai2thor/ARNet_ai2thor/data/thorDBv2')
    Base_Maker.make_base_db()
    #
    # ROI_Maker = roi_maker(ann_path='/home/ailab/DH/ai2thor/ARNet_ai2thor/data/thorDBv2_objnet',
    #                       image_path='/media/ailab/D/ai2thor/thorDBv2')
    # ROI_Maker.make_roi_db(only_json=False)

    # merge_two_roi_db(gt_roi_path='/home/ailab/DH/ai2thor/ARNet_ai2thor/data/thorDBv2/roi_db',
    #                  pred_roi_path='/home/ailab/DH/ai2thor/ARNet_ai2thor/data/thorDBv2/roi_db',
    #                  output_path='/home/ailab/DH/ai2thor/ARNet_ai2thor/data/thorDBv2/roi_db')