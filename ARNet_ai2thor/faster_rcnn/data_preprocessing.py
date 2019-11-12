# -*- coding: utf-8 -*-
import json
import os.path as osp
import cv2
import numpy as np
import time
from pprint import pprint
import random


# msdn style !!!
def generate_DB(dataPath='./thor_DB.json', categoryPath='./categories.json', \
                newDataPath='./thor_DB_msdn.json'):
    # msdn style !!!
    # GUI 툴로 생성한 ai2thor DB를 msdn 폼으로 재구성하여 저장됨

    print('-- generate thor_DB file --')

    with open(dataPath, 'r') as f:
        # DB = json.load(f)
        DB = json.load(f)
    with open(categoryPath, 'r') as f:
        categories = json.load(f)
    print('data read complete')
    print('DB len:', len(DB))

    new_DB = []
    for idx, data in enumerate(DB):
        if len(data['visible_object']) == 0:  # 물체가 없으면 넘김
            continue
        if idx % 10000 == 0:
            print('data preprocessing... ({}/{})'.format(idx, len(DB)))
        new_data = {}
        new_data['height'] = 600
        new_data['width'] = 600
        new_data['scene_name'] = data['scene']
        new_data['id'] = data['data_id']
        new_data['path'] = data['image_file']
        new_data['depth_path'] = data['depth_file']

        agent = {}
        global_rotation = data['agent']['global_rotation']['y']  # float 형태 [마이너스는 없음]
        if global_rotation == 0:  # 정면
            global_rotation = 0  # int 형태
        elif global_rotation == 90:  # 오른쪽
            global_rotation = 1
        elif global_rotation == 180:  # 뒤
            global_rotation = 2
        elif global_rotation == 270:  # 왼쪽
            global_rotation = 3
        agent['global_rotation'] = global_rotation
        agent['global_position'] = [data['agent']['global_position']['x'], data['agent']['global_position']['y'],
                                    data['agent']['global_position']['z']]
        new_data['agent'] = agent

        objects = []
        obj_g2l = {}  # global idx -> local idx
        for idx, obj in enumerate(data['visible_object']):
            b_3d = obj['global_bounds3D']
            size_3d = [b_3d[3] - b_3d[0], b_3d[4] - b_3d[1], b_3d[5] - b_3d[2]]

            objects.append({'class': categories['object'][obj['obj_class'] - 1], \
                            'box': obj['bounding_box'], \
                            'global_position': [obj['global_position']['x'], obj['global_position']['y'],
                                                obj['global_position']['z']], \
                            'size_3d': size_3d, \
                            'distance': obj['distance'], \
                            'color': categories['color'][obj['color']], \
                            'open_state': categories['open_state'][obj['open_state']]})
            # 'off_state':categories['off_state'][obj['off_state']]})
            # index는 string label로 변경
            # index가 안맞을 수 있음. 확인요망
            obj_g2l[obj['id']] = idx
        new_data['objects'] = objects
        relationships = []
        for rel in data['relation']:  # relation...!!!!
            relationships.append({'sub_id': obj_g2l[rel['subject_id']], \
                                  'predicate': categories['predicate'][rel['rel_class'] - 1], \
                                  'obj_id': obj_g2l[rel['object_id']]})
        new_data['relationships'] = relationships
        global_relationships = []
        for rel in data['global_relation']:  # global relation...!!!!
            global_relationships.append({'sub_id': obj_g2l[rel['subject_id']], \
                                  'predicate': categories['predicate'][rel['rel_class'] - 1], \
                                  'obj_id': obj_g2l[rel['object_id']]})
        new_data['global_relationships'] = global_relationships
        new_DB.append(new_data)
    print('data preprocessing complete')
    print('new_DB len:', len(new_DB))
    print('data dumpping...')
    with open(newDataPath, 'w') as f:
        json.dump(new_DB, f, indent='\t')
    print('data dumpping complete')
    print('file path:', newDataPath)


def generate_categoriesFile(dataPath='./class', newDataPath='./categories.json'):
    # object, rel, color 등의 클래스 별로 파일을 나누었는데,
    # msdn의 폼에서는 categories 파일에 다 저장됨
    # 이를 위해 여러 클래스 정의 파일들을 categories.json으로 통일시킴

    print('-- generating categories file --')
    with open(osp.join(dataPath, 'objects.txt'), 'r') as f:
        object_label = f.read().split('\n')
    with open(osp.join(dataPath, 'relationships.txt'), 'r') as f:
        relationship_label = f.read().split('\n')
    with open(osp.join(dataPath, 'colors.txt'), 'r') as f:
        color_label = f.read().split('\n')
    with open(osp.join(dataPath, 'open_states.txt'), 'r') as f:
        openState_label = f.read().split('\n')
    with open(osp.join(dataPath, 'off_states.txt'), 'r') as f:
        offState_label = f.read().split('\n')
    categories = {}
    categories['object'] = object_label[1:]  # background 제거
    categories['predicate'] = relationship_label[1:]  # background 제거
    categories['color'] = color_label
    categories['open_state'] = openState_label
    # categories['off_state'] = offState_label
    with open(newDataPath, 'w') as f:
        json.dump(categories, f, indent='\t')
    print('data dumpping complete')
    print('file path:', newDataPath)


def get_image_info(dataPath='./thor_DB_msdn.json', imageDir='./images', depth_image=False):
    # DB 파일에 속한 영상들의 평균과 표준 편차를 계산
    print('-- get image info --')
    print('image path : {}'.format(imageDir))
    with open(dataPath, 'r') as f:
        DB = json.load(f)
    print('data read complete')
    len_DB = len(DB)
    print('DB len:', len_DB)
    max_value = 255.
    mean = np.zeros(3)
    std = np.zeros(3)

    start_time = time.time()
    for idx, data in enumerate(DB):
        if idx % 1000 == 0 and idx != 0:
            dt = time.time() - start_time
            print('data preprocessing... ({}/{}), {} fps, {}s left..'.format(idx, len(DB), \
                                                                             round(1000. / dt, 2),
                                                                             (len(DB) - idx) * round(dt / 1000., 4)))
            start_time = time.time()

        if depth_image:
            if 'path' in data:
                img_fn = data['depth_path']
            elif 'image_file' in data:
                img_fn = data['depth_file']
            else:
                raise Exception('where is the image path !!')
        else:
            if 'path' in data:
                img_fn = data['path']
            elif 'image_file' in data:
                img_fn = data['image_file']
            else:
                raise Exception('where is the image path !!')

        img = cv2.imread(osp.join(imageDir, img_fn))

        for i in range(3):
            mean[i] += img[:, :, i].mean() / max_value
            std[i] += img[:, :, i].std() / max_value
    mean /= len_DB
    std /= len_DB

    print('mean :', mean.round(3))
    print('std :', std.round(3))

    return mean, std


def divide_DB_by_scene(dataPath='./thor_DB_msdn.json'):
    # msdn 폼의 DB를 train과 test 집합으로 나눔 [test scene을 설정하고 나눔]
    print('-- divide DB --')
    with open(dataPath, 'r') as f:
        DB = json.load(f)

    test_scene = ['FloorPlan10', 'FloorPlan11']
    train_DB = []
    test_DB = []
    for idx, data in enumerate(DB):
        if idx % 10000 == 0:
            print('data preprocessing... ({}/{})'.format(idx, len(DB)))
        if data['scene_name'] in test_scene:
            test_DB.append(data)
        else:
            train_DB.append(data)

    print('train.json#:', len(train_DB))
    print('test.json#:', len(test_DB))

    with open('./train.json', 'w') as f:
        json.dump(train_DB, f, indent='\t')
    print('generate train.json')
    with open('./test.json', 'w') as f:
        json.dump(test_DB, f, indent='\t')
    print('generate test.json')


def divide_DB_by_random(dataPath='./thor_DB_msdn.json', test_rate=0.2):
    # msdn 폼의 DB를 train과 test 집합으로 나눔 [test set 비율만 정하고 랜덤으로 나눔]
    print('-- divide DB --')
    with open(dataPath, 'r') as f:
        DB = json.load(f)
    num_data = len(DB)
    test_idx = random.sample(list(range(num_data)), int(num_data * test_rate))
    test_idx.sort()

    train_DB = []
    test_DB = []
    for idx, data in enumerate(DB):
        if idx % 10000 == 0:
            print('data preprocessing... ({}/{})'.format(idx, len(DB)))
        if idx in test_idx:
            test_DB.append(data)
        else:
            train_DB.append(data)

    print('train.json#:', len(train_DB))
    print('test.json#:', len(test_DB))

    with open('./train.json', 'w') as f:
        json.dump(train_DB, f, indent='\t')
    print('generate train.json')
    with open('./test.json', 'w') as f:
        json.dump(test_DB, f, indent='\t')
    print('generate test.json')


def get_class_weights(dataPath='./train.json'):
    print('-- get_class_info --')
    with open(dataPath, 'r') as f:
        DB = json.load(f)
    obj_dict = {}
    color_dict = {}
    os_dict = {}
    rel_dict = {}  # 현재 negative는 아예 고려를 안하고 positive에서만 구함 (나머지도 마찬가지인데, 나머지는 상관없음)
    for idx, data in enumerate(DB):
        for obj in data['objects']:
            if obj['class'] in obj_dict:
                obj_dict[obj['class']] += 1
            else:
                obj_dict[obj['class']] = 1
            if obj['color'] in color_dict:
                color_dict[obj['color']] += 1
            else:
                color_dict[obj['color']] = 1
            if obj['open_state'] in os_dict:
                os_dict[obj['open_state']] += 1
            else:
                os_dict[obj['open_state']] = 1
        if len(data['objects']) > 0:
            rel_dict['background'] = len(data['objects']) * (len(data['objects'])-1) - len(data['global_relationships'])
        for rel in data['global_relationships']:
            if rel['predicate'] in rel_dict:
                rel_dict[rel['predicate']] += 1
            else:
                rel_dict[rel['predicate']] = 1

    output = [obj_dict, color_dict, os_dict, rel_dict]
    print('results')
    pprint(output)

    def get_weight(class_dict):
        values = np.array(list(class_dict.values())).astype(float)
        weights = np.zeros_like(values)
        print(values)
        sum = values.sum()
        for i, v in enumerate(values):
            values[i] /= sum

        max = values.max()
        for i, v in enumerate(values):
            weights[i] = max / values[i]

        weights_dict = dict(zip(class_dict.keys(), weights.tolist()))
        return weights_dict

    print('\nweights')
    all_weights = {}
    all_weights['object'] = get_weight(obj_dict)
    all_weights['color'] = get_weight(color_dict)
    all_weights['open_state'] = get_weight(os_dict)
    all_weights['relationship'] = get_weight(rel_dict)
    pprint(all_weights['object'])
    pprint(all_weights['color'])
    pprint(all_weights['open_state'])
    pprint(all_weights['relationship'])

    with open('./class_weights.json', 'w') as f:
        json.dump(all_weights, f, indent='\t')
    print('generate class_weights.json')


def make_color_balance(dataPath='./thor_DB_msdn.json'):
    print('-- make_color_balance --')
    with open(dataPath, 'r') as f:
        DB = json.load(f)
    idx_for_unique_cls = {}
    len_for_unique_cls = {}

    for idx, data in enumerate(DB):
        color_counts = {}
        for obj in data['objects']:
            if obj['color'] in color_counts:
                color_counts[obj['color']] += 1
            else:
                color_counts[obj['color']] = 1
        values = np.array(list(color_counts.values()))

        if sum(values > 0) == 1:
            unique_cls = values.argmax()
            cls = list(color_counts.keys())[unique_cls]

            if cls in idx_for_unique_cls:
                idx_for_unique_cls[cls].append(idx)
                len_for_unique_cls[cls] += values.max()
            else:
                idx_for_unique_cls[cls] = [idx]
                len_for_unique_cls[cls] = values.max()

    # pprint(idx_for_unique_cls)
    for k, v in len_for_unique_cls.items():
        print(k + ': ' + str(v))


def generate_prior_knowledge(dataPath='./thor_DB_msdn.json'):
    print('-- generate_prior_knowledge --')
    with open(dataPath) as f:
        db = json.load(f)

    prior_knowledge = {}
    obj_category = dict()
    for data in db:
        objs = data['objects']
        for obj in objs:
            if obj['class'] in obj_category:
                if obj['size_3d'][0] >= obj['size_3d'][2]:  # 회전된 물체 통일하기
                    size = [obj['size_3d'][0], obj['size_3d'][1], obj['size_3d'][2]]
                else:
                    size = [obj['size_3d'][2], obj['size_3d'][1], obj['size_3d'][0]]
                obj_category[obj['class']]['size_3d'].append(size)
            else:
                obj_category[obj['class']] = {}
                if obj['size_3d'][0] >= obj['size_3d'][2]:  # 회전된 물체 통일하기
                    size = [obj['size_3d'][0], obj['size_3d'][1], obj['size_3d'][2]]
                else:
                    size = [obj['size_3d'][2], obj['size_3d'][1], obj['size_3d'][0]]
                obj_category[obj['class']]['size_3d'] = [size]
    for k, v in obj_category.items():
        # print(np.array(v['size_3d']).mean(0))
        obj_category[k]['size_3d'] = np.array(v['size_3d']).mean(0).tolist()

    prior_knowledge['objects'] = obj_category
    with open('./prior_knowledge.json', 'w') as f:
        json.dump(prior_knowledge, f, indent='\t')
    print('generate prior_knowledge.json')


if __name__ == '__main__':
    # generate_categoriesFile()
    #generate_DB(dataPath='/media/ailab/D/ai2thor/thor_DB.json')
    # divide_DB_by_scene(dataPath='./thor_DB_msdn.json')
    #divide_DB_by_random(dataPath='./thor_DB_msdn.json', test_rate=0.1)
    # get_image_info(imageDir='/media/ailab/D/ai2thor/images')
    # get_image_info(imageDir='/media/ailab/D/ai2thor/depth_images', depth_image=True)
    get_class_weights(dataPath='/home/ailab/DH/ai2thor/ARNet_ai2thor/data/thorDBv2/train.json')
    # make_color_balance(dataPath='./thor_DB_msdn.json')
    #generate_prior_knowledge(dataPath='./thor_DB_msdn.json')
