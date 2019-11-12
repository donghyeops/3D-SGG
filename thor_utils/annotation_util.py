#-*- coding:utf-8 -*-

from math import sqrt
import colormap
import numpy as np

# class type 호출
with open('./class/objects.txt', 'r') as f:  # background 존재
    object_label = f.read().split('\n')
with open('./class/relationships.txt', 'r') as f:  # background 존재
    relationship_label = f.read().split('\n')
with open('./class/colors.txt', 'r') as f:  # unkown 존재
    color_label = f.read().split('\n')
with open('./class/open_states.txt', 'r') as f:
    openState_label = f.read().split('\n')

def make_label_dict(label_list):
    s2i = {value: idx for idx, value in enumerate(label_list)} # c2n
    i2s = {idx:value for idx, value in enumerate(label_list)} # n2c
    return s2i, i2s

# class label 참조 딕셔너리 생성 [class label to number: s2i]
obj_s2i, obj_i2s = make_label_dict(object_label)
rel_s2i, rel_i2s = make_label_dict(relationship_label)
color_s2i, color_i2s = make_label_dict(color_label)
openState_s2i, openState_i2s = make_label_dict(openState_label)
if 'left_of' in relationship_label and 'right_of' in relationship_label:
    print('include left/right relationships !!')
    INCLUDE_LR = True
else:
    INCLUDE_LR = False

def get_on_objects():
    # on에 해당하는 물체들
    return ['Tabletop', 'Countertop']

def get_relations(objects, objects_id, agent_info, get_global=False):
    local_relation = get_local_relations(objects, objects_id, agent_info)
    if not get_global:
        return local_relation

    global_relation = get_global_relations(objects, objects_id)
    return local_relation, global_relation

def get_local_relations(objects, objects_id, agent_info):
    # object 간의 관계를 구해서 리턴
    relations = []
    in_objects = get_on_objects()
    UD_xz_threshold = 0.3
    UD_y_threshold = 0.3

    FB_xy_threshold = 150
    FB_z_threshold = 0.3
    agent_height = agent_info['position']['y']

    LR_xz_threshold = 0.3
    agent_angle = agent_info['rotation']['y']

    done_pair = [] # 관계가 있음을 등록. 우선순위에 따라 하나의 관계만 가질 수 있도록 함

    # in/on 관계
    for idx, obj in enumerate(objects):
        for idx2, obj2 in enumerate(objects):
            if obj2['objectId'] in obj['receptacleObjectIds']:
                # obj가 obj2를 포함하고 있을 때, pass (in 관계는 반대의 case일 때 라벨링함)
                continue
            if obj2['objectId'] == obj['parentReceptacle']:
                if obj2['objectType'] in in_objects:
                    # obj가 obj2안에 들어있을 때 && obj2가 올려두는 장소일 때, obj-on-obj2
                    relations.append({'subject_id': objects_id[obj['objectId']],
                                      'object_id': objects_id[obj2['objectId']],
                                      'rel_class': rel_s2i['on']})
                else:
                    # obj가 obj2안에 들어있을 때, obj-in-obj2
                    relations.append({'subject_id': objects_id[obj['objectId']],
                                      'object_id': objects_id[obj2['objectId']],
                                      'rel_class': rel_s2i['in']})
                done_pair.append([idx, idx2])
                done_pair.append([idx2, idx])

    # over/under 관계
    for idx, obj in enumerate(objects):
        for idx2, obj2 in enumerate(objects):
            if [idx, idx2] in done_pair:
                continue
            if obj['position']['y'] - obj2['position']['y'] > UD_y_threshold \
                    and abs(obj['position']['x'] - obj2['position']['x']) < UD_xz_threshold \
                    and abs(obj['position']['z'] - obj2['position']['z']) < UD_xz_threshold:
                # obj가 obj2 위에 들어있을 때, obj-over-obj2, obj2-under-obj
                relations.append({'subject_id': objects_id[obj['objectId']],
                                  'object_id': objects_id[obj2['objectId']],
                                  'rel_class': rel_s2i['over']})
                relations.append({'subject_id': objects_id[obj2['objectId']],
                                  'object_id': objects_id[obj['objectId']],
                                  'rel_class': rel_s2i['under']})
                done_pair.append([idx, idx2])
                done_pair.append([idx2, idx])

    # in_front_of/behind 관계
    for idx, obj in enumerate(objects):
        for idx2, obj2 in enumerate(objects):
            if [idx, idx2] in done_pair:
                continue
            if 'bb' in obj and 'bb' in obj2 \
                    and sqrt(obj2['distance']**2 - (agent_height-obj2['position']['y'])**2) - sqrt(obj['distance']**2 - (agent_height-obj['position']['y'])**2) > FB_z_threshold \
                    and abs((obj['bb'][2] + obj['bb'][0]) / 2 - (obj2['bb'][2] + obj2['bb'][0]) / 2) < FB_xy_threshold \
                    and abs((obj['bb'][3] + obj['bb'][1]) / 2 - (obj2['bb'][3] + obj2['bb'][1]) / 2) < FB_xy_threshold:
                # obj가 obj2 위에 들어있을 때, obj-is_front_of-obj2, obj2-behind-obj
                relations.append({'subject_id': objects_id[obj['objectId']],
                                  'object_id': objects_id[obj2['objectId']],
                                  'rel_class': rel_s2i['in_front_of']})
                relations.append({'subject_id': objects_id[obj2['objectId']],
                                  'object_id': objects_id[obj['objectId']],
                                  'rel_class': rel_s2i['behind']})
                done_pair.append([idx, idx2])
                done_pair.append([idx2, idx])

    # INCLUDE_LR 모드가 아니면 바로 반환
    if not INCLUDE_LR:
        return relations

    # left_of/right_of 관계
    for idx, obj in enumerate(objects):
        for idx2, obj2 in enumerate(objects):
            if [idx, idx2] in done_pair:
                continue
            if 'bb' in obj and 'bb' in obj2:
                if ((agent_angle == 0 or agent_angle == 180) \
                    and abs(obj['position']['x'] - obj2['position']['x']) > LR_xz_threshold \
                    and abs(obj['position']['z'] - obj2['position']['z']) < LR_xz_threshold) \
                    or \
                    ((agent_angle == 90 or agent_angle == 270) \
                    and abs(obj['position']['x'] - obj2['position']['x']) < LR_xz_threshold \
                    and abs(obj['position']['z'] - obj2['position']['z']) > LR_xz_threshold):

                    if obj['bb'][2] + obj['bb'][0] < obj2['bb'][2] + obj2['bb'][0]:
                        relations.append({'subject_id': objects_id[obj['objectId']],
                                          'object_id': objects_id[obj2['objectId']],
                                          'rel_class': rel_s2i['left_of']})
                        relations.append({'subject_id': objects_id[obj2['objectId']],
                                          'object_id': objects_id[obj['objectId']],
                                          'rel_class': rel_s2i['right_of']})
                        done_pair.append([idx, idx2])
                        done_pair.append([idx2, idx])
    return relations

def get_global_relations(objects, objects_id):
    # object 간의 관계를 구해서 리턴
    relations = []
    if len(objects) <= 1:
        return relations
    in_objects = get_on_objects()
    UD_xz_threshold = 0.3
    UD_y_threshold = 0.3

    if objects_id is None:  # objects_id 안주어지면, objects 순서대로 id 부여 (local)
        objects_id = {obj['objectId']:i for i, obj in enumerate(objects)}

    done_pair = [] # 관계가 있음을 등록. 우선순위에 따라 하나의 관계만 가질 수 있도록 함

    # in/on 관계
    for idx, obj in enumerate(objects):
        for idx2, obj2 in enumerate(objects):
            if obj2['objectId'] in obj['receptacleObjectIds']:
                # obj가 obj2를 포함하고 있을 때, pass (in 관계는 반대의 case일 때 라벨링함)
                continue
            if obj2['objectId'] == obj['parentReceptacle']:
                if obj2['objectType'] in in_objects:
                    # obj가 obj2안에 들어있을 때 && obj2가 올려두는 장소일 때, obj-on-obj2
                    relations.append({'subject_id': objects_id[obj['objectId']],
                                      'object_id': objects_id[obj2['objectId']],
                                      'rel_class': rel_s2i['on']})
                else:
                    # obj가 obj2안에 들어있을 때, obj-in-obj2
                    relations.append({'subject_id': objects_id[obj['objectId']],
                                      'object_id': objects_id[obj2['objectId']],
                                      'rel_class': rel_s2i['in']})
                done_pair.append([idx, idx2])
                done_pair.append([idx2, idx])

    # over/under 관계
    for idx, obj in enumerate(objects):
        for idx2, obj2 in enumerate(objects):
            if [idx, idx2] in done_pair:
                continue
            if obj['position']['y'] - obj2['position']['y'] > UD_y_threshold \
                    and abs(obj['position']['x'] - obj2['position']['x']) < UD_xz_threshold \
                    and abs(obj['position']['z'] - obj2['position']['z']) < UD_xz_threshold:
                # obj가 obj2 위에 들어있을 때, obj-over-obj2, obj2-under-obj
                relations.append({'subject_id': objects_id[obj['objectId']],
                                  'object_id': objects_id[obj2['objectId']],
                                  'rel_class': rel_s2i['over']})
                relations.append({'subject_id': objects_id[obj2['objectId']],
                                  'object_id': objects_id[obj['objectId']],
                                  'rel_class': rel_s2i['under']})
                done_pair.append([idx, idx2])
                done_pair.append([idx2, idx])

    # in_front_of/behind 관계
    for idx, obj in enumerate(objects):
        for idx2, obj2 in enumerate(objects):
            if [idx, idx2] in done_pair:
                continue
            sub_x = obj['position']['x'] - obj2['position']['x']
            sub_y = obj['position']['y'] - obj2['position']['y']
            sub_z = obj['position']['z'] - obj2['position']['z']
            if sub_z < - 0.1 \
                and abs(sub_x) < 0.2 \
                and abs(sub_y) < 0.2 \
                and abs(sub_z) - abs(sub_y) > 0 \
                and abs(sub_z) - abs(sub_x) > 0:
                # obj가 obj2 앞에있고,  obj와 obj2의 좌우거리차가 가깝고, 좌우거리차보다 앞뒤거리차가 더 클때
                # obj-in_front_of-obj2, obj2-behind-obj
                relations.append({'subject_id': objects_id[obj['objectId']],
                                  'object_id': objects_id[obj2['objectId']],
                                  'rel_class': rel_s2i['in_front_of']})
                relations.append({'subject_id': objects_id[obj2['objectId']],
                                  'object_id': objects_id[obj['objectId']],
                                  'rel_class': rel_s2i['behind']})
                done_pair.append([idx, idx2])
                done_pair.append([idx2, idx])

    # INCLUDE_LR 모드가 아니면 바로 반환
    if not INCLUDE_LR:
        return relations

    # left_of/right_of 관계
    for idx, obj in enumerate(objects):
        for idx2, obj2 in enumerate(objects):
            if [idx, idx2] in done_pair:
                continue
            sub_x = obj['position']['x'] - obj2['position']['x']
            sub_y = obj['position']['y'] - obj2['position']['y']
            sub_z = obj['position']['z'] - obj2['position']['z']
            if sub_x < - 0.1 \
                and abs(sub_y) < 0.2 \
                and abs(sub_z) < 0.2 \
                and abs(sub_x) - abs(sub_y) > 0 \
                and abs(sub_x) - abs(sub_z) > 0:
                relations.append({'subject_id': objects_id[obj['objectId']],
                                  'object_id': objects_id[obj2['objectId']],
                                  'rel_class': rel_s2i['left_of']})
                relations.append({'subject_id': objects_id[obj2['objectId']],
                                  'object_id': objects_id[obj['objectId']],
                                  'rel_class': rel_s2i['right_of']})
                done_pair.append([idx, idx2])
                done_pair.append([idx2, idx])
    return relations


def get_attributes(objects, objects_id, image, instance_masks, TEST_MODE=False):
    # object들의 att들을 구해서 리턴 (현재는 color만)
    # image는 BGR 영상임
    attributes = {}
    if TEST_MODE:
        print('-')  ##
    for idx, obj in enumerate(objects):
        # color 값 계산
        att = {}
        '''
        if 'bb' in obj:
            att['color'] = get_color_of_bb(obj['bb'], image)
        else:
            att['color'] = color_s2i['unknown']
        '''
        if obj['objectId'] in instance_masks:
            att['color'], att['hsv'] = get_color_of_bb(image[instance_masks[obj['objectId']]], TEST_MODE=TEST_MODE, object_id=obj['objectId'])
        else:
            att['color'] = color_s2i['unknown']
            att['hsv'] = []
        if objects_id is not None:
            attributes[objects_id[obj['objectId']]] = att
        else:
            attributes[idx] = att
    return attributes

def get_color_of_bb(segment, valid_range=0.3, TEST_MODE = False, object_id=None):
    # image로부터 bb의 색상을 결정하여 string으로 리턴
    '''
    # [기존] bbox로부터 계산하는 방법

    # valid_range는 bb에서 가운데 몇 %까지만 사용할 지 명시 (배경 제거를 위해) # 아직 미구현
    x_len, y_len = (bb[2] - bb[0]) * valid_range, (bb[3] - bb[1]) * valid_range
    x_mid, y_mid = (bb[2] + bb[0])/2, (bb[3] + bb[1])/2
    valid_bb =  [int(x_mid - x_len/2), int(y_mid - y_len/2), int(x_mid + x_len/2), int(y_mid + y_len/2)]

    mean_RGB = image[valid_bb[1]:valid_bb[3], valid_bb[0]:valid_bb[2], :].mean(axis=0).mean(axis=0)[::-1]
    #mean_RGB = image[bb[1]:bb[3], bb[0]:bb[2], :].mean(axis=0).mean(axis=0)[::-1]
    mean_RGB = (mean_RGB.astype('uint8')/255.).tolist()
    mean_HSV = colormap.rgb2hsv(mean_RGB[0], mean_RGB[1], mean_RGB[2]) # hue를 통한 색상 검출을 위해 변환
    mean_HSV = [mean_HSV[0]*360., mean_HSV[1]*100., mean_HSV[2]*100.]
    '''

    # segment를 직접 환경으로부터 받아 계산하는 방법
    bgr = segment.mean(axis=0)
    bgr /= 255.
    mean_HSV = colormap.rgb2hsv(bgr[2], bgr[1], bgr[0])  # hue를 통한 색상 검출을 위해 변환
    mean_HSV = [mean_HSV[0] * 360., mean_HSV[1] * 100., mean_HSV[2] * 100.]
    # max_HSV : 360, 100, 100
    # https://www.google.co.kr/search?q=rgb+hex&newwindow=1&rlz=1C1ASUM_enKR734KR734&source=lnms&tbm=isch&sa=X&ved=0ahUKEwiKw4q7q7fcAhWDnJQKHTguD4EQ_AUICigB&biw=1920&bih=947#imgrc=9QZuRh-ldFvxvM: # hex 표
    # https://ko.wikipedia.org/wiki/HSV_%EC%83%89_%EA%B3%B5%EA%B0%84 # hsv 정의
    # https://search.naver.com/search.naver?where=nexearch&sm=tab_jum&query=%EC%83%89%EC%83%81%ED%91%9C # 네이버 색상표
    # http://www.workwithcolor.com/magenta-pink-color-hue-range-01.htm # 색상 범위표

    if mean_HSV[2] <= 20:
        if TEST_MODE:
            print_color(object_id, 'black', mean_HSV)
        return color_s2i['black'], mean_HSV
    elif mean_HSV[2] >= 95 and mean_HSV[1] <= 8 or mean_HSV[2] >= 90 and mean_HSV[1] <= 3:
        if TEST_MODE:
            print_color(object_id, 'white', mean_HSV)
        return color_s2i['white'], mean_HSV
    elif mean_HSV[2] <= 80 and mean_HSV[1] <= 10 or mean_HSV[1] <= 5 or mean_HSV[2] <= 40 and mean_HSV[1] <= 20:
        if TEST_MODE:
            print_color(object_id, 'gray', mean_HSV)
        return color_s2i['gray'], mean_HSV

    if mean_HSV[0] >= 300 or mean_HSV[0] <= 10:
        if TEST_MODE:
            print_color(object_id, 'red', mean_HSV)
        return color_s2i['red'], mean_HSV
    #elif mean_HSV[0] >= 300 and mean_HSV[0] <= 340:
    #    return 'pink' + '__(' +  str(round(mean_HSV[0], 1))+', '+str(round(mean_HSV[1], 1))+', '+str(round(mean_HSV[2], 1)) + ')'
    elif mean_HSV[0] >= 260:
        if TEST_MODE:
            print_color(object_id, 'purple', mean_HSV)
        return color_s2i['purple'], mean_HSV
    elif mean_HSV[0] >= 171:
        if TEST_MODE:
            print_color(object_id, 'blue', mean_HSV)
        return color_s2i['blue'], mean_HSV
    elif mean_HSV[0] >= 70 and mean_HSV[0] <= 170:
        if TEST_MODE:
            print_color(object_id, 'green', mean_HSV)
        return color_s2i['green'], mean_HSV
    elif mean_HSV[0] >= 40 and mean_HSV[0] <= 70:
        if mean_HSV[0] >= 60 and mean_HSV[2] <= 30:
            if TEST_MODE:
                print_color(object_id, 'green', mean_HSV)
            return color_s2i['green'], mean_HSV
        elif mean_HSV[0] < 60 and mean_HSV[2] <= 30:
            if TEST_MODE:
                print_color(object_id, 'brown', mean_HSV)
            return color_s2i['brown'], mean_HSV
        if TEST_MODE:
            print_color(object_id, 'yellow', mean_HSV)
        return color_s2i['yellow'], mean_HSV
    elif mean_HSV[0] >= 10 and mean_HSV[0] <= 40:
        if TEST_MODE:
            print_color(object_id, 'brown', mean_HSV)
        return color_s2i['brown'], mean_HSV
    else:
        if TEST_MODE:
            print_color(object_id, 'unknown', mean_HSV)
        return color_s2i['unknown'], mean_HSV # + '__(' +  str(round(mean_HSV[0], 1))+', '+str(round(mean_HSV[1], 1))+', '+str(round(mean_HSV[2], 1)) + ')'

def print_color(object_id, color_str, mean_HSV):
    print('[' + object_id + '] ' + color_str+'[hsv]: (' + str(round(mean_HSV[0], 1)) + ', ' + str(
        round(mean_HSV[1], 1)) + ', ' + str(round(mean_HSV[2], 1)) + ')')

