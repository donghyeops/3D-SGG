# -*- coding:utf-8 -*-
import os
import os.path as osp
import json
import random
from collections import Counter
from pprint import pprint as pp

from thor_utils import annotation_util as au


class QuestionGenerator:
    def __init__(self, gsg_dir, output_dir):
        self.gsg_dir = gsg_dir
        self.question_dir = output_dir
        self.question_types = ['existence', 'counting', 'attribute', 'relation', 'include', 'agenthave']
        self.target_labels = [idx for idx, v in au.obj_i2s.items() if v != 'background']
        pp(au.obj_i2s.items())

    def get_article(self, object_class):
        object_class = object_class.lower()
        if object_class in ['bread']:
            return f'a loaf of {object_class}'
        elif object_class in ['lettuce']:
            return f'a head of {object_class}'
        elif object_class.startswith(('a', 'e', 'i', 'o', 'u')):
            return f'an {object_class}'
        else:
            return f'a {object_class}'
    def get_multiple_form(self, object_class):
        object_class = object_class.lower()
        if object_class in ['bread']:
            return f'loaves of {object_class}'
        elif object_class in ['lettuce']:
            return f'heads of {object_class}'
        elif object_class.endswith('y'):
            return f'{object_class}ies'
        elif object_class.endswith(('sh', 'ch', 's', 'o', 'x')):
            return f'{object_class}es'
        else:
            return f'{object_class}s'

    def wrap_templete(self, qtype, sub_qtype=None, subject=None, object=None):
        if qtype == 'existence':
            assert subject is not None, 'subject is None'
            vote = random.randrange(0, 4)
            article = self.get_article(subject)
            if vote == 0:
                return f'Is there {article} somewhere in the room?'
            elif vote == 1:
                return f'Please tell me if there is {article} somewhere in the room.'
            else:
                return f'I think {article} is in the room. Is that correct?'

        elif qtype == 'counting':
            assert subject is not None, 'subject is None'
            vote = random.randrange(0, 3)
            article = self.get_multiple_form(subject)
            if vote == 0:
                return f'Please tell me how many {article} are around here?'
            elif vote == 1:
                return f'Count the number of {article} in this room.'
            else:
                return f'How many {article} are there in the room?'

        elif qtype == 'attribute':
            assert subject is not None, 'subject is None'
            assert sub_qtype is not None, 'sub_qtype is None'
            if sub_qtype == 'open_state' or sub_qtype == 'open state':
                sub_qtype = 'openness'
            vote = random.randrange(0, 3)
            if vote == 0:
                return f'What is the {subject.lower()}\'s {sub_qtype}?'
            elif vote == 1:
                return f'Please tell me what is {sub_qtype} of the {subject.lower()} in this room.'
            else:
                return f'What is the {sub_qtype} of {subject.lower()}?'

        elif qtype == 'relation':
            assert subject is not None, 'subject is None'
            assert object is not None, 'object is None'
            vote = random.randrange(0, 3)
            if vote == 0:
                return f'What is relationship between {subject.lower()} and {object.lower()}?'
            elif vote == 1:
                return f'Please tell me what is relationship between {subject.lower()} and {object.lower()}.'
            else:
                return f'What is the relationship of {subject.lower()} about {object.lower()}?'

        elif qtype == 'include':
            assert subject is not None, 'subject is None'
            vote = random.randrange(0, 2)
            if vote == 0:
                return f'What things are in the {subject.lower()}?'
            else:
                return f'Please tell me what things are in the {subject.lower()}.'

        elif qtype == 'agenthave':
            vote = random.randrange(0, 2)
            if vote == 0:
                return f'What is the thing agent have?'
            else:
                return f'Please tell me what is the thing agent have.'

        else:
            raise Exception('wrong qtype in wrap_templete() :', qtype)



    def get_action_files(self):
        files = os.listdir(self.gsg_dir)
        return [file for file in files if file.split('.')[-1] == 'json']

    def make_questions(self, qstep=10):
        print('make QA scenario ...')
        counter = {}
        qa_scenarioes = {}
        for room_name in sorted(os.listdir(self.gsg_dir)):  # 방 번호
            print(f'\t{room_name}...')
            files = os.listdir(osp.join(self.gsg_dir, room_name))
            for file in sorted(files):  # seed 별 파일들
                with open(osp.join(self.gsg_dir, room_name, file), 'r') as f:
                    data = json.load(f)
                self.__preprocessing_gsg_history(data['gsg_history'])
                qas = {}
                for time, gsg in data['gsg_history'].items():
                    if int(time) % qstep == 0:  # 스탭 별로 question set 생성
                        qa = self.make_question(gsg, int(time))  # question set 생성 (최대 len(self.question_types)개)
                        qas[int(time)] = qa
                        for qt in qa.keys():
                            try:
                                counter[qt] += 1
                            except:
                                counter[qt] = 1
                qa_scenarioes[file] = qas
        os.makedirs(self.question_dir, exist_ok=True)
        with open(osp.join(self.question_dir, 'qa_scenario.json'), 'w') as f:
            json.dump(qa_scenarioes, f, indent='\t')
        print('save QA scenario [PATH: {}]'.format(osp.join(self.question_dir, 'qa_scenario.json')))
        print('=== counting ===')
        print(f'total: {sum(counter.values())}')
        pp(counter)

    def __preprocessing_gsg_history(self, gsg_history):
        opened_obj_ids = []  # 이전에 열렸던적이 있는지. (include Q를 할지 말지 결정)
        for time, gsg in gsg_history.items():
            for obj_id, obj in gsg['objects'].items():
                if obj['open_state'] == au.openState_s2i['open']:
                    opened_obj_ids.append(obj_id)
                if obj['open_state'] != au.openState_s2i['unable']:
                    if obj_id in opened_obj_ids:
                        gsg_history[time]['objects'][obj_id]['ckecked'] = True
                    else:
                        gsg_history[time]['objects'][obj_id]['ckecked'] = False

    def make_question(self, gsg, time, token='/'):
        if not gsg['objects']:
            return {}
        qa = {}  # {'qt':[q, a]}
        for qt in self.question_types:
            if qt == 'existence':
                # Q: Exsitence/물체/시간
                target_label = random.choice(self.target_labels)
                question = token.join([qt.capitalize(), au.obj_i2s[target_label], str(time)])
                answer = False
                natural_question = self.wrap_templete(qt, subject=au.obj_i2s[target_label])
                predicate = 'objectExistence'
                subject = au.obj_i2s[target_label]
                object = None

                for obj in gsg['objects'].values():
                    if obj['label'] == target_label:
                        answer = True
                        break

            elif qt == 'counting':
                # Q: Counting/물체/시간
                target_label = random.choice(self.target_labels)
                question = token.join([qt.capitalize(), au.obj_i2s[target_label], str(time)])
                natural_question = self.wrap_templete(qt, subject=au.obj_i2s[target_label])
                predicate = 'numberOfObject'
                subject = au.obj_i2s[target_label]
                object = None

                answer = 0
                for obj in gsg['objects'].values():
                    if obj['label'] == target_label:
                        answer += 1

            elif qt == 'attribute':
                # Q: Attribute/속성/물체/시간
                # 개수가 하나인 물체 중에서 선택
                target_obj, target_label = self.__select_unique_object(gsg, k=1)
                if target_obj is None: # 하나인 물체가 없으면 패스
                    continue
                if target_obj['open_state'] != 0: # 여닫을 수 있는 물체만.
                    sub_qt = random.choice(['color'] + ['open_state']*3)
                else:
                    sub_qt = 'color'
                if sub_qt == 'color':
                    predicate = 'mainColorObject'
                    answer = au.color_i2s[target_obj[sub_qt]]
                else:
                    predicate = 'openStateOfObject'
                    answer = au.openState_i2s[target_obj[sub_qt]]
                natural_question = self.wrap_templete(qt, sub_qtype=sub_qt, subject=au.obj_i2s[target_label])
                subject = au.obj_i2s[target_label]
                question = token.join([qt.capitalize(), sub_qt.replace('_', '').capitalize(),
                                       au.obj_i2s[target_label], str(time)])
            elif qt == 'relation':
                # Q: Relation/obj물체/sub물체/시간
                unique_objs = self.__get_unique_objects(gsg)
                if not unique_objs:
                    continue
                unique_objs_ids = [obj['id'] for obj in unique_objs]
                ballot_box = []
                for key, rel in gsg['relations'].items():
                    if rel['subject_id'] in unique_objs_ids and rel['object_id'] in unique_objs_ids:
                        ballot_box.append(key)
                if not ballot_box:
                    continue
                target_key = random.choice(ballot_box)
                target_rel = gsg['relations'][target_key]
                answer = au.rel_i2s[target_rel['rel_class']]
                question = token.join([qt.capitalize(), au.obj_i2s[gsg['objects'][str(target_rel['subject_id'])]['label']],
                                       au.obj_i2s[gsg['objects'][str(target_rel['object_id'])]['label']], str(time)])

                natural_question = self.wrap_templete(qt,
                                                      subject=au.obj_i2s[gsg['objects'][str(target_rel['subject_id'])]['label']],
                                                      object=au.obj_i2s[gsg['objects'][str(target_rel['object_id'])]['label']])
                predicate = 'objectSpatialRelation'
                subject = au.obj_i2s[gsg['objects'][str(target_rel['subject_id'])]['label']]
                object = au.obj_i2s[gsg['objects'][str(target_rel['object_id'])]['label']]
            elif qt == 'include':
                # Q: Include/sub물체/시간
                target_ids = []  # 이전에 열렸던 물체들 (냉장고, 전자레인지 등이 들어갈 수 있음)
                for obj in gsg['objects'].values():
                    if obj['open_state'] != au.openState_s2i['unable'] and obj['ckecked']:
                        target_ids.append(obj['id'])
                if not target_ids:
                    continue
                target_id = random.choice(target_ids)
                target_obj = gsg['objects'][str(target_id)]
                question = token.join([qt.capitalize(), au.obj_i2s[target_obj['label']], str(time)])

                natural_question = self.wrap_templete(qt, subject=au.obj_i2s[target_obj['label']])
                predicate = 'objectsInsideOf'
                subject = au.obj_i2s[target_obj['label']]
                object = None

                answer = []  # 정답은 여러 물체가 될 수 있음
                for key, rel in gsg['relations'].items(): # target object와 in 관계인 물체들을 정답으로 선정
                    if rel['object_id'] == target_id and au.rel_i2s[rel['rel_class']] == 'in':
                        answer.append(au.obj_i2s[gsg['objects'][str(rel['subject_id'])]['label']])

            elif qt == 'agenthave':
                # Q: AgentHave/시간
                if 'object_id' not in gsg['agent'] or gsg['agent']['object_id'] is None:
                    continue
                question = token.join(['AgentHave', str(time)])
                # answer = au.obj_i2s[gsg['agent']['object']['label']]
                answer = au.obj_i2s[gsg['objects'][str(gsg['agent']['object_id'])]['label']]

                natural_question = self.wrap_templete(qt)
                predicate = 'agentHaveObject'
                subject = None
                object = None

            else:
                raise Exception('wrong question type !! [{}]'.format(qt))

            qa[qt] = {'question':question,
                      'answer':answer,
                      'natural_question':natural_question,
                      'predicate':predicate,
                      'subject':subject,
                      'object':object}
        return qa

    def __select_unique_object(self, gsg, k=1):
        # 개수가 하나인 물체 중에서 선택
        obj_map = Counter(obj['label'] for obj in gsg['objects'].values())
        target_labels = [k for k, v in obj_map.items() if v == 1]
        if not target_labels:
            return None, None
        if len(target_labels) < k:
            return None, None
        target_label = random.sample(target_labels, k=k)
        target_obj = []
        target_label_ = []
        for obj in gsg['objects'].values():
            if obj['label'] in target_label:
                target_label_.append(obj['label'])
                target_obj.append(obj)
                if k==1:
                    break
        assert target_obj, 'fail to find object'
        if k==1:
            return target_obj[0], target_label_[0]
        return target_obj, target_label_

    def __get_unique_objects(self, gsg):
        # 개수가 하나인 물체 리스트 반환
        obj_map = Counter(obj['label'] for obj in gsg['objects'].values())
        unique_labels = [k for k, v in obj_map.items() if v == 1]
        unique_objs = []
        for obj in gsg['objects'].values():
            if obj['label'] in unique_labels:
                unique_objs.append(obj)
        return unique_objs

if __name__ == '__main__':
    qg = QuestionGenerator(gsg_dir='./datasets/190514 gsg_gt/gsg', output_dir='./datasets')
    qg.make_questions(qstep=10)
