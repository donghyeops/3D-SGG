import os
from thor_utils import annotation_util as au

rel_owl_class_naming = {'left_of': 'toTheLeftOf',
                        'right_of': 'toTheRightOf',
                        'in_front_of': 'inFrontOf-Generally',
                        'behind': 'behind-Generally',
                        'over': 'aboveOf',
                        'under': 'belowOf',
                        'on': 'on-Physical',
                        'in': 'insideOf',
                        'has': 'has',
                        'beowned': 'beOwned'
                        }
rel_o2a = {v:k for k, v in rel_owl_class_naming.items()}
os_owl_class_naming = {'open': 'ObjectStateOpen',
                       'close': 'ObjectStateClosed',
                       'unable': None
                       }
os_o2a = {v:k for k, v in os_owl_class_naming.items()}
INABLE_OBJECTS = [
    'Potato',
    'Apple',
    'Lettuce',
    'Egg',
    'Tomato',
    'Bread',
    'Mug',
    'Plate',
    'Bowl',
    'Spoon',
    'ButterKnife',
    'Knife',
    'Container'
]
OBJECTS = [
    'Background',
    'Potato',
    'Apple',
    'Lettuce',
    'Egg',
    'Tomato',
    'Bread',
    'Mug',
    'Toaster',
    'CoffeeMachine',
    'Microwave',
    'GarbageCan',
    'Chair',
    'HousePlant',
    'Plate',
    'Bowl',
    'Spoon',
    'ButterKnife',
    'Knife',
    'Container',
    'Pot',
    'Pan',
    'Sink',
    'Fridge'
    ]

def owl_class_naming(objects):
    for idx, obj in enumerate(objects):
        obj['owlId'] = 'object' + str(idx)
        obj['objectType'] = obj['objectType'][0].upper() + obj['objectType'][1:]
        obj['color'] = obj['color'][0].upper() + obj['color'][1:] + 'Color'
        if obj['open_state'].lower() != 'unable':
            obj['open_state'] = os_owl_class_naming[obj['open_state'].lower()]

    for idx, obj in enumerate(objects):  # owlId가 할당 안된 경유를 방지하기 위해 따로 뻄
        for i, rel in enumerate(obj['relations']):
            obj['relations'][i]['target_owlId'] = objects[rel['target']]['owlId']
            obj['relations'][i]['relation'] = rel_owl_class_naming[rel['relation'].lower()]
    return objects

def _object_ThorToOwl(color):
    return color[0].upper() + color[1:] + 'Color'

def get_init_text():
    return '''<?xml version="1.0"?>
<!DOCTYPE rdf:RDF [
    <!ENTITY owl "http://www.w3.org/2002/07/owl#" >
    <!ENTITY owl2 "http://www.w3.org/2006/12/owl2#" >
    <!ENTITY xsd "http://www.w3.org/2001/XMLSchema#" >
    <!ENTITY owl2xml "http://www.w3.org/2006/12/owl2-xml#" >
    <!ENTITY knowrob "http://knowrob.org/kb/knowrob.owl#" >
    <!ENTITY rdfs "http://www.w3.org/2000/01/rdf-schema#" >
    <!ENTITY rdf "http://www.w3.org/1999/02/22-rdf-syntax-ns#" >
    <!ENTITY arbi "http://www.arbi.com/ontologies/arbi.owl#" >
    <!ENTITY protege "http://protege.stanford.edu/plugins/owl/protege#" >
]>

<rdf:RDF xmlns="http://knowrob.org/kb/ias_semantic_map.owl#"
     xml:base="http://knowrob.org/kb/ias_semantic_map.owl#"
     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
     xmlns:arbi="http://www.arbi.com/ontologies/arbi_knowrob.owl#"
     xmlns:owl="http://www.w3.org/2002/07/owl#"
     xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
     xmlns:srdl2-comp="http://knowrob.org/kb/srdl2-comp.owl#"
     xmlns:knowrob="http://knowrob.org/kb/knowrob.owl#">
    <owl:Ontology rdf:about="http://knowrob.org/kb/ias_semantic_map.owl#">
        <owl:imports rdf:resource="package://knowrob_common/owl/knowrob.owl"/>
    </owl:Ontology>
'''

def get_end_text():
    return '</rdf:RDF>'

def write_owl_from_gsg(gsg_history, file_name='sg.owl', dir_path='./results'):
    # 동적 모드에서 사용
    os.makedirs(dir_path, exist_ok=True)
    objects = []
    for time, gsg in gsg_history.items():
        objects += parse_gsg(gsg, time)
    contents = ''
    contents += get_init_text()
    for idx, obj in enumerate(objects):
        contents += get_individual_obj(obj) + '\n'
    contents += get_end_text()
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    with open(os.path.join(dir_path, file_name), 'wt') as f:
        f.write(contents)
    print('save owl file ({})'.format(os.path.join(dir_path, file_name)))

def parse_gsg(gsg, time):
    obj_dict = {}
    for key, obj in gsg['objects'].items():
        new_obj = {
            'owlId': 'object' + str(key),
            'objectType': au.obj_i2s[obj['label']],
            'color': au.color_i2s[obj['color']],
            'open_state': au.openState_i2s[obj['open_state']],
            'detection': obj['detection'],
            'relations': [],
            'time': time
        }
        obj_dict[int(key)] = new_obj
    for key, rel in gsg['relations'].items():
        obj_dict[rel['subject_id']]['relations'].append({
            'target_owlId': obj_dict[rel['object_id']]['owlId'],
            'relation': au.rel_i2s[rel['rel_class']]
        })

    # agent = {
    #     'owlId': 'agent',
    #     'relations': [],
    #     'time': time
    # }
    if 'object_id' in gsg['agent'] and gsg['agent']['object_id'] is not None:
        obj_dict[gsg['agent']['object_id']]['relations'].append({
            'target_owlId': 'agent',
            'relation': 'beOwned'
        })
    # if 'object' in gsg['agent'] and gsg['agent']['object'] is not None:
    #     # agent가 현재 소유한 물체 정보
    #     obj = gsg['agent']['object']
    #     new_obj = {
    #         'owlId': 'object' + str(obj['id']),
    #         'objectType': au.obj_i2s[obj['label']],
    #         'color': au.color_i2s[obj['color']],
    #         'open_state': au.openState_i2s[obj['open_state']],
    #         'detection': obj['detection'],
    #         'relations': [{
    #             'target_owlId': 'agent',
    #             'relation': 'beOwned'
    #         }],
    #         'time': time
    #     }
    #     obj_dict[obj['id']] = new_obj

        # # 에이전트 has 관계 추가
        # agent['relations'].append({
        #     'target_owlId': new_obj['owlId'],
        #     'relation': 'has'
        # })
    # obj_dict['agent'] = agent # obj_dict에 agent 포함 (같이 처리)
    obj_list = list(obj_dict.values())
    for obj in obj_list:
        if 'objectType' in obj:
            obj['objectType'] = obj['objectType'][0].upper() + obj['objectType'][1:]
        if 'color' in obj:
            obj['color'] = obj['color'][0].upper() + obj['color'][1:] + 'Color'
        if 'open_state' in obj:
            if obj['open_state'].lower() != 'unable':
                obj['open_state'] = os_owl_class_naming[obj['open_state'].lower()]
            else:
                obj.pop('open_state') # unable이면 아예 표기 안함
        if 'relations' in obj:
            for i, rel in enumerate(obj['relations']):
                obj['relations'][i]['relation'] = rel_owl_class_naming[rel['relation'].lower()]

    return obj_list

def write_owl(objects, file_name='sg.owl', dir_path='./results'):
    # 기존 버전. (save 버튼 눌러서 작동됨)
    # 'local_id', 'objectType', 'relationship', 'target', 'color', 'open_state'
    objects = owl_class_naming(objects)
    contents = ''
    contents += get_init_text()
    for idx, obj in enumerate(objects):
        contents += get_individual_obj(obj) + '\n'
    contents += get_end_text()
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    with open(os.path.join(dir_path, file_name), 'wt') as f:
        f.write(contents)
    print('save owl file ({})'.format(os.path.join(dir_path, file_name)))

def get_individual_obj(obj):
    content = ''
    content += ' '*4 + '<owl:NamedIndividual rdf:about=\"&arbi;{}\">\n'.format(obj['owlId'])
    if 'objectType' in obj:
        content += ' '*8 + '<rdf:type rdf:resource=\"&knowrob;{}\"/>\n'.format(obj['objectType'])
    if 'time' in obj:
        content += ' '*8 + '<knowrob:startTime rdf:resource=\"http://www.arbi.com/ontologies/arbi_map.owl#timepoint_{}\"/>\n'.format(obj['time'])
    if 'color' in obj:
        content += ' '*8 + '<knowrob:mainColorOfObject rdf:resource=\"&knowrob;{}\"/>\n'.format(obj['color'])
    if 'open_state' in obj:
        content += ' '*8 + '<knowrob:stateOfObject rdf:resource=\"&knowrob;{}\"/>\n'.format(obj['open_state'])
    for rel in obj['relations']:
        content += ' '*8 + '<knowrob:{} rdf:resource=\"&arbi;{}\"/>\n'.format(rel['relation'], rel['target_owlId'])
    content += ' '*4 + '</owl:NamedIndividual>\n'
    return content

if __name__ == '__main__':
    write_owl(([{'objectType': 'tomato',
                 'color': 'red',
                 'open_state': 'unable'},
                {'objectType': 'fridge',
                 'color': 'gray',
                 'open_state': 'close'}
                ]))

