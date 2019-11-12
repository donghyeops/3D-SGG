# -*- coding:utf-8 -*-
import json
import os.path as osp

class DatasetLoader:
    def __init__(self, qas_path, owl_path):
        self.owl_path = owl_path
        self.qas_path = qas_path
        self.owls = dict() # {scene_name: owl_contents}
        self.qas = self.load_qa_scenario(qas_path)
        # qas: {'FloorPlan1_S0.json": {
        #                              "10": {
        #                                   'existence': ["Exsistence/Bread/10", false],
        #                                   'counting':[], 'attribute':[], 'relation':[],'agenthave':[], 'include':[]
        #                                   }}}
        self.scene_names = ['FloorPlan'+str(sn) for sn in range(1, 31)]

    def __len__(self):
        pass

    def load_qa_scenario(self, path):
        with open(path, 'r') as f:
            qas = json.load(f)
        return qas

    def generator(self):
        for fname, qa_steps in self.qas.items():
            room_name = fname.split('_')[0] # "FloorPan30_S0.json"
            seed_name = fname.split('_')[1].split('.')[0]
            room_dir = osp.join(self.owl_path, room_name, seed_name)
            owl_path = osp.join(room_dir, fname.split('.')[0]+'_T') #  "FloorPan30_S0"

            for step, qa_set in qa_steps.items():
                try:
                    with open(owl_path+step+'.owl') as f: # 그냥 step 안쓰고 하나 빼준 이유는 gt에 비해 pred가 항상 하나씩 적은데, 알고리즘적으로 하나 스텝이 딸리는거가틈
                        owl = f.read()
                except:
                    print(f'no file !! [{owl_path+step+".owl"}') # (str(int(step)-1))
                    continue

                yield (qa_set, owl, room_dir)

if __name__ == '__main__':
    '''
    p = os.listdir('./results/owl')
    with open(os.path.join('./results/owl', p[0])) as f:
        a = f.read()

    qa = h5py.File('./existence.h5')
    print(list(qa['questions']['question']))
    '''
    dataset = DatasetLoader(qas_path='/home/ailab/DH/ai2thor/datasets/qa_scenario.json',
                            owl_path='/home/ailab/DH/ai2thor/datasets/gsg_pred/owl')
    for i, (qa_set, owl, room_name) in enumerate(dataset.generator()):
        print(qa_set)
        print(room_name)
        print(owl)
        break

