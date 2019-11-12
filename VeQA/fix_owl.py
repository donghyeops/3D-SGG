import json
import glob

import os
import sys
_op = os.getcwd()
os.chdir('/home/ailab/DH/ai2thor')
sys.path.append('/home/ailab/DH/ai2thor')
from thor_utils import owl_util as ou
os.chdir(_op)

# 이건 agent have가 owl에 안박혀서, owl을 다시 만들기 위함임

target_dir = '/home/ailab/DH/ai2thor/datasets/(qa) gsg_pred_R'
owl_path = target_dir+'/fixed_owl'
os.makedirs(owl_path, exist_ok=True)
for room in sorted(os.listdir(target_dir+'/gsg')):  # 방 번호
    for gsg_name in sorted(os.listdir(target_dir+'/gsg/'+room)):
        file_dir = target_dir+'/gsg/'+room+'/'+gsg_name
        seed = gsg_name.split('.')[0][-2:]
        with open(file_dir, 'r') as f:
            file = json.load(f)
        for k, gsg in file['gsg_history'].items():
            ou.write_owl_from_gsg({k:gsg},
                                  file_name=f'{room}_{seed}_T{k}.owl',
                                  dir_path=owl_path+'/'+room+'/'+seed)  # only_now 안쓰면 한 파일에 모두 씀