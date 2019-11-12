import json
import glob
from pprint import pprint

p = glob.glob('./ah_withFail/*')

count=0
di = {}
for pp in p:
    with open(pp,'r') as f:
        d = json.load(f)
    for a in d['actions']:
        try:
            di[a["action"]] += 1
        except:
            di[a["action"]] = 1
    count += len(d['actions'])
print('action count:', count)
pprint(di)

with open('qa_scenario.json', 'r') as f:
    qas = json.load(f)

qtype = ['existence', "counting", "attribute", "relation", "include", "agenthave"]
qa_dict = {qt:{} for qt in qtype}
qcount = {qt:0 for qt in qtype}

for k, v in qas.items():
    for k2, v2 in v.items():
        for k3, v3 in v2.items():
            qcount[k3] += 1
            if k3 == 'include':
                continue
            try:
                qa_dict[k3][v3['answer']] += 1
            except:
                qa_dict[k3][v3['answer']] = 1

for q in qtype:
    print(f'qtype : [{q}] # {qcount[q]}')
    if q == 'include':
        continue   

    target = qa_dict[q]
    total = sum(target.values())
    for k, v in target.items():
        print(f'\t{k}: {v/total}')
    print('')

print(f'total Q count : {sum(qcount.values())}')
