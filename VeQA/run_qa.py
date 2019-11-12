from connection_manager import SocketConnection
from dataset_loader import DatasetLoader
import numpy as np
from keras.models import load_model
from semantic_parser import SPMgr

import time
import sys
import os
_op = os.getcwd()
os.chdir('/home/ailab/DH/ai2thor')
sys.path.append('/home/ailab/DH/ai2thor')
from thor_utils import owl_util as ou
os.chdir(_op)


HOST = '203.249.22.18'
PORT = 7770

qtypes = ['existence', 'counting', 'attribute', 'relation', 'agenthave', 'include']

def run_qa_system():
    target_path='/home/ailab/DH/ai2thor/datasets/190514 gsg_pred_OATR_only_recog/owl'
    print('target_path:', target_path)

    dataset = DatasetLoader(qas_path='/home/ailab/DH/ai2thor/datasets/190514 gsg_gt/qa_scenario.json',
                            owl_path=target_path)

    spMgr = SPMgr(db_path='/home/ailab/DH/ai2thor/datasets/190514 gsg_gt/qa_scenario.json',
                  word_book_path='./word_book.json')
    spMgr.load_model(model_path='./test_semantic_parser.h5')

    sc = SocketConnection()
    sc.make_connection(HOST, PORT)

    counts = {qtype: 0 for qtype in qtypes}  # 질문 타입별로 정확도 구함
    corrects = {qtype: 0 for qtype in qtypes}

    # print(scene_names)
    for i, (qa_set, owl, room_dir) in enumerate(dataset.generator()):
        sc.send_message(owl, end_mark=True)  # owl 파일 전송
        recv_msg = sc.receive_msg()  # 수신 확인 및 준비 완료 메시지 받기
        if recv_msg == "-1":
            for qtype, qa in qa_set.items():
                query, result = spMgr.predict(qa['natural_question'])
                query += '/' + qa['question'].split('/')[-1] # time token 붙이기
                question_msg = query + '\n' # end 토큰 (필수임)
                question_msg = qa['question'] + '\n' # GT

                sc.send_message(question_msg, end_mark=False)  # 질의 전송
                answer = sc.receive_msg()  # 추론 결과 받기

                pred_qtype = query.split('/')[0] # ori
                #pred_qtype = question_msg.split('/')[0]
                if pred_qtype == 'Attribute':
                    if query.split('/')[1] == 'Color': # ori
                        #if question_msg.split('/')[1] == 'Color':  # ori
                        answer = answer.replace('Color', '').lower()
                    else:
                        answer = ou.os_o2a.get(answer, answer)
                elif pred_qtype == 'Relation':
                    answer = ou.rel_o2a.get(answer, answer)
                elif pred_qtype == 'Include':
                    answer = answer.split('/')
                    remv = []
                    for _i, t in enumerate(answer):
                        if t == '':
                            remv.append(_i)
                        elif t not in ou.INABLE_OBJECTS:
                            remv.append(_i)
                    for _i in remv[::-1]:
                        answer.pop(_i)

                #if qa[1] != answer: # 오답이면

                counts[qtype] += 1
                correctness = False
                if qtype == 'include': # 리스트끼리 비교해야함.
                    if set(qa['answer']) == set(answer):
                        corrects[qtype] += 1
                        correctness = True
                else:
                    if str(qa['answer']) == answer:
                        corrects[qtype] += 1
                        correctness = True

                #print(room_dir)
                #print('gt qustion: {} (answer: {})'.format(qa['question'], qa['answer']))
                #print('send msg: {}'.format(query))

                #result = [qa['predicate'], qa['subject'], qa['object']] #gt
                result = list(result)
                if result[0] == 'null':
                    result[0] = '?'
                if result[1] == 'null':
                    result[1] = '?'
                if result[2] == 'null':
                    result[2] = '?'

                print(f'[{i}] '+'input question:', qa['natural_question'])
                print('parsing results: [S: {}, P: {}, O: {}]'.format(result[1], result[0], result[2]))
                print('send msg: {}({}, {})'.format(result[0], result[1], result[2]))
                print('recv msg: {} ({})'.format(answer, correctness))
                print('')
            sc.send_message("-1", end_mark=False)  # 질의 끝 메시지 전송
        else:
            raise Exception("서버에 이상 생김")

    accuracy = {}
    for qtype in qtypes:
        accuracy[qtype] = corrects[qtype] / float(counts[qtype])

    print('Acc results')
    for k, v in accuracy.items():
        print('\t{}: {} ({}/{})'.format(k, str(v)[:6], corrects[k], counts[k]))
    print('')
    print('\ttotal: {} ({}/{})'.format(str(np.sum(list(corrects.values())) / np.sum(list(counts.values())))[:6],
                                       np.sum(list(corrects.values())),
                                       np.sum(list(counts.values()))))


if __name__ == '__main__':
    st = time.time()
    run_qa_system()
    sp = time.time() - st
    print(f'\ndone [{int(sp // 3600)}h {int(sp % 3600 // 60)}m {int(sp % 60)}s]\n')
