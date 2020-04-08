# -*- coding:utf-8 -*-

from __future__ import print_function
import time
import argparse

from runner import CT



# 토마토 집어서 냉장고에 넣기
def plan1(ct, interval=0.5):
    ct.set_scene('FloorPlan1')
    print('plan1 start')
    time.sleep(interval*3)
    
    # do plan
    move1 = [ct.left] + [ct.go]*4 + [ct.right] + [ct.go]*8 + [ct.right] + [ct.down]
    for act in move1: # 토마토까지 이동하기
        act()
        time.sleep(interval)
    time.sleep(interval) # 추가 지연
    ct.pickup(1) # 토마토잡기
    time.sleep(interval*2)
    move2 = [ct.up] + [ct.left] + [ct.go]*4 + [ct.left]
    for act in move2: # 냉장고까지 이동하기
        act()
        time.sleep(interval)
    time.sleep(interval) # 추가 지연
    ct.open(0) # 냉장고 열기
    time.sleep(interval*2)
    ct.put() # 넣기
    time.sleep(interval*2)
    ct.close(2) # 냉장고 닫기
    
    time.sleep(interval)
    print('plan1 done')

# 토마토 집어서 냉장고에 넣고, 냉장고 안의 계란 꺼내서 전자레인지에 넣기
def plan2(ct, interval=0.5):
    ct.set_scene('FloorPlan1')
    print('plan2 start')
    time.sleep(interval*3)
    
    # do plan
    move1 = [ct.left] + [ct.go]*4 + [ct.right] + [ct.go]*8 + [ct.right] + [ct.down]
    for act in move1: # 토마토까지 이동하기
        act()
        time.sleep(interval)
    time.sleep(interval) # 추가 지연
    
    ct.pickup(1) # 토마토잡기
    time.sleep(interval*2)
    
    move2 = [ct.up] + [ct.left] + [ct.go]*4 + [ct.left]
    for act in move2: # 냉장고까지 이동하기
        act()
        time.sleep(interval)
    time.sleep(interval) # 추가 지연
    
    ct.open(0) # 냉장고 열기
    time.sleep(interval*2)
    ct.put() # 넣기
    time.sleep(interval*2)
    ct.pickup(0) # 계란 잡기
    time.sleep(interval*2)
    ct.close(1) # 냉장고 닫기
    time.sleep(interval*2)
    
    move3 = [ct.left] + [ct.go]*11 + [ct.left] + [ct.go]*3 + [ct.right]
    for act in move3: # 냉장고까지 이동하기
        act()
        time.sleep(interval)
    time.sleep(interval) # 추가 지연
    
    ct.open(0) # 전자레인지 열기
    time.sleep(interval*2)
    ct.put(0) # 계란 넣기
    time.sleep(interval*2)
    ct.close(0) # 전자레인지 닫기
    time.sleep(interval*2)
    
    ct.left()
    time.sleep(interval)
    ct.left()
    time.sleep(interval)
    
    time.sleep(interval*2)
    print('plan2 done')
    
if __name__ == '__main__':
    #from Face_Localization import MTCNN
    parser = argparse.ArgumentParser('미생 얼굴 전처리')
    parser.add_argument('plan', type=str, default='plan1', help='플랜')
    args = parser.parse_args()
    plan = ['plan1', 'plan2']
    if not args.plan in plan:
        raise('no plan {}'.format(args.plan))
    ct = CT()
    eval(args.plan + '(ct)')
    time.sleep(10)
