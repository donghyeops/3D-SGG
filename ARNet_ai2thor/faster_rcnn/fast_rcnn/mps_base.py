#-*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from utils.timer import Timer
import pdb


class Message_Passing_Unit_v2(nn.Module):
    def __init__(self, fea_size, filter_size = 128):
        super(Message_Passing_Unit_v2, self).__init__()
        self.w = nn.Linear(fea_size, filter_size, bias=True)
        self.fea_size = fea_size
        self.filter_size = filter_size

    def forward(self, unary_term, pair_term):
        #print('v2')
        if unary_term.size()[0] == 1 and pair_term.size()[0] > 1:
            unary_term = unary_term.expand(pair_term.size()[0], unary_term.size()[1])
        if unary_term.size()[0] > 1 and pair_term.size()[0] == 1:
            pair_term = pair_term.expand(unary_term.size()[0], pair_term.size()[1])
        
        # print '[unary_term, pair_term]', [unary_term, pair_term]
        gate = self.w(F.relu(unary_term)) * self.w(F.relu(pair_term))
        gate = F.sigmoid(gate.sum(1))
        # print 'gate', gate
        output = pair_term * gate.expand(gate.size()[0], pair_term.size()[1])
        
        return output

# 식 (1)[r*input] & (2)[update gate]
class Message_Passing_Unit_v1(nn.Module):
    def __init__(self, fea_size, filter_size = 128):
        super(Message_Passing_Unit_v1, self).__init__()
        self.w = nn.Linear(fea_size * 2, filter_size, bias=True) 
        self.fea_size = fea_size
        self.filter_size = filter_size

    def forward(self, unary_term, pair_term):
        # unary_term는 식 (2)의 첫 번째 인자
        # pair_term은 식 (2)의 두 번째 인자
        # 각 입력값의 shape을 맞춰줌. 부족한 쪽을 늘림
        #print('v1')
        if unary_term.size()[0] == 1 and pair_term.size()[0] > 1:
            unary_term = unary_term.expand(pair_term.size()[0], unary_term.size()[1])
        if unary_term.size()[0] > 1 and pair_term.size()[0] == 1:
            pair_term = pair_term.expand(unary_term.size()[0], pair_term.size()[1])
        
        # print '[unary_term, pair_term]', [unary_term, pair_term]
        gate = torch.cat([unary_term, pair_term], 1)
        gate = F.relu(gate)
        gate = F.sigmoid(self.w(gate)).mean(1) # r값
        # print 'gate', gate
        # torch.tensor.view(X) <= tf.reshape(tensor, (X)) 이랑 똑같음
        # torch.Size(tensor) == tf.get_shape(tensor)
        # tc.tensor.expand(X) ~~ tf.tile(tensor, (X)) tf.tile은 해당 부분만 몇 배를 해줄 지 쓰지만, 토치는 원하는 shape을 쓰면 알아서 복사됨
        output = pair_term * gate.view(-1, 1).expand(gate.size()[0], pair_term.size()[1])
        
        return output

# 식 (3) or (4)
class Gated_Recurrent_Unit(nn.Module):
    def __init__(self, fea_size, dropout):
        super(Gated_Recurrent_Unit, self).__init__()
        self.wih = nn.Linear(fea_size, fea_size, bias=True)
        self.whh = nn.Linear(fea_size, fea_size, bias=True)
        self.dropout = dropout

    def forward(self, input, hidden):
        output = self.wih(F.relu(input)) + self.whh(F.relu(hidden))
        if self.dropout:
            output = F.dropout(output, training=self.training)
        return output



class Hierarchical_Message_Passing_Structure_base(nn.Module):
    def __init__(self, fea_size, dropout=False, gate_width=128, use_region=True, use_kernel_function=False):
        # fea_size            = 1024
        # dropout             = 
        # gate_width          = 128
        # use_kernel_function = False
        
        super(Hierarchical_Message_Passing_Structure_base, self).__init__()
        #self.w_object = Parameter()
        if use_kernel_function:
            Message_Passing_Unit = Message_Passing_Unit_v2
        else: #false로 들어감
            Message_Passing_Unit = Message_Passing_Unit_v1
            
        self.gate_sub2pred = Message_Passing_Unit(fea_size, gate_width) 
        self.gate_obj2pred = Message_Passing_Unit(fea_size, gate_width) 
        self.gate_pred2sub = Message_Passing_Unit(fea_size, gate_width) 
        self.gate_pred2obj = Message_Passing_Unit(fea_size, gate_width) 

        self.GRU_object = Gated_Recurrent_Unit(fea_size, dropout) # nn.GRUCell(fea_size, fea_size) #
        self.GRU_phrase = Gated_Recurrent_Unit(fea_size, dropout)

        if use_region:
            self.gate_pred2reg = Message_Passing_Unit(fea_size, gate_width) 
            self.gate_reg2pred = Message_Passing_Unit(fea_size, gate_width) 
            self.GRU_region = Gated_Recurrent_Unit(fea_size, dropout)
        


    def forward(self, feature_obj, feature_phrase, feature_region, mps_object, mps_phrase, mps_region):
        raise Exception('Please implement the forward function')

    # Here, we do all the operations outof loop, the loop is just to combine the features
    # Less kernel evoke frequency improve the speed of the model
    def prepare_message(self, target_features, source_features, select_mat, gate_module):
        # target_features : 영향 받을 놈
        # source_features : 영향 줄 놈
        # select_mat : 동적 그래프 (2차원임) (타겟노드, 소스노드)
        # gate_module : MPU 함수 (두 피쳐에 대한 r 값을 리턴함)
        feature_data = []

        transfer_list = np.where(select_mat > 0)
        source_indices = Variable(torch.from_numpy(transfer_list[1]).type(torch.LongTensor)).cuda()
        target_indices = Variable(torch.from_numpy(transfer_list[0]).type(torch.LongTensor)).cuda()
        source_f = torch.index_select(source_features, 0, source_indices)
        target_f = torch.index_select(target_features, 0, target_indices)
        transferred_features = gate_module(target_f, source_f)

        for f_id in range(target_features.size()[0]):
            if len(np.where(select_mat[f_id, :] > 0)[0]) > 0:
                feature_indices = np.where(transfer_list[0] == f_id)[0]
                indices = Variable(torch.from_numpy(feature_indices).type(torch.LongTensor)).cuda()
                features = torch.index_select(transferred_features, 0, indices).mean(0).view(-1)
                feature_data.append(features)
            else:
                temp = Variable(torch.zeros(target_features.size()[1:]), requires_grad=True).type(torch.FloatTensor).cuda()
                feature_data.append(temp)
        return torch.stack(feature_data, 0)