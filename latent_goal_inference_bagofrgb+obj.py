#!/usr/bin/env python
# coding: utf-8

# In[1]:


#
#
# ===================== 
# Training a Classifier
# =====================
# 
#

import time, os, copy, numpy as np

import torch, torchvision
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn import Parameter, init
import torch.nn.functional as F
import sys
import pickle
from queue import PriorityQueue
import heapq as hq 

import lmdb
import operator

import json

from torchsummary import summary

from pretrainedmodels import bninception

#torch.backends.cudnn.enabled = False
#torch.backends.cudnn.benchmark = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(device)


from collections import defaultdict
# ### Debug features class

import pickle

import random

import pandas as pd

import json

class EPICval(Dataset):
    def __init__(self, ann_file, obs_seg, seg_length, obj_lmdb_path, tsn_lmdb_path, classifier, transform=None):
        self.obj_feat_list = []
        self.tsn_feat_list = []
        self.uid_list = []
        
        count_empty_segs = 0
        
        action_annotation = pd.read_csv(ann_file, header=None)
        # [VID, Obs_start, Obs_end, Obs_noun, Obs_verb, Fut_start, Fut_end, Fut_noun,Fut_verb]
        
        obj_env = lmdb.open(obj_lmdb_path, readonly=True, lock=False)
        tsn_env = lmdb.open(tsn_lmdb_path, readonly=True, lock=False)
        
        video_ids = list(action_annotation[1].unique())
        
        for video_id in video_ids:
            video_id = video_id.strip('\n')
            starts = list(action_annotation.loc[action_annotation[1] == video_id][2].values)
            stops = list(action_annotation.loc[action_annotation[1] == video_id][3].values)
            uids = list(action_annotation.loc[action_annotation[1] == video_id][0].values)
            #print(stops[-1])
            
            # print(tsn_feat.shape)
            video_id = video_id.split()[0]
            with obj_env.begin() as obj_feats, tsn_env.begin() as tsn_feats:    

                tsn_feat_in_video = []
                obj_feat_in_video = []
                uid_in_video = []
                for start, stop, uid in zip(starts, stops, uids):
                    feat_frames = []
                    tsn_feat_in_seg = []
                    obj_feat_in_seg = []
                    # stop = stop - 30
                    # if stop - start > seg_length:
                    for i in range(int(stop)-seg_length,int(stop)):
                        # 'P24_03_frame_0000000578.jpg'
                        frame_num = video_id+'_frame_'+str(i).zfill(10)+'.jpg'
                        
                        ff_obj = obj_feats.get(frame_num.encode('utf-8'))
                        ff_tsn = tsn_feats.get(frame_num.encode('utf-8'))
                        
                        if ff_obj is not None or ff_tsn is not None:
                            obj_feat = np.frombuffer(ff_obj, 'float32')
                            obj_feat_in_seg.append(obj_feat.copy())

                            tsn_feat = torch.tensor(np.frombuffer(ff_tsn, 'float32').copy())
                            # print(tsn_feat.shape)
                            tsn_feat = classifier(tsn_feat).detach().cpu().numpy() # obtain softmax scores
                            tsn_feat_in_seg.append(tsn_feat)
                    # else:
                        # continue
                    #print(len(tsn_feat_in_seg))    
                    if len(obj_feat_in_seg) != 0 or len(tsn_feat_in_seg) != 0:
                        obj_feat_in_video.append(obj_feat_in_seg)
                        tsn_feat_in_video.append(tsn_feat_in_seg)
                        uid_in_video.append(uid)
                    else:
                        count_empty_segs += 1
                if len(obj_feat_in_video) != 0 or len(obj_feat_in_video) != 0:
                    self.obj_feat_list.extend(obj_feat_in_video)
                    self.tsn_feat_list.extend(tsn_feat_in_video)
                    self.uid_list.extend(uid_in_video)
                else:
                    print(video_id)
        print(count_empty_segs)         
        
    def __getitem__(self, index):

        obj_feat_seq = self.obj_feat_list[index]
        tsn_feat_seq = self.tsn_feat_list[index]
        uid = self.uid_list[index]
        
        return obj_feat_seq, tsn_feat_seq, uid

    def __len__(self):
        return len(self.uid_list)



# class definition 
class BeamSearchNode(object):
    def __init__(self, action_state, feat_state, goal_state, hidden, score, prev_node): 
        self.action_state = action_state
        self.feat_state = feat_state 
        self.goal_state = goal_state
        self.hidden = hidden
        self.prev_node = prev_node
        self.score = score

    def __lt__(self, nxt): 
        return self.score < nxt.score
        
    def __gt__(self, nxt): 
        return self.score > nxt.score
                
    
class Anticipator(nn.Module):
    def __init__(self, feature_dim, goal_smoothness, goal_closeness, hidden_size):
        super(Anticipator, self).__init__()
        
        self.hidden_size = hidden_size
        #self.goalpredictor = GoalPredictor()
        self.feat_embedding = nn.Linear(feature_dim, self.hidden_size)
        self.goal_steps = 5
        self.goalpredictor = nn.LSTM(2*self.hidden_size, self.hidden_size, self.goal_steps)
        self.epsilon = goal_closeness
        self.rnn = nn.LSTM(2*self.hidden_size, self.hidden_size, 1)
        self.predictor = nn.Linear(3*self.hidden_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.delta = goal_smoothness
        self.embedding2action = nn.Linear(self.hidden_size, 2513)
        self.embedding2verb = nn.Linear(self.hidden_size, 125)
        self.embedding2noun = nn.Linear(self.hidden_size, 352)
        self.embedding2feature = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(p=0.7)
        
        self.softmax = nn.Softmax(dim=1)
        
    def featpredictor(self, input1, input2, input3):
        #print(input1.shape)
        pred_feat = torch.cat((input1, input2),1)
        # print(input3.shape)
        # print(pred_feat.shape)
        pred_feat = self.predictor(torch.cat((pred_feat, input3),1))
        pred_feat = self.dropout(pred_feat)
        pred_feat = self.relu(pred_feat)
        
        return pred_feat
    
    def heapsort(iterable):
        h = []
        for value in iterable:
            hq.heappush(h, value)
        return [hq.heappop(h) for i in range(len(h))]
    
    def target_feat_embedding(self, target_feat):
        # print(target_feat.shape)
        feat_state, _ = torch.max(target_feat,1)
        #feat_state = feat_state[1,:,:]
        #feat_state = feat_state.squeeze(0)
        return feat_state
    
    def forward(self, tsn_feat_seq, batch_size=None):
        
        
        if len(tsn_feat_seq.shape) == 1:
            tsn_feat_seq = tsn_feat_seq.unsqueeze(0)
        #print(tsn_feat_seq.shape)
        feat_state, _ = torch.max(tsn_feat_seq,0)
        
        if len(feat_state.shape) == 1:
            feat_state = feat_state.unsqueeze(0)
        #print('feat_state',feat_state.shape)
        feat_state = self.feat_embedding(feat_state)

        obs_acts = feat_state.shape[0]+1
        # print(torch.isnan(feat_states).any())
        #print(obs_acts)
        
        #print('feat_state',feat_state.shape)
        
        best_feat_state = feat_state
        
        nodes = PriorityQueue()
        
        end_nodes = []
        
        action_state = torch.zeros(1, self.hidden_size).to(device)
        score = 1.0
        
        #print(torch.cat((feat_state, action_state),2).shape)
        # print(best_action_state.shape)
        # print(best_feat_state.shape)
        hidden = torch.zeros(self.goal_steps, 1, self.hidden_size).to(device)
        cell = torch.zeros(self.goal_steps, 1, self.hidden_size).to(device)
        # print(feat_state.shape)
        # print(action_state.shape)
        
        goal_state, (hidden, _) = self.goalpredictor(torch.cat((feat_state.unsqueeze(0), action_state.unsqueeze(0)), 2), (hidden, cell))
        goal_state = goal_state.squeeze(1)
        node = BeamSearchNode(action_state, feat_state, goal_state, hidden, score, None)
        nodes.put(node)
        qsize = 1
        endnodes = []
        while True:
            #print(best_feat_state.shape)
            # print(best_action_state.shape)
            if qsize > obs_acts*6:
                break
            if nodes.qsize() != 0:    
                n = nodes.get()
            else:
                break
            feat_state = n.feat_state
           
            action_state = n.action_state
            goal_state = n.goal_state
            hidden = n.hidden
            #print('feat_state',feat_state)
            #print('dist',torch.square(torch.dist(feat_state, goal_state,2))/self.hidden_size)
            if torch.square(torch.dist(feat_state, goal_state,2))/self.hidden_size > self.epsilon:
                h_act = torch.zeros(1, 1, self.hidden_size).to(device)
                c_act = torch.zeros(1, 1, self.hidden_size).to(device)

                #print(len(end_nodes))
                #print(number_required)
                endnodes.append(n)
                if len(end_nodes) >= obs_acts:
                    break
                
                action_state_list = []
                for j in range(10): 
                    _, (h_act, c_act) = self.rnn(torch.cat((action_state.unsqueeze(0), feat_state.unsqueeze(0)), 2), (h_act, c_act))
                    #print(j)
                    action_state_list.append(h_act.squeeze(0))   
                next_nodes = []
                
                # print(len(action_state_list))
                for action_state in action_state_list:
                    next_feat = self.featpredictor(feat_state, action_state, goal_state)    
                    #next_feat = feat_states[i,:].unsqueeze(0)
                    #next_hidden, next_goal_state = self.goalpredictor(feat_state, action_state, hidden)
                    # next_goal_state, (next_hidden, _) = \
                    #         self.goalpredictor(torch.cat((feat_state.unsqueeze(0), action_state.unsqueeze(0)), 2), (hidden, cell))
                    # next_goal_state = self.relu(next_goal_state.squeeze(1))
                    score = torch.square(torch.dist(best_feat_state, goal_state, 2))/self.hidden_size \
                                - torch.square(torch.dist(next_feat, goal_state, 2))/self.hidden_size 
                                #+ torch.square(torch.dist(next_goal_state, goal_state,2))/self.hidden_size
                    next_node = BeamSearchNode(action_state, next_feat, goal_state, hidden, score, n)
                    next_nodes.append(next_node)
                    if torch.square(torch.dist(next_feat, goal_state,2))/self.hidden_size < torch.square(torch.dist(best_feat_state, goal_state, 2))/self.hidden_size:\
                        #torch.square(torch.dist(next_goal_state, goal_state,2))/self.hidden_size < self.delta:
                        best_feat_state = next_feat
                
                for k in range(len(next_nodes)):
                    node = next_nodes[k]
                    nodes.put(node)
                qsize += len(next_nodes) - 1
        #print('num nodes',nodes.qsize())
        #print(len(end_nodes))
        if len(end_nodes) == 0:
            if nodes.qsize() != 0:
                end_nodes = [nodes.get() for _ in range(obs_acts)]
        #print('num end_nodes', len(end_nodes))
        if nodes.qsize() == 0:
            print(len(end_nodes))
            print('empty')
        pred_action_states = []
        pred_feat_states = []
        pred_goal_states = []
        #print('before sorted')
        # print(obs_acts)
        l = 0
        for n in sorted(end_nodes, key=operator.attrgetter('score')):
            # print(n.score)
            l += 1
            pred_action_state = []
            pred_feat_state = []
            pred_goal_state = []
            pred_action_state.append(n.action_state)
            pred_feat_state.append(n.feat_state)
            pred_goal_state.append(n.goal_state)
            # back trace
            while n.prev_node != None:
                #print(l)
                n = n.prev_node
                pred_action_state.append(n.action_state)
                pred_feat_state.append(n.feat_state)
                pred_goal_state.append(n.goal_state)
                
            pred_action_state = pred_action_state[::-1]
            pred_feat_state = pred_feat_state[::-1]
            pred_goal_state = pred_goal_state[::-1]
            pred_action_states.append(pred_action_state)
            pred_feat_states.append(pred_feat_state)
            pred_goal_states.append(pred_goal_state)
                
        
        # for pred_feat_state in pred_feat_states:
            # for pred_feat in pred_feat_state:
                # print(pred_feat.shape)
        pred_action_states = [ torch.stack(state).squeeze(1) for state in pred_action_states]
        pred_action_states = [ torch.mean(state, 0, keepdim=True) for state in pred_action_states]
        pred_action_states = torch.stack(pred_action_states).squeeze(1)
        # print(pred_action_states.shape)
        pred_action_states = self.dropout(pred_action_states)
        pred_actions = self.embedding2action(pred_action_states)
        pred_verbs = self.embedding2verb(pred_action_states)
        pred_nouns = self.embedding2noun(pred_action_states)
        #pred_actions = self.relu()
        
        # print(pred_actions.shape)
        pred_feat_states = [ torch.stack(state).squeeze(1) for state in pred_feat_states]
        pred_feat_states = [ torch.mean(state, 0, keepdim=True) for state in pred_feat_states]
        pred_feat_states = torch.stack(pred_feat_states).squeeze(1)
        # print(pred_feat_states.shape)
        pred_goal_states = [ torch.stack(state).squeeze(1) for state in pred_goal_states]
        pred_goal_states = [ torch.mean(state, 0, keepdim=True) for state in pred_goal_states]
        pred_goal_states = torch.stack(pred_goal_states).squeeze(1)
        # print(pred_goal_states.shape)
        return pred_actions, pred_verbs, pred_nouns, pred_feat_states, pred_goal_states, feat_state

def largest_indices(array: np.ndarray, n: int) -> tuple:
    """Returns the n largest indices from a numpy array.
    Arguments:
        array {np.ndarray} -- data array
        n {int} -- number of elements to select
    Returns:
        tuple[np.ndarray, np.ndarray] -- tuple of ndarray
        each ndarray is index
    """
    flat = array.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, array.shape)

def evaluate(test_ann_file, json_file, obj_lmdb_path, tsn_lmdb_path, verb_anticipation_model, noun_anticipation_model, classifier):
    test_set = EPICval(test_ann_file, obs_seg, int(seg_length_sec*frame_rate), obj_lmdb_path, tsn_lmdb_path, classifier)
    print('{} test_seen instances.'.format(len(test_set)))
    testloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    predictions = {}
    predictions = {'version': '0.1',\
                  'challenge': 'action_anticipation', 'results': {}}
    possible_actions = pd.read_csv('actions.csv',index_col='id')

    for i, data in enumerate(testloader, 0): 
        obj_feat_seq = []
        tsn_feat_seq = []
        
        obj_seq_segs, tsn_seq_segs, uid = data 
        for obj_seq, tsn_seq in zip(obj_seq_segs, tsn_seq_segs): 
            #print(tsn_seq.shape)
            obj_feat_seq.append(obj_seq)
            tsn_feat_seq.append(tsn_seq)
        obj_feat_seq = torch.stack(obj_feat_seq)
        tsn_feat_seq = torch.stack(tsn_feat_seq)
        
        # tsn_feat_seq = (tsn_feat_seq - mean_values)/(std_values+1e-9)
        obj_feat_seq = obj_feat_seq.float().squeeze(0).to(device)
        tsn_feat_seq = tsn_feat_seq.float().squeeze(0).to(device)
        # print(tsn_feat_seq.shape)
        # print(obj_feat_seq.shape)
        if obj_feat_seq.shape[1] == 0 or tsn_feat_seq.shape[1] == 0:
            continue
        
        pred_actions_tsn, pred_verbs_tsn, pred_nouns_tsn, pred_feats,\
            goal_states, obs_feat_states = verb_anticipation_model(tsn_feat_seq)
        pred_actions_obj, pred_verbs_obj, pred_nouns_obj, pred_feats,\
            goal_states, obs_feat_states = noun_anticipation_model(obj_feat_seq)

        # print(len(pred_verb))
        pred_verb = pred_verbs_tsn[-1,:].squeeze().detach().cpu().numpy()
        top_verb = list(largest_indices(pred_verb, 125)[0])
        pred_noun = pred_nouns_obj[-1,:].squeeze().detach().cpu().numpy()
        top_noun = list(largest_indices(pred_noun, 352)[0])
        pred_action = pred_actions_tsn[-1,:].squeeze().detach().cpu().numpy()
        top_action = list(largest_indices(pred_action, 100)[0])
        top_action_tuples = []
        for action in top_action:
            top_action_tuples.append((possible_actions.loc[action]['verb'], possible_actions.loc[action]['noun']))
                
        # print(top100_action)
        # print(uid)
        # print(len(pred_verb))
        uid = int(uid.item())
      
        predictions['results'][str(uid)] = {}
        predictions['results'][str(uid)]['verb'] = {str(ii): float(pred_verb[ii]) for ii in top_verb}
        predictions['results'][str(uid)]['noun'] = {str(ii): float(pred_noun[ii]) for ii in top_noun}
        predictions['results'][str(uid)]['action'] = {str(v)+','+str(n): float(pred_action[ii]) for (v,n),ii in zip(top_action_tuples, top_action)}      
          
 
    with open(json_file, 'w') as fp:
        json.dump(predictions, fp,  indent=4)

# ### define hyperparameters
goal_closeness = 1e-5
goal_smoothness = 1e-5
#instantiate the model
obs_segs = [1, 2, 3, 4]
seg_length_secs = [0.5, 1, 2, 3, 5, 10, 15, 20 ,25]

obs_seg = obs_segs[0]
seg_length_sec = seg_length_secs[2]
frame_rate = 30 # tsn features calculated @ 10fps
    
# writer = SummaryWriter('runs/epic_noun_max_{:1.1f}x{:d}'.format(seg_length_sec, obs_seg))
nepochs = 20

goal_closeness = 1e-5
goal_smoothness = 1e-5
criterion = nn.CrossEntropyLoss()

tsn_lmdb_path = '/home/roy/epic_rgb_full_features'
obj_lmdb_path = '/home/roy/epic_bagofobj_full_features'

model = bninception(pretrained=None)
state_dict = torch.load('../rulstm/FEATEXT/models/TSN-rgb.pth.tar')['state_dict']
state_dict = {k.replace('module.base_model.','') : v for k,v in state_dict.items()}
state_dict = {k.replace('module.','') : v for k,v in state_dict.items()}
del model.last_linear
model.new_fc= nn.Linear(1024, 2513)
model.load_state_dict(state_dict)

classifier = model.new_fc

# predict with tsnrgb
verb_feature_dim = 2513
verb_hidden_size = 1024
verb_anticipation_model = Anticipator(verb_feature_dim, goal_smoothness, goal_closeness, verb_hidden_size)
verb_ckpt_path = 'ckpt/bagoftsnrgb_action_latent_goal_{:1.1f}sx{:d}_obs_max.pt'.format(seg_length_sec, obs_seg)
verb_state = torch.load(verb_ckpt_path)
verb_anticipation_model.load_state_dict(verb_state['model'])
verb_anticipation_model.to(device)    

# predict with bagofobj
noun_feature_dim = 352
noun_hidden_size = 1024
noun_anticipation_model = Anticipator(noun_feature_dim, goal_smoothness, goal_closeness, noun_hidden_size)
noun_ckpt_path = 'ckpt/bagofobj_action_latent_goal_{:1.1f}sx{:d}_obs_max.pt'.format(seg_length_sec, obs_seg)
noun_state = torch.load(noun_ckpt_path)
noun_anticipation_model.load_state_dict(noun_state['model'])
noun_anticipation_model.to(device)

test_ann_file = 'test_seen.csv'
json_file = 'seen.json'
evaluate(test_ann_file, json_file, obj_lmdb_path, tsn_lmdb_path, verb_anticipation_model, noun_anticipation_model, classifier)
test_ann_file = 'test_unseen.csv'
json_file = 'unseen.json'
evaluate(test_ann_file, json_file, obj_lmdb_path, tsn_lmdb_path, verb_anticipation_model, noun_anticipation_model, classifier)
