#
# =====================
# Training a Classifier
# =====================
# 
 

import time, os, copy, numpy as np

import torch, torchvision
import torch.nn as nn
from torch.nn import Parameter, init
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import sys
import pickle
import math
from collections import defaultdict
import random


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(device)
# ## Get object features, distances and frame labels

# ### Helper functions


from PIL import Image
object_classes = {'bottle': 0, 'bowl': 1, 'cheese': 2, 'cucumber': 3, 'knife': 4, 
                  'lettuce': 5, 'peeler': 6, 'spoon': 7, 'tomato': 8, 'hand': 9}

class LongTermFrameFeat(Dataset):
    features = None
    def __init__(self, annot_dir, i3d_feature_dir, sequence_ids, obs_seg, seg_length, transform=None):
        self.label_sequences = []
        self.i3d_sequences = []
        self.target_labels = []
        self.target_features = []
        
        self.action_classes = {'SIL':0,'cut_fruit':1,'put_fruit2bowl':2,'peel_fruit':3,'stir_fruit':4,
                               'crack_egg':5,'add_saltnpepper':6,'stir_egg':7,'pour_oil':8,
                               'pour_egg2pan':9,'stirfry_egg':10,'take_plate':11,'put_egg2plate':12,
                               'pour_coffee':13,'pour_milk':14,'spoon_powder':15,'stir_milk':16,
                               'pour_cereals':17,'stir_cereals':18,'pour_flour':19,'stir_dough':20,
                               'pour_dough2pan':21,'fry_pancake':22,'put_pancake2plate':23,
                               'add_teabag':24,'pour_water':25,'cut_orange':26,'squeeze_orange':27,
                               'take_glass':28,'pour_juice':29,'fry_egg':30,'cut_bun':31,
                               'take_butter':32,'smear_butter':33,'put_toppingOnTop':34,
                               'put_bunTogether':35,'spoon_flour':36,'butter_pan':37,'take_eggs':38,
                               'take_cup':39,'pour_sugar':40,'stir_coffee':41,'take_bowl':42,
                               'take_knife':43,'spoon_sugar':44,'take_topping':45,'take_squeezer':46,
                               'stir_tea':47}
        self.transform = transform
        
        for sequence_id in sequence_ids:
            # print(sequence_id)
            try:
                labels = open(os.path.join(annot_dir,sequence_id.strip('\n')+'.txt'),'r').readlines()
                i3d_sequence_file = os.path.join(i3d_feature_dir, sequence_id.strip('\n')+'.npy')
                i3d_feat = np.load(i3d_sequence_file)
            except:
                continue
            
            label_sequence = []
            i3d_sequence = []
            label_segs = []
            i3d_segs = []
            # print(len(labels))
            
                
            # print(i3d_feat.shape)
            for frame_num in range(len(labels)):
                action_label = labels[frame_num].strip('\n')
                action_label = self.action_classes[action_label] 
                # print(len(features_frame))
                # if action_label == -1:
                #     continue
                try:
                    i3d_feature_frame = i3d_feat[frame_num,:]
                    label_sequence.append(action_label)
                    i3d_sequence.append(i3d_feature_frame)
                except:
                    continue
                    
            #print(len(label_sequence))
            #print(len(i3d_sequence))
            # print(len(dt_sequence))
            seq_len = len(label_sequence)
            for i in range(seq_len):
                if i % seg_length == 0 and i > 0:
                    label_segs.append(label_sequence[i-seg_length+15])
                    i3d_segs.append(i3d_sequence[i-seg_length:i:5]) # 3fps
            #print(seg_start)
            final = len(label_segs)- obs_seg - 2
            for i in range(0, final):
                self.label_sequences.append(label_segs[i:i+obs_seg])
                self.i3d_sequences.append(i3d_segs[i:i+obs_seg])
                self.target_labels.append(label_segs[i+obs_seg])
                self.target_features.append(i3d_segs[i+obs_seg][3:])

   
    def __getitem__(self, index):
        
        i3d_seq_segs = self.i3d_sequences[index]
        label_segs = self.label_sequences[index]
        target_label = self.target_labels[index]
        target_feats = self.target_features[index]
        
        return i3d_seq_segs, label_segs, target_label, target_feats  
    
    def __len__(self):
        return len(self.label_sequences)
    
    def get_weights(self):
        weights = np.zeros(47)
        values, counts = np.unique(self.target_labels, return_counts=True)
        for value, count in zip(values, counts):
            weights[value] = count
        weights = weights/len(self.target_labels)
        return weights
    
class GoalPredictor(nn.Module):
    def __init__(self):
        super(GoalPredictor, self).__init__()
        
        self.rnn1 = nn.LSTMCell(2048, 2048)
        self.softmax = nn.Softmax(dim=1)
        self.W_h = nn.Linear(2048, 2048)
        self.W_c = nn.Linear(2048, 2048)
        self.pool = nn.AvgPool1d(obs_segs[-1])
        self.W_f = nn.Linear(2048, 2048)
        self.relu = nn.ReLU()
        self.rnn2 = nn.LSTMCell(2*2048, 2048)
        
    def forward(self, feat_state, action_state, hidden_now, batch_size=None):
           
        future_hidden = self.relu(self.W_h(hidden_now))
        r_t = action_state
        future_hiddens = []
        cell_state = torch.zeros(1, 2048).to(device)
        for i in range(obs_segs[-1]):
            future_hidden, cell_state = self.rnn1(r_t, (future_hidden, cell_state))
            r_t = self.softmax(self.W_c(future_hidden))
            future_hiddens.append(future_hidden)
        future_hiddens = torch.stack(future_hiddens).permute(2,1,0)
        #print(future_hiddens.shape)
        pooled = self.pool(future_hiddens).squeeze().unsqueeze(0)
        # print(pooled.shape)
        x_t_est = self.relu(self.W_f(pooled))
        # print(x_t_est.shape)
        cell_state = torch.zeros(1, 2048).to(device)
        h_next, _ = self.rnn2(torch.cat((feat_state, x_t_est),1), (hidden_now, cell_state))
        # print(h_next.shape)
                
        return h_next, x_t_est
        
class ActionAnticipator(nn.Module):
    def __init__(self):
        super(ActionAnticipator, self).__init__()
        
        self.feat_embedding = nn.LSTM(2048, 2048, 1)
        self.action_embedding = nn.Linear(48, 2048)
        self.goalpredictor = GoalPredictor()
        self.epsilon = 0.0005
        self.rnn = nn.LSTMCell(2*2048, 2048)
        self.predictor = nn.Linear(2*2048, 2048)
        self.relu = nn.ReLU()
        self.delta = 0.0005
        self.embedding2action = nn.Linear(2048, 48)
        self.softmax = nn.Softmax(dim=1)
        
    def feat_predictor(self, input1, input2, input3):
        #print(input1.shape)
        pred_feat = self.predictor(torch.cat((input1, input2),1))
        pred_feat = self.predictor(torch.cat((pred_feat, input3),1))
        pred_feat = self.relu(pred_feat)
        
        return pred_feat
        
    def target_feat_embedding(self, target_feat):
        # print(target_feat.shape)
        _, (feat_state, _) = self.feat_embedding(target_feat)
        feat_state = feat_state.squeeze(0)
        return feat_state
    
    def forward(self, i3d_feat_seq, batch_size=None):
        # print(i3d_feat_seq.shape)
        # print(type(i3d_feat_seq))
        if len(i3d_feat_seq.shape) == 4:
            i3d_feat_seq = i3d_feat_seq.squeeze().permute(1,0,2)
            #print(i3d_feat_seq.shape)
            _, (feat_states, _) = self.feat_embedding(i3d_feat_seq)
        else:
            _, (feat_states, _) = self.feat_embedding(i3d_feat_seq)
            
        
        feat_states = feat_states.squeeze(0)
        #print(feat_states.shape)
        obs_acts = feat_states.shape[0]
        best_feat_state = feat_states[0,:].unsqueeze(0)
        
        action_state = torch.zeros(1,2048).to(device)
        best_action_state = action_state
        
        
        # print(best_action_state.shape)
        # print(best_feat_state.shape)
        best_action_states = []
        best_feat_states = []
        hidden = torch.zeros(1, 2048).to(device)
        hidden, goal_state = self.goalpredictor(best_feat_state, best_action_state, hidden)
        for i in range(obs_acts):
            action_list = []
            # print(best_feat_state.shape)
            # print(best_action_state.shape)
            
            
            if torch.square(torch.dist(feat_states[i,:], goal_state,2)) > self.epsilon:
                h_act = torch.zeros(1, 2048).to(device)
                c_act = torch.zeros(1, 2048).to(device)
                
                for j in range(6):
                    h_act, c_act = self.rnn(torch.cat((best_action_state, best_feat_state), 1), (h_act, c_act))
                    action_list.append(h_act)   
                goal_state_est = torch.zeros_like(goal_state)
                hidden_est = torch.zeros_like(hidden)
                for action in action_list:
                    next_feat = self.feat_predictor(best_feat_state, action, goal_state)    
                    hidden_new, goal_state_new = self.goalpredictor(best_feat_state, best_action_state, hidden)
                    if torch.square(torch.dist(next_feat, goal_state,2)) < torch.square(torch.dist(best_feat_state, goal_state, 2)) and \
                        torch.square(torch.dist(goal_state_new, goal_state,2)) < self.delta:
                        best_feat_state = next_feat
                        best_action_state = action
                        hidden_est = hidden_new
                        goal_state_est = goal_state_new
                        
                best_action_states.append(best_action_state)
                hidden = hidden_est
                goal_state = goal_state_est
        h_act = torch.zeros(1, 2048).to(device)
        c_act = torch.zeros(1, 2048).to(device)
        ant_action, _ = self.rnn(torch.cat((best_action_state, best_feat_state), 1), (h_act, c_act))
        
        best_action_states.append(ant_action)
        best_action_states = torch.stack(best_action_states).squeeze(1)
        best_actions = self.embedding2action(best_action_states)
       
        #print(best_action_states.shape)
        return best_actions, best_feat_state

class TrainTest():
    
    def __init__(self, model, trainset, testset, batch_size, nepoch, ckpt_path):
        self.model = model
        self.optimizer = optim.Adam(model.parameters())
        #self.optimizer = optim.SGD(model.parameters(), lr=0.0005, weight_decay=0.9 )
        self.criterion1 = nn.CrossEntropyLoss()
        self.criterion2 = nn.MSELoss()
        
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=True, num_workers=0)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                          shuffle=False, num_workers=0)
          
        self.model.to(device)    
        self.chkpath = ckpt_path
        self.batch_size = batch_size
        self.nepoch = nepoch
        self.trainset_size = len(trainset)
        self.mse = nn.MSELoss(reduction='none')
        if not os.path.exists('ckpt/'):
            os.mkdir('ckpt/')            
        print(self.chkpath)
        if os.path.exists(self.chkpath) == True:
            print('load from ckpt', end=' ')
            self.state = torch.load(self.chkpath)
            self.model.load_state_dict(self.state['model'])
            best_acc = self.state['acc']
            start_epoch = self.state['epoch']
            print('Epoch {}'.format(start_epoch))
            if start_epoch == self.nepoch:
                print('existing as epoch is max.')
            self.details = self.state['details']    
            self.best_acc = best_acc
            self.start_epoch = start_epoch + 1
            self.model.to(device)                    
        else:
            self.best_acc = -1.
            self.details = []   
            self.start_epoch = 0
    
    def test(self):
        correct = 0
        count = 0
        sequence = []
        with torch.no_grad():
            for i, data in enumerate(self.testloader, 0):    
                #print(len(data))
                loss = 0.
                i3d_feat_seq = []
                obs_label_seq = []
                i3d_seq_segs, obs_label_segs, target_label, target_feats = data   
                for i3d_seq, obs_label in zip(i3d_seq_segs, obs_label_segs): 
                    seg_feat = torch.stack(i3d_seq)
                    i3d_feat_seq.append(seg_feat)
                    obs_label_seq.append([int(obs_label)])
                i3d_feat_seq = torch.stack(i3d_feat_seq).float().squeeze(0).to(device)
                # print(i3d_feat_seq.shape)
                
                target_feat = torch.stack(target_feats).float().to(device)
                target_feat = self.model.target_feat_embedding(target_feat)
                
                target_actions = obs_label_seq
                target_actions.append([int(target_label)])
                action_label = torch.LongTensor(obs_label_seq[0]).unsqueeze(0)
                # print(action_label.shape)
                # print(obs_label_tensor.shape)
                # print(target_seq.shape)
                # print(action_label_tensor.is_cuda)
                pred_actions, pred_feat = self.model(i3d_feat_seq)
                target_actions = torch.LongTensor(target_actions).squeeze().to(device)
                # print(action_label.shape)

                
                # print(obs_label_tensor.shape)
                # print(i3d_feat_seq.shape)
                #print(target_seq.shape)
                if target_actions.shape[0] != pred_actions.shape[0]:
                    continue
                
                
                pred_action = pred_actions[-1,:].view(1,-1)
                # print(pred_action.shape)
                target_action = target_actions[-1].unsqueeze(0)
                loss += self.criterion1(pred_actions, target_actions)/target_actions.shape[0]
                # print(loss)
                loss += self.criterion2(pred_feat, target_feat)
                # print(loss) 
                # print(pred_seq.shape)
                ant_action = torch.argmax(pred_actions[-1,:])
                correct = correct + torch.sum(ant_action==target_actions[-1]).item()                                     
                
                print("\rIteration: {}/{}, Loss: {}.".format(i+1, len(self.testloader), loss), end="")
                sys.stdout.flush()
                count += 1
                #print(count)    
        return correct/count*100., correct   
        
    def train(self):        
        for epoch in range(self.start_epoch,self.nepoch):  
            start_time = time.time()        
            running_loss = 0.0
            correct = 0.
            count = 0.
            total_loss = 0
            self.optimizer.zero_grad()   
            loss = 0.
            iterations = 0
            for i, data in enumerate(self.trainloader, 0):        
                #print(len(data))
                i3d_feat_seq = []
                obs_label_seq = []
                i3d_seq_segs, obs_label_segs, target_label, target_feats = data   
                for i3d_seq, obs_label in zip(i3d_seq_segs, obs_label_segs): 
                    seg_feat = torch.stack(i3d_seq)
                    i3d_feat_seq.append(seg_feat)
                    obs_label_seq.append([int(obs_label)])
                i3d_feat_seq = torch.stack(i3d_feat_seq).float().squeeze(0).to(device)
                # print(i3d_feat_seq.shape)
                
                target_feat = torch.stack(target_feats).float().to(device)
                target_feat = self.model.target_feat_embedding(target_feat)
                
                target_actions = obs_label_seq
                target_actions.append([int(target_label)])
                action_label = torch.LongTensor(obs_label_seq[0]).unsqueeze(0)
                # print(action_label.shape)
                
                # print(action_label_tensor.is_cuda)
                pred_actions, pred_feat = self.model(i3d_feat_seq)
                target_actions = torch.LongTensor(target_actions).squeeze().to(device)
                # print(target_actions.shape)
                
                if target_actions.shape[0] != pred_actions.shape[0]:
                    continue
                
                pred_action = pred_actions[-1,:].view(1,-1)
                # print(pred_action.shape)
                target_action = target_actions[-1].unsqueeze(0)
                loss += self.criterion1(pred_actions, target_actions)/target_actions.shape[0]
                # print(loss)
                loss += self.criterion2(pred_feat, target_feat)
                # print(loss) 
                # print(pred_seq.shape)
                ant_action = torch.argmax(pred_actions[-1,:])
                # print(ant_action)
                with torch.no_grad():
                    correct = correct + torch.sum(ant_action==target_actions[-1]).item()                    
                
                if i % batch_size == 0 and i>1:
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()                    
                    running_loss = running_loss + loss.item()
                    print("\rIteration: {}/{}, Loss: {}.".format(i+1, len(self.trainloader), loss), end="")
                    loss = 0.
                    count += self.batch_size
                    iterations += 1
            TRAIN_LOSS =  running_loss/iterations
            TRAIN_ACC = correct/count*100
            TEST_ACC, TEST_COUNT = self.test()
            self.details.append((TRAIN_LOSS,TRAIN_ACC,0.,TEST_ACC))
            #utils.draw_Fig(self.details)
            #plt.savefig('results/sum_classifier.png')        
            #plt.close()
            if TEST_ACC > self.best_acc:                
                self.state = {
                    'model': self.model.state_dict(),
                    'acc': TEST_ACC,
                    'epoch': epoch,
                    'details':self.details,            
                }        
                torch.save(self.state, self.chkpath)
                self.best_acc = TEST_ACC
            else:
                self.state['epoch'] = epoch
                torch.save(self.state, self.chkpath)
            elapsed_time = time.time() - start_time
            print('[{}] [{:.1f}] [Loss {:.3f}] [Correct : {}] [Trn. Acc {:.1f}] '.format(epoch, elapsed_time,
                    TRAIN_LOSS, correct,TRAIN_ACC),end=" ")
            print('[Test Cor {}] [Acc {:.1f}]'.format(TEST_COUNT,TEST_ACC))

# ### define hyperparameters

num_classes = 48

obs_segs = [1, 2, 3, 4]
seg_lengths = [30, 45, 75, 150]
seg_length_secs = [2, 3, 5, 10]

obs_seg = obs_segs[0]
seg_length = seg_lengths[3]
seg_length_sec = seg_length_secs[3]
    
nepochs = 10
batch_size = 8

annot_dir = '../breakfast_framewise_annotations/'
i3d_feature_dir = '/home/roy/breakfast_i3dfeatures/'

sequence_ids_train = sorted(open('train.s1').readlines())
training_set = LongTermFrameFeat(annot_dir, i3d_feature_dir, sequence_ids_train[:], obs_seg, seg_length)
print('{} training instances.'.format(len(training_set)))


sequence_ids_test = open('test.s1').readlines()
test_set = LongTermFrameFeat(annot_dir, i3d_feature_dir, sequence_ids_test[:], obs_seg, seg_length)
print('{} test instances.'.format(len(test_set)))


model_ft = ActionAnticipator()
ckpt_path = 'ckpt/i3d_latent_goal_{:d}sx{:d}_obs_lstm.pt'.format(seg_length_sec, obs_seg)

# tgtembed = TargetEmbedding(seg_length_sec)
# weights = torch.Tensor(training_set.get_weights()).to(device)
# print(weights.shape)
EXEC = TrainTest(model_ft, training_set, test_set, batch_size, nepochs, ckpt_path)
EXEC.train()

testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

from sklearn.metrics import classification_report
pred_labels = []
target_labels = []
state = torch.load(ckpt_path)
model_ft.load_state_dict(state['model'])
model_ft.to(device)

correct = 0
actual_pred_seq = []
actual_target_seq = []
with torch.no_grad():
    for i, data in enumerate(testloader, 0):    
        loss = 0.
        #print(len(data))
        i3d_feat_seq = []
        obs_label_seq = []
        i3d_seq_segs, obs_label_segs, target_label, target_feats = data   
        for i3d_seq, obs_label in zip(i3d_seq_segs, obs_label_segs): 
            seg_feat = torch.stack(i3d_seq)
            i3d_feat_seq.append(seg_feat)
            obs_label_seq.append([int(obs_label)])
        i3d_feat_seq = torch.stack(i3d_feat_seq).float().squeeze(0).to(device)
        # print(i3d_feat_seq.shape)
        
        target_action = target_label
        pred_actions, pred_feat = model_ft(i3d_feat_seq)
        
        actual_pred_seq.append(torch.argmax(pred_actions[-1,:]).item())
        actual_target_seq.append(target_action)
        
    print(classification_report(actual_target_seq, actual_pred_seq, digits=4))
