import pandas as pd
import re
import random
from random import shuffle
import sys

from joblib import dump, load 
import numpy as np
from utils import *
from tqdm import tqdm
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy


feat_direc = sys.argv[1]
feat_dim = int(sys.argv[2])
frame_rate = int(sys.argv[3]) 
# subset = sys.argv[3]
fout = open(feat_direc+'/probing.log', 'w')
cuda = False


ali = pd.read_csv('LibriSpeech/dev-clean.ali', delimiter=' ')
ali['spk_id'] = list(map(lambda x: re.match('([0-9]+)-*',x).group(1), ali.utt_id.values))
ali['utt_only'] = list(map(lambda x: re.search('-([0-9]+)-*',x).group(1), ali.utt_id.values))

only_ph = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH','IY','OW', 'OY', 'UH', 'UW', 'M', 'N', 'NG', 'R', 'L', 'Y', 'W', 'P', 'B', 'T', 'D', 'K', 'G', 'JH',  'HH', 'F', 'V', 'S', 'Z', 'DH', 'SH', 'CH', 'ZH', 'TH']
clean_spk = ['1272', '1462', '1673', '174', '1919', '1988', '1993', '2035', '2078', '2086', '2277', '2412', '2428', '251', '2803', '2902', '3000', '3081', '3170', '3536', '3576', '3752', '3853', '422', '5338', '5536', '5694', '5895', '6241', '6295', '6313', '6319', '6345', '652', '777', '7850', '7976', '8297', '84', '8842']
other_spk = ['116', '1255', '1585', '1630', '1650', '1651', '1686', '1701', '2506', '3660', '3663', '3915', '4153', '4323', '4515', '4570', '4572', '4831', '5543', '5849', '6123', '6267', '6455', '6467', '6599', '6841', '700', '7601', '7641', '7697', '8173', '8254', '8288']
test_spk = ['1188', '260', '5142', '1995', '4970', '1221', '121', '1320', '61', '7127', '7176', '1580', '2830', '7729', '1089', '1284', '2300', '3729', '8230', '6829', '3570', '5639', '237', '8224', '4992', '5683', '8463', '4507', '672', '2094', '6930', '908', '5105', '7021', '2961', '3575', '4077', '8455', '4446', '8555']

# if subset == 'dev-clean':
#     spk_list = clean_spk
# elif subset == 'dev-other':
#     spk_list = other_spk
spk_list = clean_spk
    
train_utts = load('probing_split/train_utts')
test_utts = load('probing_split/test_utts')
# train_utts = load('sterile_split/dev-other/train_utts')
# test_utts = load('sterile_split/dev-other/test_utts')
# train_utts = load('probing_exp_split/test-clean/train_utts')
# test_utts = load('probing_exp_split/test-clean/test_utts')


def load_utt_set(utt_list,frame_rate=100):
    '''
    return features, speaker, phone labels
    '''
    utt_feat = []
    utt_speaker = []
    utt_label = []
    for utt_id in utt_list:
        spk, utt_loc = spk_utt_loc(utt_id)
        feat = np.load(feat_direc + '/' + utt_loc)
        phone_labels = create_phone_labels(utt_id, len(feat), ali, frame_rate)
        utt_feat.extend(list(feat))
        utt_speaker.extend([spk]*len(phone_labels))
        utt_label.extend(phone_labels)
    return utt_feat, utt_speaker, utt_label

def remove_sil(utt_feat, utt_speaker, utt_label):
    new_feat = []
    new_speaker = []
    new_label = []
    for i, phone in enumerate(utt_label):
        if phone not in ['SIL', 'SPN']:
            new_feat.append(utt_feat[i])
            new_speaker.append(utt_speaker[i])
            new_label.append(utt_label[i])
    return new_feat, new_speaker, new_label

def shuffle_set(utt_feat, utt_speaker, utt_label):
    utt_id = np.arange(len(utt_label))
    shuffle(utt_id)
    utt_feat = [utt_feat[i] for i in utt_id]
    utt_speaker = [utt_speaker[i] for i in utt_id]
    utt_label = [utt_label[i] for i in utt_id]
    return utt_feat, utt_speaker, utt_label

class LibriSpeechDataset(Dataset):
    def __init__(self, utt_ids, spk_set, ph_set, frame_rate=100):
        self.spk_dict = {spk: i for i, spk in enumerate(sorted(spk_set))}
        self.ph_dict = {phone: i for i,phone in enumerate(sorted(ph_set))} 
        self.ph_dict['SIL'] = -1
        self.ph_dict['SPN'] = -1
        
        # self.init_items(utt_ids)
        self.load_utt_set(utt_ids, frame_rate)
        self.remove_sil()
        
    def __len__(self):
        return len(self.phone) 
    
    def __getitem__(self, idx):
        return self.feat[idx], self.spk[idx], self.label[idx]
        
#     def __getitem__(self, idx):
#         utt_id = self.utt_ids[idx]
#         utt_offset = self.utt_offset[idx]
#         utt_spk = self.spk[idx]
        
#         spk, utt_loc = self.spk_utt_loc(utt_id)
#         assert self.spk_dict[spk] == utt_spk
#         utt_feat = np.load(feat_direc + '/' + utt_loc)
#         frame_feat = list(utt_feat)[utt_offset]
#         return frame_feat, self.spk[idx], self.label[idx]
        
    def init_items(self, utt_list, frame_rate=100):
        utt_ids = []
        utt_offset = []
        utt_speaker = []
        utt_label = []
        utt_phone = []
        for utt_id in tqdm(utt_list):
            spk, utt_loc = self.spk_utt_loc(utt_id)
            feat = np.load(feat_direc + '/' + utt_loc)
            phone_labels = create_phone_labels(utt_id, len(feat), ali, frame_rate)
            utt_ids.extend([utt_id]*len(feat))
            utt_offset.extend(list(np.arange(len(feat))))
            utt_speaker.extend([self.spk_dict[spk]]*len(phone_labels))
            utt_phone.extend(phone_labels)
            utt_label.extend([self.ph_dict[ph] for ph in phone_labels])
        self.spk = utt_speaker
        self.phone = utt_phone
        self.label= utt_label
        self.utt_ids = utt_ids
        self.utt_offset = utt_offset
            
    def spk_utt_loc(self, spk_utt_id):
        spk, utt_group = re.match('(\d+)-(\d+)-\d+',spk_utt_id).group(1,2)
        return spk, '%s/%s/%s.npy' % (spk, utt_group, spk_utt_id)
                              
        
    def load_utt_set(self, utt_list, frame_rate=100):
        '''
        return features, speaker, phone labels
        '''
        utt_feat = []
        utt_speaker = []
        utt_label = []
        utt_phone = []
        for utt_id in utt_list:
            spk, utt_loc = self.spk_utt_loc(utt_id)
            feat = np.load(feat_direc + '/' + utt_loc)
            phone_labels = create_phone_labels(utt_id, len(feat), ali, frame_rate)
            utt_feat.extend(list(feat))
            utt_speaker.extend([self.spk_dict[spk]]*len(phone_labels))
            utt_phone.extend(phone_labels)
            utt_label.extend([self.ph_dict[ph] for ph in phone_labels])
        self.feat = utt_feat
        self.spk = utt_speaker
        self.phone = utt_phone
        self.label= utt_label

    def remove_sil(self):
        # utt_ids = []
        # utt_offset = []
        new_feat = []
        new_speaker = []
        new_label = []
        new_phone = []
        for i, ph in enumerate(self.phone):
            if ph not in ['SIL', 'SPN']:
                # utt_ids.append(self.utt_ids[i])
                # utt_offset.append(self.utt_offset[i])
                new_feat.append(self.feat[i])
                new_speaker.append(self.spk[i])
                new_label.append(self.label[i])
                new_phone.append(self.phone[i])
        self.spk = new_speaker 
        self.label = new_label
        self.phone = new_phone
        self.feat = new_feat
        # self.utt_ids = utt_ids
        # self.utt_offset = utt_offset
        
    def fewer_categories(self, ph_subset):
        new_feat = []
        new_speaker = []
        new_label = []
        new_phone = []
        for i, phone in enumerate(self.phone):
            if phone in ph_subset:
                new_feat.append(self.feat[i])
                new_speaker.append(self.spk[i])
                new_label.append(self.label[i])
                new_phone.append(self.phone[i])
        self.feat = new_feat
        self.spk = new_speaker 
        self.label = new_label
        self.phone = new_phone

    def shuffle_set(self, utt_feat, utt_speaker, utt_label):
        utt_id = np.arange(len(utt_label))
        shuffle(utt_id)
        utt_feat = [utt_feat[i] for i in utt_id]
        utt_speaker = [utt_speaker[i] for i in utt_id]
        utt_label = [utt_label[i] for i in utt_id]
        return utt_feat, utt_speaker, utt_label
        
    
    
class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearClassifier, self).__init__()
        self.layer = torch.nn.Linear(input_dim, output_dim)
        # self.prob = nn.Softmax(dim=0)
        
    def forward(self, input):
        out = self.layer(input.float())
        # out = self.prob(out)
        return out
    
fout.write('loading training utterances\n')
print('loading training utterances')
train_set = LibriSpeechDataset(train_utts, spk_list, only_ph, frame_rate)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)
fout.write('loading test utterances\n')
print('loading test utterances')
test_set = LibriSpeechDataset(test_utts, spk_list, only_ph, frame_rate)
# test_feat, test_spk, test_phone = next(iter(torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=True)))
# test_feat, test_spk, test_phone = next(iter(torch.utils.data.DataLoader(train_set, batch_size=1000, shuffle=True)))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set))

fout.write("start training speaker probe\n")
print("start training speaker probe")
speaker_probe = LinearClassifier(feat_dim, 40)
if cuda: 
    speaker_probe = speaker_probe.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(speaker_probe.parameters(), lr=0.001)
speaker_probe.eval()
test_feat, test_spk, test_phone = next(iter(test_loader))
if cuda:
    test_feat, test_spk, test_phone = test_feat.cuda(), test_spk.cuda(), test_phone.cuda()
outputs = speaker_probe(test_feat)
_, predicted = torch.max(outputs, 1)
error = Counter((predicted == test_spk).cpu().numpy())[False]
fout.write('initial error rate: %.3f\n'%(error*100/len(test_spk)))
print('initial error rate: %.3f\n'%(error*100/len(test_spk)))

# prev_loss = -1
# running_loss = 0 
# epoch = 0
for epoch in range(10):  
# while prev_loss < 0 or running_loss < prev_loss:
    # prev_loss = running_loss
    speaker_probe.train()
    fout.write('epoch %d'%epoch)
    print('epoch %d'%epoch)
    running_loss = 0.0
    for i, (feat, spk, phone) in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        if cuda:
            feat, spk, phone = feat.cuda(), spk.cuda(), phone.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = speaker_probe(feat)
        loss = criterion(outputs, spk)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(speaker_probe.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()
        
    speaker_probe.eval()
    error = 0
    for test_feat, test_spk, test_phone in test_loader:
        if cuda:
            test_feat, test_spk, test_phone = test_feat.cuda(), test_spk.cuda(), test_phone.cuda()
        outputs = speaker_probe(test_feat)
        _, predicted = torch.max(outputs, 1)
        error += Counter((predicted == test_spk).cpu().numpy())[False]
    fout.write('error rate: %.3f'%(error*100/len(test_set)))
    # fout.write(f',  loss: {running_loss/(1+i):.3f}\n')
    fout.write(f',  loss: {running_loss/len(train_set):.3f}\n')
    print('error rate: %.3f'%(error*100/len(test_set)))
    # print(f',  loss: {running_loss/(1+i):.3f}\n')
    print(f',  loss: {running_loss/len(train_set):.3f}\n')
    # epoch += 1

fout.write('Finished Training\n')
print('Finished Training')

train_loader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True)

fout.write("start training phone probe\n")
print('start training phone probe')
phone_probe = LinearClassifier(feat_dim, 39)
if cuda: 
    phone_probe = phone_probe.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(phone_probe.parameters(), lr=0.0001)
phone_probe.eval()
test_feat, test_spk, test_phone = next(iter(test_loader))
if cuda:
    test_feat, test_spk, test_phone = test_feat.cuda(), test_spk.cuda(), test_phone.cuda()
outputs = phone_probe(test_feat)
_, predicted = torch.max(outputs, 1)
error = Counter((predicted == test_spk).cpu().numpy())[False]
fout.write('initial error rate: %.3f\n'%(error*100/len(test_phone)))
print('initial error rate: %.3f\n'%(error*100/len(test_phone)))

# prev_loss = -1
# running_loss = 0 
# epoch = 0
for epoch in range(10):  
# while prev_loss < 0 or running_loss < prev_loss:
    # prev_loss = running_loss # loop over the dataset multiple times
    fout.write('epoch %d: \n'%epoch)
    print('epoch %d: '%epoch)
    running_loss = 0.0
    phone_probe.train()
    for i, (feat, spk, phone) in enumerate(train_loader):
        if cuda:
            feat, spk, phone = feat.cuda(), spk.cuda(), phone.cuda()
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = phone_probe(feat)
        loss = criterion(outputs,phone)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    phone_probe.eval()
    error = 0
    for test_feat, test_spk, test_phone in test_loader:
        if cuda:
            test_feat, test_spk, test_phone = test_feat.cuda(), test_spk.cuda(), test_phone.cuda()
        outputs = phone_probe(test_feat)
        _, predicted = torch.max(outputs.data, 1)
        error += Counter((predicted == test_phone).cpu().numpy())[False]
    fout.write('error rate: %.3f'%(error*100/len(test_set)))
    # fout.write(f',  loss: {running_loss/(1+i):.3f}\n')
    fout.write(f',  loss: {running_loss/len(train_set):.3f}\n')
    print('error rate: %.3f'%(error*100/len(test_set)))
    # print(f',  loss: {running_loss/(1+i):.3f}\n')
    print(f',  loss: {running_loss/len(train_set):.3f}\n')
    # epoch += 1
    
label2phone = {i: phone for i,phone in enumerate(sorted(only_ph))} 
predicted_ph = np.array([*map(label2phone.get, predicted.numpy())])
true_ph = np.array([*map(label2phone.get, test_phone.numpy())])
per = []
for ph in only_ph:
    prediction = predicted_ph[np.where(true_ph==ph)[0]]
    # if len(prediction)>0:
    #     print(ph, Counter(prediction)[ph]/len(prediction))
    # else:
    #     print(ph,0)
    per.append(Counter(prediction)[ph]/len(prediction))

dump(per, feat_direc+'/probing_per')
fout.write('Finished Training')
print('Finished training')
fout.close()