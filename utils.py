import pandas as pd
import numpy as np
import scipy
import scipy.spatial as sp
from scipy.stats import rankdata
from sklearn.decomposition import PCA
import sklearn

import matplotlib
import matplotlib.pyplot as plt
import seaborn
import umap
import umap.plot

from collections import Counter
from collections import defaultdict
import re
import os

vowels = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH','IY','OW', 'OY', 'UH', 'UW']
consonants = ['M', 'N', 'NG', 'R', 'L', 'Y', 'W', 'P', 'B', 'T', 'D', 'K', 'G', 'JH',  'HH' 'F', 'V', 'S', 'Z', 'DH', 'SH', 'CH', 'ZH', 'TH']
ph_list = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH','IY','OW', 'OY', 'UH', 'UW', 'M', 'N', 'NG', 'R', 'L', 'Y', 'W', 'P', 'B', 'T', 'D', 'K', 'G', 'JH',  'HH', 'F', 'V', 'S', 'Z', 'DH', 'SH', 'CH', 'ZH', 'TH', 'SIL', 'SPN']
two_letter = ['AA', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH','IY','OW', 'OY', 'UH', 'UW', 'NG', 'SH', 'DH', 'SIL', 'EH', 'SPN', 'AY', 'AW', 'AE', 'CH' 'ZH' 'JH', 'TH', 'HH']
only_ph = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH','IY','OW', 'OY', 'UH', 'UW', 'M', 'N', 'NG', 'R', 'L', 'Y', 'W', 'P', 'B', 'T', 'D', 'K', 'G', 'JH',  'HH', 'F', 'V', 'S', 'Z', 'DH', 'SH', 'CH', 'ZH', 'TH']
ph_dic = {k: v for k, v in enumerate(ph_list)}

clean_spk = ['1272', '3170', '2412', '1993', '2803', '3853', '251', '6345', '2078', '2035', '6319', '2428', '84', '3536', '422', '1673', '5536', '6313', '5338', '2277', '777', '8297', '8842', '5895', '6295', '7850', '2902', '6241', '3576', '1988', '3000', '2086', '5694', '1919', '3752', '3081', '1462', '652', '174', '7976']

def remove_bie(label):
    return re.sub('[0-9]','',label.split('_')[0])

def create_phone_labels(spk_ch_utt, feat_len, ali, frame_rate=100):
    selected_ali = ali[ali.utt_id == spk_ch_utt]
    phone_dur = selected_ali.phone_dur.values*frame_rate
    phone = selected_ali.phone.values
    phone = list(map(remove_bie, phone))

    phone_labels = []
    for dur, ph in zip(phone_dur, phone):
        phone_labels.extend([ph]*int(dur))
    if feat_len > len(phone_labels):
        pad_len = feat_len - len(phone_labels)
        phone_labels.extend(['SIL']*pad_len)
    elif feat_len < len(phone_labels):
        print('discrepant lengths - feature length: %d, label length: %d'%(feat_len, len(phone_labels)))

    return phone_labels

def aggregate_feat_phone(spk, ali, direc, frame_rate=100):
    feat = []
    phone = []
    for utt in set(ali[ali.spk_id == str(spk)].utt_id.values):
        utt_only = re.search('-([0-9]+)-*',utt).group(1)
        x = np.load(direc+'/%s/%s/%s.npy'%(str(spk),utt_only,utt))
        feat.append(x)
        phone.extend(create_phone_labels(utt, len(x), ali, frame_rate))
    return np.concatenate(feat, axis=0), phone

def compute_phone_centroid(x_feat, x_phone):
    x_phone_emb = defaultdict(list)
    x_phone_centroid = dict()
    for feat, phone in zip(x_feat, x_phone):
        x_phone_emb[phone].append(feat)
    phone_occurred = set(x_phone)
    for ph in ph_list:
        if ph in phone_occurred:
            x_phone_centroid[ph] = np.mean(np.array(x_phone_emb[ph]), axis=0)
        # print(x_phone_centroid[0].shape)
    return x_phone_centroid

def plot_cos_sim(cos_sim, ph_list):
    fig, ax = plt.subplots(figsize=(15, 15), dpi=80)
    im = ax.imshow(cos_sim, vmin=-0.3, vmax=1)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(ph_list)))
    ax.set_yticks(np.arange(len(ph_list))) 
    ax.set_xticklabels(ph_list)
    ax.set_yticklabels(ph_list)
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")
    im.set_cmap("Blues_r")
    plt.colorbar(im)
    plt.show()
    
def procrustes_mapping(source, target):
    R = scipy.linalg.orthogonal_procrustes(source, target)[0]
    return np.dot(source, R), R

def cos_sim(x_feat, y_feat):
    return 1 - sp.distance.cdist(x_feat, y_feat, 'cosine')

def compare_cos_sim(cs_a, cs_b, ph_list):
    a = [cs_a[i, i] for i in range(len(cs_a))]
    b = [cs_b[i, i] for i in range(len(cs_a))]
    for i, ph in enumerate(ph_list):
        print(ph, a[i], b[i])
        
def present_cos_sim(src1, tgt1, src2, tgt2, ph_list):
    mapped1, _ = procrustes_mapping(src1, tgt1)
    cs_1 = cos_sim(mapped1, tgt1)
    cs_2 = cos_sim(src2, tgt2)
    
    compare_cos_sim(cs_1, cs_2, ph_list)
    
def cos_sim_summary(cs_1, cs_2, ph_list):
    for i, ph in enumerate(ph_list):
        print(ph, ph_list[np.argmax(cs_1[i])],np.max(cs_1[i]), ph_list[np.argsort(cs_1[i])[-2]], sorted(cs_1[i])[-2], ph_list[np.argmax(cs_2[i])],np.max(cs_2[i]))

def avg_diagonal_sim(x_feat, y_feat):
    cs = cos_sim(x_feat, y_feat)
    diag_sim = [cs[i,i] for i in range(len(x_feat)-1)]
    # print(len(diag_sim))
    return np.mean(diag_sim)

def umap_emb(x_feat, y_feat):
    combined = np.concatenate([x_feat, y_feat])
    mapper = umap.UMAP().fit(combined)
    emb = umap.plot._get_embedding(mapper)
    x_len = len(x_feat)
    x_emb = emb[:x_len]
    y_emb = emb[x_len:]
    return x_emb, y_emb, mapper
        
def plot_phone_vec(x_feat, x_vector, y_feat, y_vector):
    x_emb, y_emb, mapper_n = umap_emb(x_feat, y_feat) 
    x_vector_emb = mapper_n.transform(x_vector)
    y_vector_emb = mapper_n.transform(y_vector)
    plt.figure(figsize=(15, 15), dpi=80)
    for i, ph in enumerate(ph_list):
        if ph in two_letter:
            plt.scatter(x_vector_emb[i,1], x_vector_emb[i,0], alpha=0.5, s=500, color='r', marker='$%s$'%ph)
            plt.scatter(y_vector_emb[i,1], y_vector_emb[i,0], alpha=0.5, s=500, color='g', marker='$%s$'%ph)
        else:
            plt.scatter(x_vector_emb[i,1], x_vector_emb[i,0], alpha=0.5, s=200, color='r', marker='$%s$'%ph)
            plt.scatter(y_vector_emb[i,1], y_vector_emb[i,0], alpha=0.5, s=200, color='g', marker='$%s$'%ph)

def umap_emb(x_feat, y_feat):
    combined = np.concatenate([x_feat, y_feat])
    mapper = umap.UMAP().fit(combined)
    emb = umap.plot._get_embedding(mapper)
    x_len = len(x_feat)
    x_emb = emb[:x_len]
    y_emb = emb[x_len:]
    return x_emb, y_emb, mapper

def pca_map(x_feat, y_feat, x_centroid, y_centroid, n_comp):
    pca_x = PCA(n_components=n_comp)
    pca_x.fit(x_feat)
    x_pc = pca_x.components_

    pca_y = PCA(n_components=n_comp)
    pca_y.fit(y_feat)
    y_pc = pca_y.components_
    x_n = pca_x.transform(x_centroid_n)
    y_n = pca_y.transform(y_centroid_n)
    _, R = procrustes_mapping(x_n, y_n)
    x_centroid_mapped = np.dot(x_n, R)
    cos_sim_mapped = 1 - sp.distance.cdist(x_centroid_mapped, y_n, 'cosine')
    # cos_sim = 1 - sp.distance.cdist(x_n, y_n, 'cosine')
    # present_cos_sim(x_centroid_mapped, y_n, x_n, y_n)
    # plot_cos_sim(cos_sim(x_n,y_n))
    # plot_cos_sim(cos_sim_mapped)
    diag_sim = np.mean([cos_sim_mapped[i,i] for i in range(len(ph_list)-1)])
    # print(diag_sim)
    return pca_x, pca_y, R

def shared_phones(a_phone, t_phone):
    shared = []
    for ph in ph_list:
        if ph in a_phone and ph in t_phone:
            shared.append(ph)
    return shared

def normalise_ph_vecs(x):
    return x / np.linalg.norm(x,axis=1)[:,None]