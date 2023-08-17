
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from itertools import islice
import sys
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
import csv
from itertools import islice
from torch.autograd import Variable
import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict
from rdkit import Chem
# from rdkit.Chem import MolFromSmiles
import networkx as nx

import random
import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch.utils.data as Data

import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd
from sklearn.model_selection import KFold
import gc
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp, global_mean_pool as gmean
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import roc_auc_score,precision_recall_curve,accuracy_score
from torch_geometric.nn import GATConv

import time


def metrics_graph(yt, yp):
    precision, recall, _, = precision_recall_curve(yt, yp)
    aupr = -np.trapz(precision, recall)
    auc = roc_auc_score(yt, yp)
    #---f1,acc,recall, specificity, precision
    real_score=np.mat(yt)
    predict_score=np.mat(yp)
    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]
    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN
    tpr = TP / (TP + FN)
    recall_list = tpr
    precision_list = TP / (TP + FP)
    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)
    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return auc, aupr, f1_score[0, 0], accuracy[0, 0] #, recall[0, 0], specificity[0, 0], precision[0, 0]



# GCN based model
class GCNNet(torch.nn.Module):
    def __init__(self, n_output=2, n_filters=32, embed_dim=128,num_features_xd=78, num_features_xt=954, output_dim=1, dropout=0.2):

        super(GCNNet, self).__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # SMILES1 graph branch
        self.n_output = n_output
        self.drug1_conv1 = GCNConv(num_features_xd, num_features_xd)
        self.drug1_conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.drug1_conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.drug1_fc_g1 = torch.nn.Linear(num_features_xd*4, 489)
        # self.drug1_fc_g2 = torch.nn.Linear(num_features_xd*2, output_dim)
        self.final = torch.nn.Linear(489 + num_features_xd*2, 489)
  

    def forward(self, data1, drug2):
        x1, edge_index1, batch1 = data1.x, data1.edge_index, data1.batch
        # deal drug1
        # x1 = torch.cat((x1,drug2),1)
    
        # x1 = self.dropout(self.relu(x1))

        x1 = self.drug1_conv1(x1, edge_index1)
        x1 = self.dropout(self.relu(x1))

        x1 = self.drug1_conv2(x1, edge_index1)
        x1 = self.dropout(self.relu(x1))

        x1 = self.drug1_conv3(x1, edge_index1)
        x1 = self.dropout(self.relu(x1))
        x1 = gmp(x1, batch1)       # global max pooling

        # flatten
        x1 = self.relu(self.drug1_fc_g1(x1))
        # x1 = self.dropout(x1)
        # x1 = self.drug1_fc_g2(x1)
        # x1 = self.dropout(x1)

        # f = torch.sigmoid(self.final(x1))
        # final_feature = torch.cat((x1,drug2),1)
        # f = self.final(final_feature)

        f = x1 + drug2

        # print('x1.shape', x1.shape)
        # print('x1', x1[0])
        return f

# GAT based model
class GATNet(torch.nn.Module):
    def __init__(self, n_output=2, n_filters=32, embed_dim=128,num_features_xd=78, num_features_xt=954, output_dim=1, dropout=0.4):

        super(GATNet, self).__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # SMILES1 graph branch
        self.n_output = n_output
        self.drug1_conv1 =  GATConv(num_features_xd, embed_dim, heads=6, dropout=dropout)
        self.drug1_gcn2 = GATConv(embed_dim * 6, embed_dim, dropout=dropout)
        self.drug1_fc_g1 = torch.nn.Linear(embed_dim, embed_dim)
        # self.drug1_fc_g2 = torch.nn.Linear(num_features_xd*2, output_dim)
        # self.cell_line1 = torch.nn.Linear(11794, 1024)
        # self.cell_line2 = torch.nn.Linear(1024, 128)
        self.final = torch.nn.Linear(489+embed_dim, 489)


    def forward(self, data1, drug2):
        x1, edge_index1, batch1 = data1.x, data1.edge_index, data1.batch
        # deal drug1
        x1 = self.drug1_conv1(x1, edge_index1)
        
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.4, training=self.training)

        x1 = self.drug1_gcn2(x1, edge_index1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.4, training=self.training)
        x1 = gmp(x1, batch1)       # global max pooling

        # flatten
        x1 = self.relu(self.drug1_fc_g1(x1))
        # x1 = self.drug1_fc_g2(x1)
        # x1 = self.dropout(x1)


        # cell line
        # c = self.cell_line1(cell_line)
        # c = self.relu(c)
        # c = self.dropout(c)
        # c = self.cell_line2(c)
        # c = self.relu(c)
        # c = self.dropout(c)

        # concat
        # final_feature = torch.cat((x1,drug2),1)
        final_feature = (x1 + drug2)/2
        # f = torch.sigmoid(self.final(final_feature))
        f = self.final(final_feature)


        # print('x1.shape', x1.shape)
        # print('x1', x1[0])
        return f


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))




class TestbedDataset(InMemoryDataset):
    def __init__(self, root='', dataset='drugtographcla',
                 xd=None, xt=None, y=None, xt_featrue=None, transform=None,
                 pre_transform=None, smile_graph=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def get_cell_feature(self, cellId, cell_features):
        for row in islice(cell_features, 0, None):
            if cellId in row[0]:
                return row[1:]
        return False

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, y, smile_graph):
        data_list = []
        data_len = len(xd)
        print('number of data', data_len)
        for i in range(data_len):
            # print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            print(edge_index)
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            if len(edge_index) != 0:
                GCNData = DATA.Data(x=torch.Tensor(features),
                                    edge_index= torch.LongTensor(edge_index).transpose(1, 0), 
                                    y=torch.Tensor([labels]))
            else:
                GCNData = DATA.Data(x=torch.Tensor(features),
                    edge_index= torch.LongTensor(edge_index),
                    y=torch.Tensor([labels]))

            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])


def train(model, device, drug_loader_train, optimizer, epoch, rna):
    model.train()
    
    # train_loader = np.array(train_loader)
    for batch_idx, data in enumerate(drug_loader_train):
        data1 = data
        data1 = data1.to(device)
        y = data.y.view(-1, 1).long().to(device).to(torch.float32)
        v_p = torch.tensor(rna).unsqueeze(0).repeat(y.size()[0], 1).to(device)
        # y = y.squeeze(1) 
        optimizer.zero_grad()
        output = model(data1, v_p).to(torch.float32)
        loss = loss_fn(output, y)
        # print('loss', loss)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data1.x),
                                                                           len(drug1_loader_train.dataset),
                                                                           100. * batch_idx / len(drug1_loader_train),
                                                                           loss.item()))


def predicting(model, datagenerator,rna):
        y_label = []
        y_pred = []
        model.eval()
        for  batch_idx, data in enumerate(datagenerator):
            data = data.to(device)
            y = data.y.view(-1, 1).long().to(device).to(torch.float32)
            v_p = torch.tensor(rna).unsqueeze(0).repeat(y.size()[0], 1).to(device)
            score = model(data, v_p)
            loss_fct = torch.nn.MSELoss()
            # n = torch.squeeze(score, 1)
            loss = loss_fct(score, y).to(device)
            logits = torch.squeeze(score).detach().cpu().numpy()
            label_ids = y.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()
            auc, aupr, F1, ACC =metrics_graph(y_label, y_pred)

        model.train()

        return y_label, y_pred, \
               mean_squared_error(y_label, y_pred), \
               np.sqrt(mean_squared_error(y_label, y_pred)), \
               pearsonr(y_label, y_pred)[0], \
               pearsonr(y_label, y_pred)[1], \
               spearmanr(y_label, y_pred)[0], \
               spearmanr(y_label, y_pred)[1], \
               concordance_index(y_label, y_pred), \
               loss, auc\

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(len(dataset) * ratio)
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2





# # 药物信息
# ic50 = pd.read_csv('new_data/sars_ic50.tsv', sep='\t')
# sub_df = ic50[['Molecule ChEMBL ID', 'Smiles', 'Standard Value']]
# # sub_df['Standard Value'] = (sub_df['Standard Value'] / 1000).apply(np.log)
# Binary_Drug_list = []
# for idx,row in sub_df.iterrows():
#         ic50_value = row['Standard Value']
#         if (int(ic50_value) < 20000):
#             Binary_Drug_list.append(1)
#         else:
#             Binary_Drug_list.append(0)
# sub_df['Binary_IC50'] = Binary_Drug_list

# sub_train, sub_test = train_test_split(sub_df, test_size=0.2,random_state=1)
# sub_train = sub_train.reset_index()
# sub_test = sub_test.reset_index()

# covid cellline 信息
# cell_line = pd.read_excel('new_data/Vero6.xlsx')
# rnadata =  pd.read_csv('GDSC_data/Cell_line_RMA_proc_basalExp.txt',sep='\t')
# gene2value = dict(zip(cell_line['Gene'],cell_line['use']))
# gene_name = [x for x in list(cell_line['Gene']) if x in list(rnadata['GENE_SYMBOLS']) and gene2value[x] != 0]
# cell_line = cell_line[cell_line['Gene'].isin(gene_name)] 
# cell_line.sort_values(by=['Gene'],ascending=False, inplace=True)
# cell_line['use'] = (cell_line['use'] - cell_line['use'].mean()) / cell_line['use'].std()
# cell_line_list = cell_line['use'].tolist()



# compound_iso_smiles = []
# compound_iso_smiles += list(sub_df['Smiles'])
# compound_iso_smiles = set(compound_iso_smiles)
# smile_graph = {}

# for smile in compound_iso_smiles:
#     g = smile_to_graph(smile)
#     smile_graph[smile] = g




# # drugdata = TestbedDataset( xd=sub_df['Smiles'], y=sub_df['Standard Value'], smile_graph=smile_graph)
# drugdata = TestbedDataset( xd=sub_df['Smiles'], y=sub_df['Binary_IC50'], smile_graph=smile_graph)


# # modeling = GCNNet
# modeling = GATNet

# TRAIN_BATCH_SIZE = 8
# TEST_BATCH_SIZE = 8
# LR = 0.0005
# LOG_INTERVAL = 20
# NUM_EPOCHS = 8000


# gcn_pcc = []
# gcn_scc = []
# y_pre = []
# y_score = []
# rmse_list = []
# seed = [1,2,3,4,5]
# for (i_s, seed) in enumerate(seed):
#     kf = KFold(n_splits=5, random_state=seed, shuffle=True)
#     foldnum = 0
#     gcn_pcc.append([])
#     gcn_scc.append([])
#     rmse_list.append([])
#     y_pre.append([])
#     y_score.append([])


#     for X_train,X_test in kf.split(drugdata):
#         foldnum += 1
#         model = modeling().to(device)
#         loss_fn =  torch.nn.MSELoss()
#         optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#         print('___________'+str(foldnum)+' _fold______________')
#         sub_train = drugdata[X_train.tolist()]
#         sub_test = drugdata[X_test.tolist()]
#         drug1_loader_train = DataLoader(sub_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None)
#         drug1_loader_test = DataLoader(sub_test, batch_size=TRAIN_BATCH_SIZE, shuffle=None)


#         best_auc = 0
#         max_rmse = 10000
#         max_loss= 10000
#         max_p = 0
#         max_s = 0
#         t_start = time.time()
#         for epoch in range(NUM_EPOCHS):
#             train(model, device, drug1_loader_train, optimizer, epoch + 1, cell_line_list)
#             y_true,y_pred, mse, rmse, \
#                     person, p_val, \
#                     spearman, s_p_val, CI,\
#                     loss_val,auc= predicting(model, drug1_loader_test, cell_line_list)
#             if(max_p < person):
#                 max_p = person
#                 max_s = spearman
#                 record_pre = y_pred
#                 record_y = y_true
#                 max_rmse = rmse
#                 # best_auc = auc
#                 print("person:",person," spearman:",spearman, " rmse:", rmse)
#                 t_now = time.time()
#                 # print("auc:",auc)
#                 print(". Total time " + str(int(t_now - t_start) / 3600)[:7] + " hours")
#         y_score[i_s].extend(record_pre)
#         y_pre[i_s].extend(record_y)


#         del model
#         torch.cuda.empty_cache()
#         gc.collect()
#         gcn_pcc[i_s].append(max_p)
#         gcn_scc[i_s].append(max_s)
#         rmse_list[i_s].append(max_rmse)

# np.savez('gat_ftest',gat_pcc=gcn_pcc,gat_scc=gcn_scc)
# np.savez('gat_for_auc2', y_score=y_score,y_pre=y_pre)
# np.savez('gat_for_rmse', rmse=rmse_list)
# 注意换cla要换process