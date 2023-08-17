
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot
# from torch_geometric.nn.dense.linear import Linear

from gat_layers import GraphAttentionLayer, SpGraphAttentionLayer
from torch_geometric.nn import GATConv, GCNConv, MessagePassing

from sklearn.ensemble import RandomForestClassifier
import os
from dataloader import load_info_data, load_pre_process
import argparse

import pickle
from utils import accuracy, precision, recall, specificity, mcc, auc, aupr
import numpy as np


class GraphAttentionLayer(MessagePassing):
    # def __init__(self, in_features: int, out_features: int, n_heads: int,
    #              residual: bool, dropout: float = 0.6, slope: float = 0.2, activation: nn.Module = nn.ELU()):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                              residual: bool, dropout: float = 0.5, slope: float = 0.5, activation: nn.Module = nn.ELU()):
        super(GraphAttentionLayer, self).__init__(aggr='mean', node_dim=0)
        self.in_features = in_features
        self.out_features = out_features
        self.heads = n_heads
        self.residual = residual

        self.attn_dropout = nn.Dropout(dropout)
        self.feat_dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(negative_slope=slope)
        self.activation = activation
        # [218,128,128]
        self.feat_lin = Linear(in_features, out_features * n_heads, bias=True, weight_initializer='glorot')
        self.attn_vec = nn.Parameter(torch.Tensor(1, n_heads, out_features))

        # use 'residual' parameters to instantiate residual structure
        if residual:
            self.proj_r = Linear(in_features, out_features, bias=False, weight_initializer='glorot')
        else:
            self.register_parameter('proj_r', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.attn_vec)

        self.feat_lin.reset_parameters()
        if self.proj_r is not None:
            self.proj_r.reset_parameters()
    # x是节点特征(218,271)
    def forward(self, x, edge_idx, size=None):
        # normalize input feature matrix
        x = self.feat_dropout(x)

        x_r = x_l = self.feat_lin(x).view(-1, self.heads, self.out_features)

        # calculate normal transformer components Q, K, V
        output = self.propagate(edge_index=edge_idx, x=(x_l, x_r), size=size)

        if self.proj_r is not None:
            output = (output.transpose(0, 1) + self.proj_r(x)).transpose(1, 0)

        # output = self.activation(output)
        output = output.mean(dim=1)

        return output


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, noutput, dropout, negative_slope, nheads):
        """ version of GAT."""
        super(GAT, self).__init__()
        # self.dropout = 0.6
        self.dropout = dropout
        self.attentions = GATConv(nfeat, nhid, nheads, True, negative_slope=negative_slope, dropout=self.dropout)
        self.out_att = GATConv(nhid*nheads, noutput, 1, False, negative_slope=negative_slope, dropout=self.dropout)
        self.BatchNorm = torch.nn.BatchNorm1d(num_features=noutput)
        self.LayerNorm = torch.nn.LayerNorm(noutput)

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.attentions(x, edge_index))
        x = F.relu(self.attentions(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, edge_index))
        x = F.relu(self.out_att(x, edge_index))
        x = self.BatchNorm(x)
        x = self.LayerNorm(x)
        return x

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, noutput, dropout):
        """version of GCN."""
        super(GCN, self).__init__()
        # self.dropout = 0.6
        self.dropout = dropout
        self.gcn1 = GCNConv(nfeat, nhid)

        self.gcn2 = GCNConv(nhid, noutput)
        self.BatchNorm = torch.nn.BatchNorm1d(num_features=noutput)
        self.LayerNorm = torch.nn.LayerNorm(noutput)

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gcn1(x, edge_index, edge_weight))
        # x = F.elu(self.gcn1(x, edge_index, edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gcn2(x, edge_index, edge_weight))
        # x = F.elu(self.gcn2(x, edge_index, edge_weight))
        x = self.BatchNorm(x)
        x = self.LayerNorm(x)
        return x

class NN(nn.Module):
    def __init__(self, ninput, nhidden, noutput, nlayers, dropout=0.3):
        """
        """
        super(NN, self).__init__()
        self.dropout = dropout
        self.encode = torch.nn.ModuleList([
            torch.nn.Linear(ninput if l == 0 else nhidden[l - 1], nhidden[l] if l != nlayers - 1 else noutput) for l in
            range(nlayers)])
        self.BatchNormList = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(num_features=nhidden[l] if l != nlayers-1 else noutput) for l in range(nlayers)])
        self.LayerNormList = torch.nn.ModuleList([
            torch.nn.LayerNorm(nhidden[l] if l != nlayers - 1 else noutput) for l in range(nlayers)])

    def forward(self, x):
        # x [B, 220] or [B, 881]
        for l, linear in enumerate(self.encode):
            x = F.relu(linear(x))
            x = self.BatchNormList[l](x)
            x = self.LayerNormList[l](x)
            x = F.dropout(x, self.dropout)
        return x

class DTI_Decoder(nn.Module):
    def __init__(self, Protein_num, Drug_num, Nodefeat_size, nhidden, nlayers, dropout=0.3):
        super(DTI_Decoder, self).__init__()
        self.Protein_num = Protein_num
        self.Drug_num = Drug_num
        self.dropout = dropout
        self.nlayers = nlayers
        self.decode = torch.nn.ModuleList([
            torch.nn.Linear(Nodefeat_size if l == 0 else nhidden[l - 1], nhidden[l]) for l in
            range(nlayers)])
        self.BatchNormList = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(num_features=nhidden[l]) for l in range(nlayers)])
        self.linear = torch.nn.Linear(nhidden[nlayers-1], 1)
        self.max_aoc = 0
        self.max_aupr = 0
        self.max_epoch = 0

        self.w_drug = nn.Parameter(torch.ones(2))
        self.w_rna = nn.Parameter(torch.ones(2))
        self.drug_linear1 = torch.nn.Linear(2048, 1024)
        self.drug_linear2 = torch.nn.Linear(1024, 128)

    def forward(self, epoch, CircRNAs, Rna_3, Drugs, Drugs_3, nodes_features, circRNA_index, drug_index):
        protein_features = nodes_features[circRNA_index]
        protein_features_ori = CircRNAs[circRNA_index]
        drug_features = nodes_features[drug_index]
        col_cdi_id = (drug_index - 271)
        drug_features_ori = Drugs[col_cdi_id]
        pair_nodes_features = torch.cat((protein_features, drug_features), 1) # (6612, 2*(489 + 256))


        for l, dti_nn in enumerate(self.decode):
            pair_nodes_features = F.dropout(pair_nodes_features, self.dropout)
            pair_nodes_features = F.relu(dti_nn(pair_nodes_features))
            pair_nodes_features = self.BatchNormList[l](pair_nodes_features)
        pair_nodes_features = F.dropout(pair_nodes_features, self.dropout)
        output = self.linear(pair_nodes_features)
        return torch.sigmoid(output)

class DTI_Graph(nn.Module):
    """
    Model for Drug-CircRNA interaction Graph
    pnn_hyper = [circRNA_ninput, cnn_nhid, gat_ninput, cnn_nlayers]
    dnn_hyper = [drug_ninput, dnn_nhid, gat_ninput, dnn_nlayers]
    GAT_hyper = [gat_ninput, gat_nhid, gat_noutput, gat_negative_slope, nheads]
    Deco_hyper = [gat_noutput, DTI_nn_nhid]
    GCN_hyper = [gcn_ninput, gcn_nhid, gcn_noutput, gat_negative_slope]
    """
    def __init__(self, GAT_hyper, CNN_hyper, DNN_hyper, DECO_hyper, CircRNA_num, Drug_num, dropout, smiles_gcn):
        super(DTI_Graph, self).__init__()
        self.drug_nn = NN(DNN_hyper[0], DNN_hyper[1], DNN_hyper[2], DNN_hyper[3], dropout)
        self.circRNA_nn = NN(CNN_hyper[0], CNN_hyper[1], CNN_hyper[2], CNN_hyper[3], dropout)
        self.gat = GAT(489, GAT_hyper[1], 256, dropout, GAT_hyper[3], GAT_hyper[4])
        self.DTI_Decoder = DTI_Decoder(CircRNA_num, Drug_num, DECO_hyper[0] + DECO_hyper[0] + 271 + 218, DECO_hyper[1],
                                        DECO_hyper[2], dropout)
        self.CircRNA_num = CircRNA_num
        self.Drug_num = Drug_num
        self.BatchNorm = torch.nn.BatchNorm1d(num_features=489)
        self.LayerNorm = torch.nn.LayerNorm(489)
        self.drug_linear3 = torch.nn.Linear(271, 128)
        self.drug_linear4 = torch.nn.Linear(645, 489)
        self.rna_linear1 = torch.nn.Linear(218, 128)
        self.rna_linear2 = torch.nn.Linear(218, 128)
        self.dropout = torch.nn.Dropout(0.5)

        self.w_drug = nn.Parameter(torch.ones(2))
        self.w_rna = nn.Parameter(torch.ones(2))
        self.w = nn.Parameter(torch.ones(2))
        self.gcn = GCN(489, GAT_hyper[1], GAT_hyper[2], dropout)
        self.ori_linear = torch.nn.Linear(489, 256)

        self.smiles_gcn = smiles_gcn



    def forward(self, epoch, CircRNAs, Drugs, edge_index, circRNA_index, drug_index, edge_weight,drugdata):



        # CircRNA and Drug embeding
        Drugs_1 = Drugs[:, 0:271]
        Drugs_2 = Drugs[:, 271:489]
        Drugs = Drugs[:, :489]
        Rna_1 = CircRNAs[:, 0:218]
        Rna_2 = CircRNAs[:, 218:489]
        CircRNAs = CircRNAs[:,:489]
        Drugs = self.smiles_gcn(drugdata, Drugs)

        # Drugs = self.dropout(F.relu(self.drug_linear4(Drugs)))
        Nodes_features_ori = torch.cat((CircRNAs, Drugs), 0)
        Nodes_features_ori = self.BatchNorm(Nodes_features_ori)
        Nodes_features_ori = self.LayerNorm(Nodes_features_ori)
        Nodes_features = self.gat(Nodes_features_ori, edge_index)
        # Decoder
        output = self.DTI_Decoder(epoch, CircRNAs, Rna_1, Drugs, Drugs_1, Nodes_features, circRNA_index, drug_index)
        output = output.view(-1)


        return output




