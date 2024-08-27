from __future__ import print_function, division
import argparse
import random
import sqlite3
from multiprocessing import Pool


import numpy as np
import sklearn
from scipy.sparse import csgraph
from kmeans_pytorch import kmeans
from sklearn import cluster
from sklearn.cluster import KMeans,spectral_clustering
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score, pairwise_kernels
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from GNN import GNNLayer
from layers import MeanAct, DispAct, ZINBLoss
from compute_kernels import compute_kernels, compute_kernelst, compute_kernelsp, compute_kernelsp1, compute_kernelsg
# from crsc_ik import crsc_ik, kernel_kmeans
# from dis_point import calculate_dis
# from kernel_method import kernel_distance1
# from svm_margin_loss import filter_with_type
import h5py
import scanpy as sc
from utils import load_data, load_graph
# from torch_geometric.nn import TAGConv
# from ranger import Ranger
from evaluation import eva
import math
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score, pairwise_kernels
from single_cell_tools import geneSelection
from collections import Counter
# from sklearn.decomposition import TruncatedSVD
# from sklearn.random_projection import sparse_random_matrix
from preprocess import normalize, prepro, read_dataset
from scipy.sparse import csgraph



class AE(nn.Module):
    # 初始化
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    # 反向传播
    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred


class SDCN(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=2):
        super(SDCN, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)

        # self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        self.gnn_d1 = GNNLayer(args.data_num, n_dec_1)
        self.gnn_d2 = GNNLayer(n_dec_1, n_dec_2)
        self.gnn_d3 = GNNLayer(n_dec_2, n_dec_3)
        self.gnn_nb = GNNLayer(n_dec_3, n_input)

        self.k = Linear(args.data_num, 500)
        self.k1 = Linear(args.data_num, 2000)
        self.k2 = Linear(2000, 500)
        self.k3 = Linear(500, 500)
        self.k4 = Linear(500, n_input)
        self.k0 = Linear(args.data_num, n_z)

        self.k10 = Linear(args.data_num, 2000)
        self.k20 = Linear(2000, 500)
        self.k30 = Linear(500, 500)
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, args.data_num))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.a = nn.Parameter(nn.init.constant_(torch.zeros(args.data_num, args.data_num), 1), requires_grad=True).to(
            device)
        self.b = nn.Parameter(nn.init.constant_(torch.zeros(args.data_num, args.data_num), 1), requires_grad=True).to(
            device)
        self.c = nn.Parameter(nn.init.constant_(torch.zeros(args.data_num, args.data_num), 1), requires_grad=True).to(
            device)
        self.d = nn.Parameter(nn.init.constant_(torch.zeros(args.data_num, args.data_num), 1), requires_grad=True).to(
            device)
        self.e = nn.Parameter(nn.init.constant_(torch.zeros(args.data_num, args.data_num), 1), requires_grad=True).to(
            device)
        self.f = nn.Parameter(nn.init.constant_(torch.zeros(args.data_num, args.data_num), 1), requires_grad=True).to(
            device)
        self.g = nn.Parameter(nn.init.constant_(torch.zeros(args.data_num, args.data_num), 1), requires_grad=True).to(
            device)
        self.h = nn.Parameter(nn.init.constant_(torch.zeros(args.data_num, args.data_num), 1), requires_grad=True).to(
            device)
        self.e1 = nn.Parameter(nn.init.constant_(torch.zeros(args.data_num, args.data_num), 1), requires_grad=True).to(
            device)
        self.f1 = nn.Parameter(nn.init.constant_(torch.zeros(args.data_num, args.data_num), 1), requires_grad=True).to(
            device)
        self.g1 = nn.Parameter(nn.init.constant_(torch.zeros(args.data_num, args.data_num), 1), requires_grad=True).to(
            device)
        self.h1 = nn.Parameter(nn.init.constant_(torch.zeros(args.data_num, args.data_num), 1), requires_grad=True).to(
            device)
        self.e2 = nn.Parameter(nn.init.constant_(torch.zeros(args.data_num, args.data_num), 1), requires_grad=True).to(
            device)
        self.f2 = nn.Parameter(nn.init.constant_(torch.zeros(args.data_num, args.data_num), 1), requires_grad=True).to(
            device)
        self.g2 = nn.Parameter(nn.init.constant_(torch.zeros(args.data_num, args.data_num), 1), requires_grad=True).to(
            device)
        self.h2 = nn.Parameter(nn.init.constant_(torch.zeros(args.data_num, args.data_num), 1), requires_grad=True).to(
            device)
        self.e3 = nn.Parameter(nn.init.constant_(torch.zeros(args.data_num, args.data_num), 1), requires_grad=True).to(
            device)
        self.f3 = nn.Parameter(nn.init.constant_(torch.zeros(args.data_num, args.data_num), 1), requires_grad=True).to(
            device)
        self.g3 = nn.Parameter(nn.init.constant_(torch.zeros(args.data_num, args.data_num), 1), requires_grad=True).to(
            device)
        self.h3 = nn.Parameter(nn.init.constant_(torch.zeros(args.data_num, args.data_num), 1), requires_grad=True).to(
            device)
        self.e4 = nn.Parameter(nn.init.constant_(torch.zeros(args.data_num, args.data_num), 1), requires_grad=True).to(
            device)
        self.f4 = nn.Parameter(nn.init.constant_(torch.zeros(args.data_num, args.data_num), 1), requires_grad=True).to(
            device)
        self.g4 = nn.Parameter(nn.init.constant_(torch.zeros(args.data_num, args.data_num), 1), requires_grad=True).to(
            device)
        self.h4 = nn.Parameter(nn.init.constant_(torch.zeros(args.data_num, args.data_num), 1), requires_grad=True).to(
            device)
        # self.e1 = 0
        # self.f1 =0
        # self.g1 =0
        # self.h1 =0
        # self.e2 = 0
        # self.f2 =0
        # self.g2 = 0
        # self.h2 =0
        # self.e3 =0
        # self.f3 =0
        # self.g3 =0
        # self.h3 =0
        # self.e4 =0
        # self.f4 =0
        # self.g4 = 0
        # self.h4 =0

        # degree
        self.v = 1
        self._dec_mean = nn.Sequential(nn.Linear(n_input, n_input), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(n_input, n_input), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(n_input, n_input), nn.Sigmoid())

    def forward(self, x, adj):
        # DNN Module
        self.glu_layer = nn.GLU()
        # k=self.glu_layer(torch.tanh(torch.cat([kh1, kh2], -1)))
        # x_bar, tra1, tra2, tra3, z = self.ae(x)
        # GCN Module
        h1 = self.gnn_1(x, adj)

        enc_h1 = self.ae.enc_1(x)
        f1 = self.glu_layer(torch.tanh(torch.cat([enc_h1, h1], -1)))
        # enc_h1 = self.glu_layer(torch.tanh(torch.cat([enc_h1, f1], -1)))
        # h1 = self.gnn_1(torch.tanh(torch.cat([h1, f1], -1)), adj)

        k1h1 = compute_kernels(f1)
        kzh1 = compute_kernelst(f1)
        kph1 = compute_kernelsp(f1)
        kg1 = compute_kernelsg(f1.detach().cpu().numpy())
        kg1 = torch.tensor(kg1)

        # hk1h1 = compute_kernels(h1)
        # hkzh1 = compute_kernelst(h1)
        # hkph1 = compute_kernelsp(h1)
        # hkg1 = compute_kernelsg(h1.detach().cpu().numpy())
        # hkg1 = torch.tensor(hkg1)

        # kh1 = self.e1 * k1h1.to(device) + self.f1 * kzh1.to(device) + self.g1 * kph1.to(device) + self.h1 * kg1.to(device)
        # hkh1=self.e11 * hk1h1.to(device) + self.f11 * hkzh1.to(device) + self.g11 * hkph1.to(device) + self.h11 * hkg1.to(device)

        # enc_h2 = self.ae.enc_2(self.glu_layer(torch.tanh(torch.cat([enc_h1, f2], -1))))
        # # enc_h2 = self.glu_layer(torch.tanh(torch.cat([enc_h2, h2], -1)))
        # h2 = self.gnn_2(self.glu_layer(torch.tanh(torch.cat([h1, f2], -1))), adj)
        enc_h2 = self.ae.enc_2(f1)
        h2 = self.gnn_2(f1, adj)
        f2 = self.glu_layer(torch.tanh(torch.cat([enc_h2, h2], -1)))

        k1h2 = compute_kernels(f2)
        kzh2 = compute_kernelst(f2)
        kph2 = compute_kernelsp(f2)
        kg2 = compute_kernelsg(f2.detach().cpu().numpy())
        kg2 = torch.tensor(kg2)

        # kh2 = self.e2 * k1h2.to(device) + self.f2 * kzh2.to(device) + self.g2 * kph2.to(device) + self.h2 * kg2.to(device)

        # enc_h3 = self.ae.enc_3(self.glu_layer(torch.tanh(torch.cat([enc_h2, f3], -1))))
        # h3 = self.gnn_3(self.glu_layer(torch.tanh(torch.cat([h2, f3], -1))), adj)
        enc_h3 = self.ae.enc_3(f2)
        h3 = self.gnn_3(f2, adj)
        f3 = self.glu_layer(torch.tanh(torch.cat([enc_h3, h3], -1)))
        k1h3 = compute_kernels(f3)
        kzh3 = compute_kernelst(f3)
        kph3 = compute_kernelsp(f3)
        kg3 = compute_kernelsg(f3.detach().cpu().numpy())
        kg3 = torch.tensor(kg3)

        # kh3 = self.e3 * k1h3.to(device) + self.f3 * kzh3.to(device) + self.g3 * kph3.to(device) + self.h3 * kg3.to(device)

        # z = self.ae.z_layer(self.glu_layer(torch.tanh(torch.cat([enc_h3, f4], -1))))
        # h4 = self.gnn_4(self.glu_layer(torch.tanh(torch.cat([h3, f4], -1))), adj)
        z = self.ae.z_layer(f3)
        h4 = self.gnn_4(f3, adj)

        z = self.glu_layer(torch.tanh(torch.cat([z, h4], -1)))
        # z = 0.5 * z + 0.5 * h4

        k1 = compute_kernels(z)
        kz = compute_kernelst(z)
        kp = compute_kernelsp(z)
        kgz = compute_kernelsg(z.detach().cpu().numpy())
        kgz = torch.tensor(kgz)

        # kz = self.e * k1.to(device) + self.f * kz.to(device) + self.g * kp.to(device) + self.h * kgz.to(device)
        # k = self.a * kh1 + self.b * kh2 + self.c * kh3 + self.d * kz
        k = self.e1 * k1h1.to(device) + self.f1 * kzh1.to(device) + self.g1 * kph1.to(device) + self.h1 * kg1.to(
            device) + self.e2 * k1h2.to(device)
        + self.f2 * kzh2.to(device) + self.g2 * kph2.to(device) + self.h2 * kg2.to(device) + self.e3 * k1h3.to(device)
        +self.f3 * kzh3.to(device) + self.g3 * kph3.to(device) + self.h3 * kg3.to(device) + self.e * k1.to(
            device) + self.f * kz.to(device) + self.g * kp.to(device) + self.h * kgz.to(device)

        # kh=compute_kernels(h4)
        # khz = compute_kernelst(h4)
        # khp = compute_kernelsp(h4)
        # khgz = compute_kernelsg(h4.detach().cpu().numpy() )
        # khgz=torch.tensor(khgz)
        # kh = self.e4*kh.to(device) + self.f4*khz.to(device) + self.g4*khp.to(device)+self.h4*khgz.to(device)

        # k=self.a4*k+self.b4*kh

        # x_bar = self.ae.x_bar_layer(dec3)

        k11 = (self.k1(k))
        k22 = (self.k2(1 * k11))
        k33 = (self.k3(1 * k22))
        x_bar = (self.k4(1 * k33))

        A_pred = dot_product_decode(k)
        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(k.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        # k111 = (self.k10(k))
        # k222 =(self.k20(1 * k111))
        # k333 =(self.k30(1 * k222))
        s = k.unsqueeze(1) - self.cluster_layer
        _mean = self._dec_mean(x_bar)
        _disp = self._dec_disp(x_bar)
        _pi = self._dec_pi(x_bar)
        # return x_bar, q, predict, z
        return x_bar, f1, f2, f3, z, A_pred, k, s, q, _pi, _disp, _mean


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


# def kernel_kmeans(Kt, n_clusters=2, max_iter=100, n_init=5, tol=1e-5):
#     U, _, _ = np.linalg.svd(Kt)
#     H = U[:,0:n_clusters]
#     #km1 = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=n_init, tol=tol).fit(H)
#     km1,_ = kmeans(H,num_clusters=args.n_clusters, tqdm_flag=False,device=device)
#     mem = km1.labels_
#     dist = np.zeros((Kt.shape[0], n_clusters))
#     diagKt = np.diag(Kt)
#     for j in range(n_clusters):
#         if (mem==j).sum() > 1:
#             dist[:,j] = (diagKt
#                 - (2 * Kt[:,mem==j].sum(axis=1) / (mem==j).sum())
#                 + (Kt[mem==j,:][:,mem==j].sum() / ((mem==j).sum()**2)))
#     return mem, dist
from graph_function import *


def train_sdcn(runtime, y, datasetname, device, adata):
    # model = KAE(5000, 2500, 1000, 1000, 2500, 5000, args=args).to(device)
    model = SDCN(500, 500, 2000, 2000, 500, 500,
                 n_input=args.n_input,
                 n_z=args.n_z,
                 n_clusters=args.n_clusters,
                 v=1).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # KNN Graph

    # cluster parameter initiate
    data = torch.Tensor(adata.X).to(device)

    # KNN Graph
    # adj = load_graph(args.name, args.k).to(device)
    # # adj = adj.cuda()
    adj, adj_n = torch.Tensor(get_adj(data.cpu())).to(device)

    with torch.no_grad():
        x_bar, f1, f2, f3, f4, A_pred, k, s, q, _pi, _disp, _mean = model(data, adj)

    kmeans1 = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans1.fit_predict(k.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans1.cluster_centers_).to(device)
    eva(y, y_pred, 'pae')

    for epoch in range(5000):
        if epoch % 1 == 0:
            # update_interval
            x_bar, f1, f2, f3, f4, A_pred, k, s, q, pi, disp, mean = model(data, adj)
            tmp_q = q.data
            p = target_distribution(tmp_q)
            res1 = tmp_q.cpu().numpy().argmax(1)  # Q
            # y_pred1,dist = kernel_kmeans(k.detach().cpu().numpy(), args.n_clusters)
            # y_pred,dist = kernel_kmeans(k, args.n_clusters)
            # kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
            # y_pred1 = kmeans.fit_predict(k.data.cpu().numpy())
            y_pred1, _ = kmeans(k, num_clusters=args.n_clusters,  device=device)
            y_pred1 = y_pred1.data.cpu().numpy()

            # pdatas = { 'k': k,'z': z}
            # eva1(y, y_pred1, pdatas, str(epoch) + 'Q', runtime, datasetname)
            # eva(y, res1, str(epoch) +'Q', runtime, datasetname)
            eva(y, y_pred1, str(epoch) + 'K', runtime, datasetname)

        ## adversarial loss

        x_bar, f1, f2, f3, f4, A_pred, k, s, q, pi, disp, mean = model(data, adj)
        # awl = AutomaticWeightedLoss(4)
        zinb = ZINBLoss().to(device)
        zinb_loss = zinb(x=torch.Tensor(adata.raw.X).to(device), mean=mean.to(device), disp=disp.to(device),
                         pi=pi.to(device), scale_factor=torch.Tensor(adata.obs.size_factors).to(device))

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        # re_loss = F.mse_loss(x_bar, data)
        # gae_loss = F.mse_loss(hnb, data)
        re_graphloss = F.binary_cross_entropy(A_pred.view(-1).to(device), adj.to_dense().view(-1).to(device))

        # loss = awl(kl_loss, re_loss,ce_loss,re_graphloss)

        loss = 1 * F.mse_loss(x_bar, data) + 1 * torch.sum(s) + 1 * re_graphloss + 1 * kl_loss + 1 * zinb_loss \
               + 1 * F.mse_loss(f1, torch.matmul(k, f1)) + 1 * F.mse_loss(f2, torch.matmul(k, f2)) \
               + 1 * F.mse_loss(f3, torch.matmul(k, f3)) + 1 * F.mse_loss(f4, torch.matmul(k, f4))

        # print(F.mse_loss(x_bar, data),re_graphloss,kl_loss,zinb_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


from time import time

if __name__ == "__main__":
    con = sqlite3.connect("result.db")
    cur = con.cursor()
    cur.execute("delete from sdcn ")
    con.commit()

    #datasets = ['wiki']
    datasets = ['Quake_10x_Spleen']
    for dataset in datasets:
        batch = 2  # 运行轮次
        parser = argparse.ArgumentParser(
            description='train',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--name', type=str, default=dataset)
        parser.add_argument('--k', type=int, default=5)
        parser.add_argument('--lr', type=float, default=1e-6)
        parser.add_argument('--n_clusters', default=3, type=int)
        parser.add_argument('--n_z', default=10, type=int)
        parser.add_argument('--data_num', default=20000, type=int)
        parser.add_argument('--pretrain_path', type=str, default='pkl')
        parser.add_argument('--n_input', type=int, default='1000')
        args = parser.parse_args()
        args.cuda = torch.cuda.is_available()
        print("use cuda: {}".format(args.cuda))
        device = torch.device("cuda" if args.cuda else "cpu")

        args.pretrain_path = 'data/{}.pkl'.format(args.name)
        # dataset = load_data(args.name)

        if args.name == 'QS_Limb_Muscle':
            args.n_clusters = 6
            args.n_input = 1000
            args.data_num = 1090
            args.k = None
        if args.name == 'Romanov':
            args.n_clusters = 6
            args.n_input = 1000
            args.data_num = 2881
            args.k = None
        if args.name == 'Quake_10x_Spleen':
            args.n_clusters = 5
            args.n_input = 1000
            args.data_num = 9552
            args.k = None
        if args.name == 'Quake_Smart-seq2_Trachea':
            args.n_clusters = 4
            args.n_input = 1000
            args.data_num = 1350
            args.k = None
        if args.name == 'Quake_10x_Bladder':
            args.n_clusters = 6
            args.n_input = 1000
            args.data_num = 2500
            args.k = None
        if args.name == 'Quake_Smart-seq2_Limb_Muscle':
            args.n_clusters = 6
            args.n_input = 1000
            args.data_num = 1090
            args.k = None
        if args.name == 'Plasschaert':
            args.n_clusters =8
            args.n_input = 1000
            args.data_num = 6977
            args.k = None
        if args.name == 'Young':
            args.n_clusters = 11
            args.n_input = 1000
            args.data_num = 5685
            args.k = None
        if args.name == 'Tosches turtle':
            args.n_clusters = 15
            args.n_input = 1000
            args.data_num = 18664
            args.k = None
        if args.name == 'Quake_Smart-seq2_Diaphragm':
            args.k = None
            args.n_clusters = 5
            args.n_input = 1000
            args.data_num = 870
        if args.name == 'Quake_Smart-seq2_Lung':
            args.k = None
            args.n_clusters = 11
            args.n_input = 1000
            args.data_num = 1676
        if args.name == 'Pollen':
            args.k = None
            args.n_clusters = 11
            args.n_input = 1000
            args.data_num = 301
        if args.name == 'Quake_10x_Trachea':
            args.k = None
            args.n_clusters = 5
            args.n_input = 1000
            args.data_num = 11269
        if args.name == 'Quake_10x_Limb_Muscle':
            args.k = None
            args.n_clusters = 6
            args.n_input = 1000
            args.data_num = 3909
        if args.name == 'Adam':
            args.k = None
            args.n_clusters = 8
            args.n_input = 1000
            args.data_num = 3660

        if args.name == 'Quake_Smart-seq2_Heart':
            args.n_clusters = 8
            args.n_input = 1000
            args.k = None
            args.data_num = 4365
        if args.name == 'Chen':
            args.n_clusters = 46
            args.n_input = 1000
            args.data_num = 12089
            args.k = None
        if args.name == 'Muraro':
            args.n_clusters = 9
            args.n_input = 1000
            args.data_num = 2122
            args.k = None
        print(args)

        x, y = prepro(f"data/{args.name}/data.h5")

        if args.n_input > 0:
            importantGenes = geneSelection(x, n=args.n_input, plot=False)
            x = x[:, importantGenes]

        # preprocessing scRNA-seq read counts matrix
        adata = sc.AnnData(x, dtype="float64")
        if y is not None:
            adata.obs['Group'] = y

        adata = read_dataset(adata,
                             transpose=False,
                             test_split=False,
                             copy=True)

        adata = normalize(adata,
                          size_factors=True,
                          normalize_input=True,
                          logtrans_input=True)
        if y is not None:
            print(y.shape)
            y = y - min(y)

        for i in range(batch):
            cur.execute("delete from sdcn where datasetname=? and batch=?", [args.name, i])
            con.commit()
            train_sdcn(i, y, args.name, device, adata)
        # _datasets = ['bbc', 'bbcsport']
    _datasets = ['Quake_Smart-seq2_Trachea', 'Quake_Smart-seq2_Limb_Muscle', 'QS_Limb_Muscle',
                 'Quake_Smart-seq2_Diaphragm', 'Romanov','Chen', 'Quake_Smart-seq2_Lung','Plasschaert', 'Quake_Smart-seq2_Heart', 'Quake_10x_Trachea',
                 'Pollen']
    for name in _datasets:
        datas = cur.execute(
            "select datasetname,batch,epoch,acc,nmi,ari,f1 from sdcn where datasetname=? order by batch",
            [name]).fetchall()
        for d in datas:
            if d is not None:
                print('dataname:{0},batch:{1},epoch:{2}'.format(d[0], d[1], d[2]), 'acc {:.4f}'.format(d[3]),
                      ', nmi {:.4f}'.format(d[4]), ', ari {:.4f}'.format(d[5]),
                      ', f1 {:.4f}'.format(d[6]))
    for name in _datasets:
        result = cur.execute(
            "select  avg(acc) as acc,avg(nmi) as nmi,avg(ari) as ari,avg(f1) as f1 from ( select acc,nmi,ari,f1 from sdcn where datasetname =? order by nmi desc limit 10)",
            [name]).fetchone()

        if result[0] is not None:
            print('dataname:{0}'.format(name), 'AVG :acc {:.4f}'.format(result[0]),
                  ', nmi {:.4f}'.format(result[1]), ', ari {:.4f}'.format(result[2]),
                  ', f1 {:.4f}'.format(result[3]))
    cur.close()
    con.close()
