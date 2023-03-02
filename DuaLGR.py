import argparse
import os.path

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from utils import load_data, normalize_weight, cal_homo_ratio
from models import EnDecoder, DuaLGR, GNN
from evaluation import eva
from settings import get_settings
import matplotlib.pyplot as plt
from visulization import plot_loss, plot_tsne
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='acm', help='datasets: acm, dblp, texas, chameleon, acm00, acm01, acm02, acm03, acm04, acm05')
parser.add_argument('--train', type=bool, default=False, help='training mode')
parser.add_argument('--cuda_device', type=int, default=0, help='')
parser.add_argument('--use_cuda', type=bool, default=True, help='')
args = parser.parse_args()

dataset = args.dataset
train = args.train
cuda_device = args.cuda_device
use_cuda = args.use_cuda

settings = get_settings(dataset)

path = settings.path
weight_soft = settings.weight_soft
alpha = settings.alpha
quantize = settings.quantize
varepsilon = settings.varepsilon
endecoder_hidden_dim = settings.endecoder_hidden_dim
hidden_dim = settings.hidden_dim
latent_dim = settings.latent_dim
pretrain = settings.pretrain
epoch = settings.epoch
patience = settings.patience
endecoder_lr = settings.endecoder_lr
endecoder_weight_decay = settings.endecoder_weight_decay
lr = settings.lr
weight_decay = settings.weight_decay
update_interval = settings.update_interval
random_seed = settings.random_seed
torch.manual_seed(random_seed)

labels, adjs_labels, shared_feature, shared_feature_label, graph_num = load_data(dataset, path)

for v in range(graph_num):
    r = cal_homo_ratio(adjs_labels[v].cpu().numpy(), labels.cpu().numpy(), self_loop=True)
    print(r)
print('nodes: {}'.format(shared_feature_label.shape[0]))
print('features: {}'.format(shared_feature_label.shape[1]))
print('class: {}'.format(labels.max() + 1))

feat_dim = shared_feature.shape[1]
class_num = labels.max() + 1
y = labels.cpu().numpy()

endecoder = EnDecoder(feat_dim, endecoder_hidden_dim, class_num)
model = DuaLGR(feat_dim, hidden_dim, latent_dim, endecoder, class_num=class_num, num_view=graph_num)

if use_cuda:
    torch.cuda.set_device(cuda_device)
    torch.cuda.manual_seed(random_seed)
    endecoder = endecoder.cuda()
    model = model.cuda()
    adjs_labels = [adj_labels.cuda() for adj_labels in adjs_labels]
    shared_feature = shared_feature.cuda()
    shared_feature_label = shared_feature_label.cuda()
device = shared_feature.device

if train:
    # =============================================== pretrain endecoder ============================
    print('shared_feature_label for clustering...')
    kmeans = KMeans(n_clusters=class_num, n_init=5)
    y_pred = kmeans.fit_predict(shared_feature_label.data.cpu().numpy())
    eva(y, y_pred, 'Kz')
    print()

    optimizer_endecoder = Adam(endecoder.parameters(), lr=endecoder_lr, weight_decay=endecoder_weight_decay)

    for epoch_num in range(pretrain):
        endecoder.train()
        loss_re = 0.
        loss_a = 0.

        a_pred, x_pred, z_norm = endecoder(shared_feature)
        for v in range(graph_num):
            loss_a += F.binary_cross_entropy(a_pred, adjs_labels[v])
        loss_re += F.binary_cross_entropy(x_pred, shared_feature_label)

        loss = loss_re + loss_a
        optimizer_endecoder.zero_grad()
        loss.backward()
        optimizer_endecoder.step()
        print('epoch: {}, loss:{}, loss_re:{}, loss_a: {}'.format(epoch_num, loss, loss_re, loss_a))

        if epoch_num == pretrain - 1:
            print('Pretrain complete...')
            kmeans = KMeans(n_clusters=class_num, n_init=5)
            y_pred = kmeans.fit_predict(z_norm.data.cpu().numpy())
            eva(y, y_pred, 'Kz')
            break


    # =========================================Train=============================================================
    print('Begain trains...')
    param_all = []
    for v in range(graph_num+1):
        param_all.append({'params': model.cluster_layer[v]})
    param_all.append({'params': model.gnn.parameters()})
    optimizer_model = Adam(param_all, lr=lr, weight_decay=weight_decay)

    best_a = [1e-12 for i in range(graph_num)]
    weights = normalize_weight(best_a)

    with torch.no_grad():
        model.eval()
        pseudo_label = y_pred
        a_pred, x_pred, z_all, q_all, a_pred_x, x_pred_x = model(shared_feature, adjs_labels, weights, pseudo_label, alpha, quantize=quantize, varepsilon=varepsilon)
        kmeans = KMeans(n_clusters=class_num, n_init=5)
        for v in range(graph_num+1):
            y_pred = kmeans.fit_predict(z_all[v].data.cpu().numpy())
            model.cluster_layer[v].data = torch.tensor(kmeans.cluster_centers_).to(device)
            # eva(y, y_pred, 'K{}'.format(v))
        pseudo_label = y_pred

    bad_count = 0
    best_acc = 1e-12
    best_nmi = 1e-12
    best_ari = 1e-12
    best_f1 = 1e-12
    best_epoch = 0

    nmi_list = []
    acc_list = []
    loss_list = []
    for epoch_num in range(epoch):
        model.train()

        loss_re = 0.
        loss_kl = 0.
        loss_re_a = 0.
        loss_re_ax = 0.

        a_pred, x_pred, z_all, q_all, a_pred_x, x_pred_x = model(shared_feature, adjs_labels, weights, pseudo_label, alpha, quantize=quantize, varepsilon=varepsilon)
        for v in range(graph_num):
            loss_re_a += F.binary_cross_entropy(a_pred, adjs_labels[v])
        loss_re_x = F.binary_cross_entropy(x_pred, shared_feature_label)
        loss_re += loss_re_a + loss_re_x

        kmeans = KMeans(n_clusters=class_num, n_init=5)
        y_prim = kmeans.fit_predict(z_all[-1].detach().cpu().numpy())
        pseudo_label = y_prim

        for v in range(graph_num):
            y_pred = kmeans.fit_predict(z_all[v].detach().cpu().numpy())
            a = eva(y_prim, y_pred, visible=False, metrics='nmi')
            best_a[v] = a

        weights = normalize_weight(best_a, p=weight_soft)
        # print(weights)


        p = model.target_distribution(q_all[-1])
        for v in range(graph_num):
            loss_kl += F.kl_div(q_all[v].log(), p, reduction='batchmean')
        loss_kl += F.kl_div(q_all[-1].log(), p, reduction='batchmean')

        loss = loss_re + loss_kl
        loss_list.append(loss.item())
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        print('epoch: {}, loss: {}, loss_re: {}, loss_kl:{}, badcount: {}, loss_re_a: {}, loss_re_x: {}'. format(epoch_num, loss, loss_re, loss_kl, bad_count, loss_re_a, loss_re_x))

    # =========================================evaluation=============================================================
        if epoch_num % update_interval == 0:
            model.eval()
            with torch.no_grad():
                a_pred, x_pred, z_all, q_all, a_pred_x, x_pred_x = model(shared_feature, adjs_labels, weights, pseudo_label, alpha, quantize=quantize, varepsilon=varepsilon)
                kmeans = KMeans(n_clusters=class_num, n_init=5)
                y_eval = kmeans.fit_predict(z_all[-1].detach().cpu().numpy())
                nmi, acc, ari, f1 = eva(y, y_eval, str(epoch_num) + 'Kz')
                nmi_list.append(nmi.item())
                acc_list.append(acc.item())

        if acc > best_acc:
            if os.path.exists('./pkl/dualgr_{}_acc{:.4f}.pkl'.format(dataset, best_acc)):
                os.remove('./pkl/dualgr_{}_acc{:.4f}.pkl'.format(dataset, best_acc))
            best_acc = acc
            best_nmi = nmi
            best_ari = ari
            best_f1 = f1
            best_epoch = epoch_num
            bad_count = 0
            torch.save({'state_dict': model.state_dict(),
                        'state_dict_endecoder': endecoder.state_dict(),
                        'weights': weights,
                        'pseudo_label': pseudo_label},
                       './pkl/dualgr_{}_acc{:.4f}.pkl'.format(dataset, best_acc))
            print('best acc:{}, best nmi:{}, best ari:{}, best f1:{}, bestepoch:{}'.format(
                                         best_acc, best_nmi, best_ari, best_f1, best_epoch))
        else:
            bad_count += 1

        if bad_count >= patience:
            print('complete training, best acc:{}, best nmi:{}, best ari:{}, best f1:{}, bestepoch:{}'.format(
                best_acc, best_nmi, best_ari, best_f1, best_epoch))
            print()
            break

if not train:
    model_name = settings.model_name
else:
    model_name = 'dualgr_{}_acc{:.4f}'.format(dataset, best_acc)

best_model = torch.load('./pkl/'+model_name+'.pkl', map_location=shared_feature.device)
state_dic = best_model['state_dict']
state_dic_encoder = best_model['state_dict_endecoder']
weights = best_model['weights']
pseudo_label = best_model['pseudo_label']

endecoder.load_state_dict(state_dic_encoder)
model.load_state_dict(state_dic)

model.eval()
with torch.no_grad():
    model.endecoder = endecoder
    a_pred, x_pred, z_all, q_all, a_pred_x, x_pred_x = model(shared_feature, adjs_labels, weights, pseudo_label, alpha,quantize=quantize, varepsilon=varepsilon)
    kmeans = KMeans(n_clusters=class_num, n_init=5)
    y_eval = kmeans.fit_predict(z_all[-1].detach().cpu().numpy())
    nmi, acc, ari, f1 = eva(y, y_eval, 'Final Kz')

print('Test complete...')

