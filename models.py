import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils import cal_homo_ratio


class LatentMappingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=6):
        super(LatentMappingLayer, self).__init__()
        self.num_layers = num_layers
        self.enc = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim)
        ])
        for i in range(1, num_layers):
            if i == num_layers - 1:
                self.enc.append(nn.Linear(hidden_dim, output_dim))
            else:
                self.enc.append(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x, dropout=0.1):
        z = self.encode(x, dropout)
        return z

    def encode(self, x, dropout=0.1):
        h = x
        for i, layer in enumerate(self.enc):
            if i == self.num_layers - 1:
                if dropout:
                    h = torch.dropout(h, dropout, train=self.training)
                h = layer(h)
            else:
                if dropout:
                    h = torch.dropout(h, dropout, train=self.training)
                h = layer(h)
                h = F.tanh(h)
        return h


class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphEncoder, self).__init__()
        self.LatentMap = LatentMappingLayer(input_dim, hidden_dim, output_dim, num_layers=2)

    def forward(self, x, adj, order):
        adj = F.normalize(adj, p=2, dim=1)
        z = self.message_passing_global(x, adj, order)
        z = self.LatentMap(z, dropout=False)
        return z

    def message_passing_global(self, x, adj, order):
        h = x
        for i in range(order):
            h = torch.matmul(adj, h) + (1 * x)
        return h

    def normalize_adj(self, x):
        D = x.sum(1).detach().clone()
        r_inv = D.pow(-1).flatten()
        r_inv = r_inv.reshape((x.shape[0], -1))
        r_inv[torch.isinf(r_inv)] = 0.
        x = x * r_inv
        return x


class EnDecoder(nn.Module):
    def __init__(self, feat_dim, hidden_dim, latent_dim):
        super(EnDecoder, self).__init__()

        self.enc = LatentMappingLayer(feat_dim, hidden_dim, latent_dim, num_layers=2)
        self.dec_f = LatentMappingLayer(latent_dim, hidden_dim, feat_dim, num_layers=2)

    def forward(self, x, dropout=0.1):
        z = self.enc(x, dropout)
        z_norm = F.normalize(z, p=2, dim=1)
        x_pred = torch.sigmoid(self.dec_f(z_norm, dropout))
        a_pred = torch.sigmoid(torch.mm(z, z.t()))
        return a_pred, x_pred, z_norm


class DuaLGR(nn.Module):
    def __init__(self, feat_dim, hidden_dim, latent_dim, endecoder, class_num=None, num_view=None):
        super(DuaLGR, self).__init__()
        self.num_view = num_view

        self.endecoder = endecoder
        self.gnn = GNN(feat_dim, hidden_dim, latent_dim)

        self.cluster_layer = [Parameter(torch.Tensor(class_num, latent_dim)) for _ in range(num_view)]
        self.cluster_layer.append(Parameter(torch.Tensor(class_num, latent_dim)))
        for v in range(num_view+1):
            self.register_parameter('centroid_{}'.format(v), self.cluster_layer[v])

    def forward(self, x, adjs, w, pseudo_label, alpha, quantize=0.8, varepsilon=0.3):
        a_pred_x, x_pred_x, z_norm_x = self.endecoder(x)

        omega = torch.mm(z_norm_x, z_norm_x.t())
        omega[omega > quantize] = 1
        omega[omega <= quantize] = 0

        homo_r = [cal_homo_ratio(adjs[v].detach().cpu().numpy(), np.asarray(pseudo_label), True) for v in range(self.num_view)]

        order = []
        for r in homo_r:
            if r <= varepsilon:
                order.append(0)
            else:
                od = int(np.floor(1 / (1 - r + 1e-9)))
                if od >= 8:
                    od = 8
                order.append(od)

        adj_refine = [self.get_A_with_order(omega, 1) + alpha * self.get_A_with_order(adjs[v], order[v]) for v in range(self.num_view)]

        S = sum(w[v] * adj_refine[v] for v in range(self.num_view)) / sum(w)

        z_all = []
        q_all = []

        for v in range(self.num_view):
            z_norm, a_pred, x_pred = self.gnn(x, adj_refine[v], order=1)
            z_all.append(z_norm)
            q = self.predict_distribution(z_norm, v)
            q_all.append(q)
        z_norm, a_pred, x_pred = self.gnn(x, S, order=1)
        z_all.append(z_norm)
        q = self.predict_distribution(z_norm, -1)
        q_all.append(q)
        return a_pred, x_pred, z_all, q_all, a_pred_x, x_pred_x

    def get_A_with_order(self, adj_label, order):
        adj = self.normalize_adj(adj_label)
        h = adj
        if order == 0:
            return torch.eye(h.shape[0]).to('cuda')
        elif order == 1:
            return h
        else:
            for i in range(order - 1):
                h = torch.mm(adj, h) + adj
            h /= order
        return h

    def predict_distribution(self, z, v, alpha=1.0):
        c = self.cluster_layer[v]
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - c, 2), 2) / alpha)
        q = q.pow((alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def normalize_adj(self, x):
        D = x.sum(1).detach().clone()
        r_inv = D.pow(-1).flatten()
        r_inv = r_inv.reshape((x.shape[0], -1))
        r_inv[torch.isinf(r_inv)] = 0.
        x = x * r_inv
        return x


class GNN(nn.Module):
    def __init__(self, feat_dim, hidden_dim, latent_dim):
        super(GNN, self).__init__()
        self.gnn = GraphEncoder(feat_dim, hidden_dim, latent_dim)
        self.dec = LatentMappingLayer(latent_dim, hidden_dim, feat_dim, num_layers=2)

    def forward(self, x, adj, order):
        z = self.gnn(x, adj, order)
        z_norm = F.normalize(z, p=2, dim=1)
        a_pred = torch.sigmoid(torch.mm(z_norm, z_norm.t()))
        x_pred = torch.sigmoid(self.dec(F.relu(z), dropout=False))
        return z_norm, a_pred, x_pred




