import math
import random
import sys

import numpy as np
import scipy.sparse as sp
import torch
import _pickle as pkl
import networkx as nx
from scipy.linalg import fractional_matrix_power, inv
from sklearn.preprocessing import MinMaxScaler
import scipy.io as sio
import random
import pandas as pd
import itertools
from sklearn.neighbors import NearestNeighbors
from torch_cluster import knn_graph
import h5py

''' Compute Personalized Page Ranking'''


def compute_ppr(a, dataset, alpha=0.2, self_loop=True, epsilon=0.01):
    '''
    :param a: numpy; adj_noeye. (adj without self loop) dense
    :param alpha:
    :param self_loop: bool
    :return: adj_ppr;
    '''
    print('computing ppr......')
    if self_loop:
        a = a + np.eye(a.shape[0])  # A^ = A + I_n
    d = np.diag(np.sum(a, 1))  # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)  # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)  # A~ = D^(-1/2) x A^ x D^(-1/2)
    ppr_adj = alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))  # a(I_n-(1-a)A~)^-1
    # ppr_adj[ppr_adj < epsilon] = 0
    if dataset == 'citeseer':
        print('additional processing')
        ppr_adj[ppr_adj < epsilon] = 0
        scaler = MinMaxScaler()
        scaler.fit(ppr_adj)
        ppr_adj = scaler.transform(ppr_adj)
    print('complete ppr')
    ppr_adj_labels = torch.FloatTensor(ppr_adj > epsilon).contiguous()
    ppr_adj = torch.FloatTensor(ppr_adj)

    return ppr_adj, ppr_adj_labels


def sample_graph(adj, drop_rate):
    drop_rate = torch.FloatTensor(np.ones(adj.shape[0]) * drop_rate)
    masks = torch.bernoulli(1. - drop_rate).unsqueeze(1)
    adj_droped = masks * adj
    adj_noeye = adj.mul(adj_droped.T)
    adj_droped = adj_noeye + torch.eye(adj_droped.shape[0])

    rowsum = adj_droped.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5)
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    adj_droped = adj_droped.mm(r_mat_inv_sqrt).T.mm(r_mat_inv_sqrt)
    return adj_droped, adj_noeye


def normalize_spadj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_weight(weights, p=0, eps=1e-12):
    '''
    :param weights:  a list [w1, w2, w3]
    :param p: default=1
    :param eps:
    :return:
    '''
    ws = np.array(weights)
    ws = np.power(ws, p)  # label soft
    ws = ws / ws.max()
    # r = max(np.power(np.power(ws, p).sum(), 1/p), eps)
    # ws = ws / r
    return ws


def normalize_spfeatures(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx


def normalize_features(x):
    rowsum = np.array(x.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_inv = r_inv.reshape((x.shape[0], -1))
    x = x * r_inv
    return x


def normalize_adj(x):
    # rowsum = np.array(x.sum(1))
    # colsum = np.array(x.sum(0))
    # r_inv = np.power(rowsum, -0.5).flatten()
    # r_inv[np.isinf(r_inv)] = 0.
    # c_inv = np.power(colsum, -0.5).flatten()
    # c_inv[np.isinf(c_inv)] = 0.
    # r_inv = r_inv.reshape((x.shape[0], -1))
    # c_inv = c_inv.reshape((-1, x.shape[1]))
    # x = x * r_inv * c_inv

    # rowsum = np.array(x.sum(1))
    # x = x + np.diag(rowsum) - np.eye(x.shape[0])
    # rowsum = np.array(x.sum(1))
    # r_inv = np.power(rowsum, -1.).flatten()
    # r_inv[np.isinf(r_inv)] = 0.
    # r_inv = r_inv.reshape((x.shape[0], -1))
    # x = x * r_inv

    rowsum = np.array(x.sum(1))
    r_inv = np.power(rowsum, -1.).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_inv = r_inv.reshape((x.shape[0], -1))
    x = x * r_inv
    return x


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_planetoid(dataset, path):
    path = path + dataset
    print('data loading.....')
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(path, dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(path, dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features_unorm = sp.vstack((allx, tx)).tolil()
    features_unorm[test_idx_reorder, :] = features_unorm[test_idx_range, :]
    features = normalize_spfeatures(features_unorm)
    adj_noeye = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj_temp = adj_noeye

    # norm
    adj = adj_noeye + adj_noeye.T.multiply(adj_noeye.T > adj_noeye) - adj_noeye.multiply(adj_noeye.T > adj_noeye)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_spadj(adj)

    # D1 = np.array(adj.sum(axis=1)) ** (-0.5)
    # D2 = np.array(adj.sum(axis=0)) ** (-0.5)
    # D1 = sp.diags(D1[:, 0], format='csr')
    # D2 = sp.diags(D2[0, :], format='csr')
    # adj = adj.dot(D1)
    # adj = D2.dot(adj)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    adj = torch.FloatTensor(np.array(adj.todense()))
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.argmax(labels, -1))
    adj_labels = torch.FloatTensor(np.array(adj_noeye.todense()) != 0)
    feature_labels = torch.FloatTensor(np.array(features_unorm.todense()))
    print('complete data loading')

    return labels, adj, features, adj_labels, feature_labels


def load_multi(dataset, root):
    # load the data: x, tx, allx, graph
    if dataset == 'acm':
        path = root + 'ACM3025.mat'
    elif dataset == 'dblp':
        path = root + 'DBLP4057.mat'
    elif dataset == 'imdb':
        path = root + 'imdb5k.mat'
    data = sio.loadmat(path)
    # print(dataset)
    # print(data)
    # rownetworks = np.array([(data['PLP'] - np.eye(N)).tolist()]) #, (data['PLP'] - np.eye(N)).tolist() , (data['PTP'] - np.eye(N)).tolist()])

    if dataset == "acm":
        truelabels, truefeatures = data['label'], data['feature'].astype(float)
        N = truefeatures.shape[0]  # nodes num
        rownetworks = np.array([(data['PAP']).tolist(), (data['PLP']).tolist()])
    elif dataset == "dblp":
        truelabels, truefeatures = data['label'], data['features'].astype(float)
        N = truefeatures.shape[0]
        rownetworks = np.array([(data['net_APA']).tolist(), (data['net_APCPA']).tolist(), (data['net_APTPA']).tolist()])
        rownetworks[2] += np.eye(rownetworks[2].shape[0])
        rownetworks = rownetworks[:2]
    elif dataset == 'imdb':
        truelabels, truefeatures = data['label'], data['feature'].astype(float)
        N = truefeatures.shape[0]
        rownetworks = np.array([(data['MAM']).tolist(), (data['MDM']).tolist(), (data['MYM']).tolist()])
        # rownetworks = rownetworks[:2]

    numView = rownetworks.shape[0]
    adjs_labels = []
    adjs = []
    feature_labels = torch.FloatTensor(np.array(truefeatures)).contiguous()
    features = torch.FloatTensor(normalize_features(truefeatures)).contiguous()
    for i in range(numView):
        adjs_labels.append(torch.FloatTensor(np.array(rownetworks[i])))
        adjs.append(torch.FloatTensor(normalize_adj(np.array(rownetworks[i]))))

    labels = torch.LongTensor(np.argmax(truelabels, -1)).contiguous()

    return labels, adjs, features, adjs_labels, feature_labels, numView


def eliminate_self_loops1(A):
    """Remove self-loops from the adjacency matrix."""
    A = A.tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


def eliminate_self_loops(G):
    G.adj_matrix = eliminate_self_loops1(G.adj_matrix)
    return G


def create_subgraph(sparse_graph, _sentinel=None, nodes_to_remove=None, nodes_to_keep=None):
    """Create a graph with the specified subset of nodes.

    Exactly one of (nodes_to_remove, nodes_to_keep) should be provided, while the other stays None.
    Note that to avoid confusion, it is required to pass node indices as named arguments to this function.

    Parameters
    ----------
    sparse_graph : SparseGraph
        Input graph.
    _sentinel : None
        Internal, to prevent passing positional arguments. Do not use.
    nodes_to_remove : array-like of int
        Indices of nodes that have to removed.
    nodes_to_keep : array-like of int
        Indices of nodes that have to be kept.

    Returns
    -------
    sparse_graph : SparseGraph
        Graph with specified nodes removed.

    """
    # Check that arguments are passed correctly
    if _sentinel is not None:
        raise ValueError("Only call `create_subgraph` with named arguments',"
                         " (nodes_to_remove=...) or (nodes_to_keep=...)")
    if nodes_to_remove is None and nodes_to_keep is None:
        raise ValueError("Either nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None and nodes_to_keep is not None:
        raise ValueError("Only one of nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None:
        nodes_to_keep = [i for i in range(sparse_graph.num_nodes()) if i not in nodes_to_remove]
    elif nodes_to_keep is not None:
        nodes_to_keep = sorted(nodes_to_keep)
    else:
        raise RuntimeError("This should never happen.")

    sparse_graph.adj_matrix = sparse_graph.adj_matrix[nodes_to_keep][:, nodes_to_keep]
    if sparse_graph.attr_matrix is not None:
        sparse_graph.attr_matrix = sparse_graph.attr_matrix[nodes_to_keep]
    if sparse_graph.labels is not None:
        sparse_graph.labels = sparse_graph.labels[nodes_to_keep]
    if sparse_graph.node_names is not None:
        sparse_graph.node_names = sparse_graph.node_names[nodes_to_keep]
    return sparse_graph


def largest_connected_components(sparse_graph, n_components=1):
    """Select the largest connected components in the graph.

    Parameters
    ----------
    sparse_graph : SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.

    Returns
    -------
    sparse_graph : SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = sp.csgraph.connected_components(sparse_graph.adj_matrix)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    return create_subgraph(sparse_graph, nodes_to_keep=nodes_to_keep)


class SparseGraph:
    """Attributed labeled graph stored in sparse matrix form.

    """

    def __init__(self, adj_matrix, attr_matrix=None, labels=None,
                 node_names=None, attr_names=None, class_names=None, metadata=None):
        """Create an attributed graph.

        Parameters
        ----------
        adj_matrix : sp.csr_matrix, shape [num_nodes, num_nodes]
            Adjacency matrix in CSR format.
        attr_matrix : sp.csr_matrix or np.ndarray, shape [num_nodes, num_attr], optional
            Attribute matrix in CSR or numpy format.
        labels : np.ndarray, shape [num_nodes], optional
            Array, where each entry represents respective node's label(s).
        node_names : np.ndarray, shape [num_nodes], optional
            Names of nodes (as strings).
        attr_names : np.ndarray, shape [num_attr]
            Names of the attributes (as strings).
        class_names : np.ndarray, shape [num_classes], optional
            Names of the class labels (as strings).
        metadata : object
            Additional metadata such as text.

        """
        # Make sure that the dimensions of matrices / arrays all agree
        if sp.isspmatrix(adj_matrix):
            adj_matrix = adj_matrix.tocsr().astype(np.float32)
        else:
            raise ValueError("Adjacency matrix must be in sparse format (got {0} instead)"
                             .format(type(adj_matrix)))

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Dimensions of the adjacency matrix don't agree")

        if attr_matrix is not None:
            if sp.isspmatrix(attr_matrix):
                attr_matrix = attr_matrix.tocsr().astype(np.float32)
            elif isinstance(attr_matrix, np.ndarray):
                attr_matrix = attr_matrix.astype(np.float32)
            else:
                raise ValueError("Attribute matrix must be a sp.spmatrix or a np.ndarray (got {0} instead)"
                                 .format(type(attr_matrix)))

            if attr_matrix.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency and attribute matrices don't agree")

        if labels is not None:
            if labels.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the label vector don't agree")

        if node_names is not None:
            if len(node_names) != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the node names don't agree")

        if attr_names is not None:
            if len(attr_names) != attr_matrix.shape[1]:
                raise ValueError("Dimensions of the attribute matrix and the attribute names don't agree")

        self.adj_matrix = adj_matrix
        self.attr_matrix = attr_matrix
        self.labels = labels
        self.node_names = node_names
        self.attr_names = attr_names
        self.class_names = class_names
        self.metadata = metadata

    def num_nodes(self):
        """Get the number of nodes in the graph."""
        return self.adj_matrix.shape[0]

    def num_edges(self):
        """Get the number of edges in the graph.

        For undirected graphs, (i, j) and (j, i) are counted as single edge.
        """
        if self.is_directed():
            return int(self.adj_matrix.nnz)
        else:
            return int(self.adj_matrix.nnz / 2)

    def get_neighbors(self, idx):
        """Get the indices of neighbors of a given node.

        Parameters
        ----------
        idx : int
            Index of the node whose neighbors are of interest.

        """
        return self.adj_matrix[idx].indices

    def is_directed(self):
        """Check if the graph is directed (adjacency matrix is not symmetric)."""
        return (self.adj_matrix != self.adj_matrix.T).sum() != 0

    def to_undirected(self):
        """Convert to an undirected graph (make adjacency matrix symmetric)."""
        if self.is_weighted():
            raise ValueError("Convert to unweighted graph first.")
        else:
            self.adj_matrix = self.adj_matrix + self.adj_matrix.T
            self.adj_matrix[self.adj_matrix != 0] = 1
        return self

    def is_weighted(self):
        """Check if the graph is weighted (edge weights other than 1)."""
        return np.any(np.unique(self.adj_matrix[self.adj_matrix != 0].A1) != 1)

    def to_unweighted(self):
        """Convert to an unweighted graph (set all edge weights to 1)."""
        self.adj_matrix.data = np.ones_like(self.adj_matrix.data)
        return self

    # Quality of life (shortcuts)
    def standardize(self):
        """Select the LCC of the unweighted/undirected/no-self-loop graph.

        All changes are done inplace.

        """
        G = self.to_unweighted().to_undirected()
        G = eliminate_self_loops(G)
        G = largest_connected_components(G, 1)
        return G

    def unpack(self):
        """Return the (A, X, z) triplet."""
        return self.adj_matrix, self.attr_matrix, self.labels


def load_npz_to_sparse_graph(file_name):
    """Load a SparseGraph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    sparse_graph : SparseGraph
        Graph in sparse matrix format.

    """
    with np.load(file_name) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])

        if 'attr_data' in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                        shape=loader['attr_shape'])
        elif 'attr_matrix' in loader:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader['attr_matrix']
        else:
            attr_matrix = None

        if 'labels_data' in loader:
            # Labels are stored as a CSR matrix
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                                   shape=loader['labels_shape'])
        elif 'labels' in loader:
            # Labels are stored as a numpy array
            labels = loader['labels']
        else:
            labels = None

        node_names = loader.get('node_names')
        attr_names = loader.get('attr_names')
        class_names = loader.get('class_names')
        metadata = loader.get('metadata')

    return SparseGraph(adj_matrix, attr_matrix, labels, node_names, attr_names, class_names, metadata)


def mine_Amazon(dataset, root):
    X = []
    if dataset == 'amazon_photos':
        path = root + 'amazon_electronics_photo.npz'
    elif dataset == 'amazon_computers':
        path = root + 'amazon_electronics_computers.npz'
    Amazon = load_npz_to_sparse_graph(path)
    Adj = sp.csr_matrix(Amazon.standardize().adj_matrix).A
    Attr = sp.csr_matrix(Amazon.standardize().attr_matrix).A
    Gnd = sp.csr_matrix(Amazon.standardize().labels).A
    Gnd = Gnd.T.squeeze()
    Attr = np.array(Attr)
    X.append(Attr)
    feature2 = Attr.dot(Attr.T)
    # feature2 = (feature2 - feature2.min(axis=0)) / (feature2.max(axis=0) - feature2.min(axis=0))
    # feature2 = np.where(feature2 > 0.5, 1., 0.)
    X.append(feature2)
    X.append(np.array(Adj))
    return X, Gnd


def load_data(dataset, path):
    """Load data."""
    if dataset in ['cora', 'citeseer', 'pubmed']:
        num_graph = 2
        labels, adjs, feature, adjs_labels, feature_label = load_planetoid(dataset, path)
        feature1_label = torch.matmul(feature_label, feature_label.t())
        feature1 = normalize_features(feature1_label)
        features = [feature, feature1]
        feature_labels = [feature_label, feature1_label]
        shared_feature = torch.cat(features, dim=-1)
        shared_feature_label = torch.cat(feature_labels, dim=1)
        adjs = [adjs, adjs.clone()]
        adjs_labels = [adjs_labels, adjs_labels.clone()]
    elif dataset in ['acm', 'dblp']:
        labels, adjs, feature, adjs_labels, feature_label, num_graph = load_multi(dataset, path)
        features = []
        feature_labels = []
        for _ in range(num_graph):
            features.append(feature)
            feature_labels.append(feature_label)
        shared_feature = feature
        shared_feature_label = feature_label

    elif dataset in ['chameleon', 'texas', 'squirrel']:
        labels, adjs, features, adjs_labels, feature_labels, shared_feature, shared_feature_label, num_graph = load_hete_data(dataset, 1)
    elif dataset == 'acm00':
        labels, adjs_labels, shared_feature, shared_feature_label, num_graph = load_synthetic_data(r=0.00, path=path)
    elif dataset == 'acm01':
        labels, adjs_labels, shared_feature, shared_feature_label, num_graph = load_synthetic_data(r=0.10, path=path)
    elif dataset == 'acm02':
        labels, adjs_labels, shared_feature, shared_feature_label, num_graph = load_synthetic_data(r=0.20, path=path)
    elif dataset == 'acm03':
        labels, adjs_labels, shared_feature, shared_feature_label, num_graph = load_synthetic_data(r=0.30, path=path)
    elif dataset == 'acm04':
        labels, adjs_labels, shared_feature, shared_feature_label, num_graph = load_synthetic_data(r=0.40, path=path)
    elif dataset == 'acm05':
        labels, adjs_labels, shared_feature, shared_feature_label, num_graph = load_synthetic_data(r=0.50, path=path)

    else:
        assert 'Dataset is not exist.'
    return labels, adjs_labels, shared_feature, shared_feature_label, num_graph


def load_graph(dataset, k):
    if k:
        path = 'graph/{}{}_graph.txt'.format(dataset, k)
    else:
        path = 'graph/{}_graph.txt'.format(dataset)

    data = np.loadtxt('data/{}.txt'.format(dataset))
    n, _ = data.shape

    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_noeye = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj_noeye + sp.eye(adj_noeye.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj_label = sparse_mx_to_torch_sparse_tensor(adj_noeye)

    return adj, adj_label


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def bin_kl_div(target, input, eps=1e-10):
    # if sum(q < 0) or sum(q > 1) or p < 0 or p > 1:
    #     assert "bin_kl_div: illegal input probability!"
    input_ = 1 - input
    target_ = 1 - target
    kl = target * (torch.log(target + eps) - torch.log(input + eps)) + target_ * (
                torch.log(target_ + eps) - torch.log(input_ + eps))
    return kl.mean()


def get_sharp_common_z(zs, w, temp=0.5):
    sum = 0.
    for i in range(len(zs)):
        sum = sum + w[i] * zs[i]
    avg_z = sum / len(zs)
    sharp_z = (torch.pow(avg_z, 1. / temp) / torch.sum(torch.pow(avg_z, 1. / temp), dim=1, keepdim=True)).detach()
    return sharp_z


def elbo_kl_loss(q, belief, target_prob, eps=1e-10):
    neg_ent = -q * torch.log(q + eps) - (1 - q) * torch.log(1 - q + eps)

    const_prob = math.log(sum(belief)) - torch.log(sum(target_prob) + eps)
    kl_loss = neg_ent + const_prob
    kl_loss = kl_loss.mean()

    return kl_loss


def cal_homo_ratio(adj, label, self_loop=True):
    class_num = max(label) + 1
    y_onehot = np.eye(class_num)[label]
    adj_y = np.matmul(y_onehot, y_onehot.T)

    if self_loop:
        adj = adj - np.eye(adj.shape[0])

    homo = np.sum(adj_y * adj)
    homo_ratio = homo / (np.sum(adj))
    return homo_ratio


def load_hete_data(dataset, self_loop):
    path = './data/{}/'.format(dataset)

    f = np.loadtxt(path + '{}.feature'.format(dataset), dtype=float)
    l = np.loadtxt(path + '{}.label'.format(dataset), dtype=int)
    features = sp.csr_matrix(f, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))

    label = torch.LongTensor(np.array(l))

    struct_edges = np.genfromtxt(path + '{}.edge'.format(dataset), dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])),
                         shape=(features.shape[0], features.shape[0]), dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    sadj = sadj + self_loop * sp.eye(sadj.shape[0])
    adjs_labels = [torch.FloatTensor(sadj.todense())]
    adjs = [torch.FloatTensor(normalize_adj(a.cpu().numpy())) for a in adjs_labels]

    feature_labels = [features]
    features = [normalize_features(x) for x in feature_labels]
    shared_feature = features[0]
    shared_feature_label = feature_labels[0]

    adjs_labels.append(torch.FloatTensor(sadj.todense()))
    adjs.append(adjs[0])
    num_graph = 2
    return label, adjs, features, adjs_labels, feature_labels, shared_feature, shared_feature_label, num_graph

def load_synthetic_data(r=0.00, path='./data/'):
    dataset = 'acm_hete_r{:.2f}.npz'.format(r)
    acm_dataset = np.load(path + dataset)
    shared_feature = torch.FloatTensor(acm_dataset['feature'])
    shared_feature_label = torch.FloatTensor(acm_dataset['feature_label'])
    adj_hete_v0 = torch.FloatTensor(acm_dataset['adj_hete_v0'])
    adj_hete_v1 = torch.FloatTensor(acm_dataset['adj_hete_v1'])
    labels = torch.LongTensor(acm_dataset['labels'])
    adjs_labels = []
    adjs_labels.append(adj_hete_v0)
    adjs_labels.append(adj_hete_v1)
    graph_num = len(adjs_labels)
    return labels, adjs_labels, shared_feature, shared_feature_label, graph_num

