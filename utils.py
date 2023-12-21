import argparse
from collections import defaultdict

import dgl
import os.path
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from gensim.models.keyedvectors import Vocab
from six import iteritems
from sklearn.metrics import (auc, f1_score, precision_recall_curve,
                             roc_auc_score)

from walk import RWGraph


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default='../data/Movielens-100k plus',
                        #Movielens-100k plus/Amazon
                        help='Input dataset path')
    parser.add_argument('--k-layers', type=int, default=2,
                        help='K-layers')
    parser.add_argument('--features', type=str, default=None,
                        help='Input node features')

    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epoch. Default is 100.')

    parser.add_argument('--batch-size', type=int, default=512,
                        help='Number of batch_size. Default is 512.')

    parser.add_argument('--eval-type', type=str, default='all',
                        help='The edge type(s) for evaluation.')
    
    parser.add_argument('--schema', type=str, default=None,
                        help='The metapath schema (e.g., U-I-U,I-U-I).')

    parser.add_argument('--dimensions', type=int, default=200
                        ,
                        help='Number of dimensions. Default is 256.')

    parser.add_argument('--edge-dim', type=int, default=10,
                        help='Number of edge embedding dimensions. Default is 10.')
    
    parser.add_argument('--att-dim', type=int, default=20,
                        help='Number of attention dimensions. Default is 20.')

    parser.add_argument('--walk-length', type=int, default=10,
                        help='Length of walk per source. Default is 10.')

    parser.add_argument('--num-walks', type=int, default=20,
                        help='Number of walks per source. Default is 20.')

    parser.add_argument('--window-size', type=int, default=5,
                        help='Context size for optimization. Default is 5.')
    
    parser.add_argument('--negative-samples', type=int, default=5,
                        help='Negative samples for optimization. Default is 5.')
    
    parser.add_argument('--neighbor-samples', type=int, default=10,
                        help='Neighbor samples for aggregation. Default is 10.')

    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience. Default is 5.')
    
    return parser.parse_args()

def sparse_to_torch(X_train):
    values =X_train.data
    indices = np.vstack((X_train.row, X_train.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = X_train.shape
    X_train=torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return X_train

def adj_list(k,n,g):
    X_train = g.adj(scipy_fmt='coo')
    y = X_train.copy()
    adj_tensor = torch.Tensor(k, n, n)
    for i in range(k):
        adj_tensor[i] = F.normalize(dgl.khop_adj(g, k),p=1,dim=0)#sparse_to_torch(X_train).to_dense()
        #X_train = X_train.dot(y).tocoo()
    return torch.mean(adj_tensor,dim=0) ,adj_tensor

def get_DGLGraph_from_edges(pairs,vocab,self_loop):
    node1=torch.Tensor()
    node2=torch.Tensor()
    edge_type=torch.Tensor()
    for r,edges in pairs.items():
        for (x,y) in edges:
            if node1.size()[0]==0:
                node1=torch.tensor([vocab[x].index])
                node2=torch.tensor([vocab[y].index])
                edge_type=torch.tensor([int(r)])
            else:
                node1=torch.cat((node1,torch.tensor([vocab[x].index])))
                node2=torch.cat((node2,torch.tensor([vocab[y].index])))
                edge_type=torch.cat((edge_type,torch.tensor([int(r)])))

    tmp_G = dgl.graph((node1,node2))
    tmp_G = dgl.remove_self_loop(tmp_G)
    tmp_G = dgl.add_reverse_edges(tmp_G)
    if self_loop==True:
        tmp_G = dgl.add_self_loop(tmp_G)
    print("edges: {}".format(edge_type.size()[0]))
    return tmp_G,edge_type

def get_DGLHETEROGraph_from_edges(pairs,vocab):
    edge_type=torch.Tensor()
    data_dict={}
    for r,edges in pairs.items():
        node1 = torch.Tensor()
        node2 = torch.Tensor()
        for (x,y) in edges:
            if node1.size()[0]==0:
                node1=torch.tensor([vocab[x].index])
                node2=torch.tensor([vocab[y].index])
                edge_type=torch.tensor([int(r)])
            else:
                node1=torch.cat((node1,torch.tensor([vocab[x].index])))
                node2=torch.cat((node2,torch.tensor([vocab[y].index])))
                edge_type=torch.cat((edge_type,torch.tensor([int(r)])))
        data_dict[("node", r, "node")]=(node1,node2)

    tmp_G = dgl.heterograph(data_dict)
    for r, edges in pairs.items():
        tmp_G = dgl.remove_self_loop(tmp_G,etype=r)
    tmp_G = dgl.add_reverse_edges(tmp_G)
    for r, edges in pairs.items():
        tmp_G = dgl.add_self_loop(tmp_G, etype=r)
    #print(edge_type.size()[0])
    return tmp_G,edge_type

def load_training_data(f_name):
    print('We are loading data from:', f_name)
    edge_data_by_type = dict()

    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split('\t')
            #words[2]=int(words[2])-1
            if words[2] not in edge_data_by_type:
                edge_data_by_type[words[2]] = list()
            #x, y = words[0], str(int(words[1])+943)
            x, y = words[0], words[1]
            edge_data_by_type[words[2]].append((x, y))    #x,y  str

    return edge_data_by_type



def load_testing_data(f_name):
    print('We are loading data from:', f_name)
    true_edge_data_by_type = dict()
    false_edge_data_by_type = dict()
    all_edges = list()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split('\t')
            #words[2]=int(words[2])-1
            #x, y = words[0], str(int(words[1])+943)
            x, y = words[0], words[1]
            if int(words[3]) == 1:
                if words[2] not in true_edge_data_by_type:
                    true_edge_data_by_type[words[2]] = list()
                true_edge_data_by_type[words[2]].append((x, y))
            else:
                if words[2] not in false_edge_data_by_type:
                    false_edge_data_by_type[words[2]] = list()
                false_edge_data_by_type[words[2]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    return true_edge_data_by_type, false_edge_data_by_type

def load_attr_data(f_name,vocab):
    print('We are loading data from:', f_name)

    if os.path.exists(f_name+'/u_emb.user'):
        print("user attr file exist")
        user_attr = dict()

        user_file=f_name+'/u_emb.user'

        with open(user_file, 'r') as f:
            for line in f:
                words = line[:-1].split('|')
                user_attr[words[0]]=[]
                for word in words[1:]:
                    user_attr[words[0]].extend(word.strip().split(' '))


        item_attr = dict()
        item_file=f_name+'/i_emb.item'
        with open(item_file, 'r') as f:
            for line in f:
                words = line[:-1].split('|')
                #words[0]=str(int(words[0]) + 943)
                item_attr[words[0]]=[]
                for word in words[1:]:
                    item_attr[words[0]].extend(word.strip().split(' '))

        image_file = f_name + '/image_emb.item'
        with open(image_file, 'r') as f:
            for line in f:
                words = line[:-1].strip().split(' ')
                if words[0] in item_attr:
                    item_attr[words[0]].extend(words[1:])
        embs1=[]
        embs2=[]
        label1=[]
        label2=[]
        for key,value in user_attr.items():
            if key in vocab:
                emb=np.array(user_attr[key]).astype(float)
                embs1.append(emb)
                label1.append(vocab[key].index)
        for key, value in item_attr.items():
            if key in vocab:
                emb = np.array(item_attr[key]).astype(float)
                embs2.append(emb)
                label2.append(vocab[key].index)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return [[torch.as_tensor(embs2).float().to(device),torch.as_tensor(embs1).float().to(device)],[torch.as_tensor(label2).to(device),torch.as_tensor(label1).to(device)]]



def get_G_from_edges(edges):
    edge_dict = dict()
    for edge in edges:
        edge_key = str(edge[0]) + '_' + str(edge[1])
        if edge_key not in edge_dict:
            edge_dict[edge_key] = 1
        else:
            edge_dict[edge_key] += 1
    tmp_G = nx.Graph()
    for edge_key in edge_dict:
        weight = edge_dict[edge_key]
        x = edge_key.split('_')[0]
        y = edge_key.split('_')[1]
        tmp_G.add_edge(x, y)
        tmp_G[x][y]['weight'] = weight
    return tmp_G

def load_node_type(f_name):
    print('We are loading node type from:', f_name)
    node_type = {}
    with open(f_name, 'r') as f:
        for line in f:
            items = line.strip().split()
            node_type[items[0]] = items[1]
    return node_type

def generate_walks(network_data, num_walks, walk_length, schema, file_name):
    if schema is not None:
        node_type = load_node_type(file_name + '/node_type.txt')
    else:
        node_type = None

    all_walks = []
    # for layer_id in network_data:    #layer_id et edge_type
    #     tmp_data = network_data[layer_id]
    #     # start to do the random walk on a layer
    #
    #     layer_walker = RWGraph(get_G_from_edges(tmp_data),node_type)
    #     layer_walks = layer_walker.simulate_walks(num_walks, walk_length, schema=schema)
    #
    #     all_walks.append(layer_walks)
    tmp_data=[]
    for layer_id in network_data:    #layer_id et edge_type
        tmp_data.extend(network_data[layer_id])
    for layer_id in network_data:  # layer_id et edge_type
        layer_walker = RWGraph(get_G_from_edges(tmp_data), node_type)
        layer_walks = layer_walker.simulate_walks(num_walks, walk_length, schema=schema)
        all_walks.append(layer_walks)

    print('Finish generating the walks')

    return all_walks


def generate_pairs(all_walks, vocab, window_size):
    pairs = []
    skip_window = window_size // 2
    for layer_id, walks in enumerate(all_walks):
        for walk in walks:
            for i in range(len(walk)):
                for j in range(1, skip_window + 1):
                    if i - j >= 0:
                        pairs.append((vocab[walk[i]].index, vocab[walk[i - j]].index, layer_id))
                    if i + j < len(walk):
                        pairs.append((vocab[walk[i]].index, vocab[walk[i + j]].index, layer_id))

    return pairs

def generate_test_pairs(test_data, vocab,edge_types):
    pairs = []
    for r, test_pairs in test_data.items():
        for (x,y) in test_pairs:
            if x not in vocab or y not in vocab:
                continue
            else:
                for i in range(len(edge_types)):
                    if edge_types[i]==r:
                        pairs.append((vocab[x].index,vocab[y].index,i))
    return pairs

def generate_vocab(all_walks):
    index2word = []
    raw_vocab = defaultdict(int)

    for walks in all_walks:
        for walk in walks:
            for word in walk:
                raw_vocab[word] += 1

    vocab = {}
    for word, v in iteritems(raw_vocab):
        vocab[word] = Vocab(count=v, index=len(index2word))
        index2word.append(word)

    index2word.sort(key=lambda word: vocab[word].count, reverse=True)
    for i, word in enumerate(index2word):
        vocab[word].index = i

    return vocab, index2word  # 词向量，按word出现次数降序索引



def get_score(vector1, vector2):
    try:
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    except Exception as e:
        pass


def evaluate(true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    true_num = 0
    for i in range(np.shape(true_edges[0])[0]):
        tmp_score = get_score(true_edges[0][i],true_edges[1][i])
        if tmp_score is not None:
            true_list.append(1)
            prediction_list.append(tmp_score)
            true_num += 1

    for i in range(np.shape(false_edges[0])[0]):
        tmp_score = get_score(false_edges[0][i], false_edges[1][i])
        if tmp_score is not None:
            true_list.append(0)
            prediction_list.append(tmp_score)

    sorted_pred = prediction_list[:]
    sorted_pred.sort()
    threshold = sorted_pred[-true_num]

    y_pred = np.zeros(len(prediction_list), dtype=np.int32)
    for i in range(len(prediction_list)):
        if prediction_list[i] >= threshold:
            y_pred[i] = 1

    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    ps, rs, _ = precision_recall_curve(y_true, y_scores)
    return roc_auc_score(y_true, y_scores), f1_score(y_true, y_pred), auc(rs, ps)



