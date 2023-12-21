import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from numpy import random
from torch.nn.parameter import Parameter
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax

from utils import *


def get_batches(pairs, batch_size):
    n_batches = (len(pairs) + (batch_size - 1)) // batch_size

    for idx in range(n_batches):
        x, y, t = [], [], []
        for i in range(batch_size):
            index = idx * batch_size + i
            if index >= len(pairs):
                break
            x.append(pairs[index][0])
            y.append(pairs[index][1])
            t.append(pairs[index][2])
        yield torch.tensor(x).long(), torch.tensor(y).long(), torch.tensor(t).long()

def get_test_batches(pairs):
    x, y, t= [], [], []
    for i in range(len(pairs)):
        x.append(pairs[i][0])
        y.append(pairs[i][1])
        t.append(pairs[i][2])
    return torch.tensor(x).long(), torch.tensor(y).long(), torch.tensor(t).long()

class EncodingModel(nn.Module):
    def __init__(
        self,num_nodes,embedding_size,edge_type,modality,num_users,*input_size    #[[1,2,21,5],[1,19,64,1000]]
    ):
        super(EncodingModel, self).__init__()
        self.num_nodes=num_nodes
        self.embedding_size=2*embedding_size
        self.input_size=input_size
        self.edge_type=edge_type
        self.linear=nn.ModuleList()
        self.modality=modality
        # for i in input_size:
        #         self.linear.extend([nn.Linear(j, embedding_size) for j in i])
        for i in input_size:
                self.linear.extend([nn.Linear(i, self.embedding_size)])
        self.embedding = Parameter(
            torch.FloatTensor(num_users,self.modality, self.embedding_size))
        #SENet
        self.linearx=nn.Sequential(nn.Linear(self.modality, self.modality//2),nn.ReLU(inplace=True),nn.Linear(self.modality//2, self.modality),nn.Sigmoid())
        self.lineary=nn.Sequential(nn.Linear(self.embedding_size, self.embedding_size//16),nn.ReLU(inplace=True),nn.Linear(self.embedding_size//16, self.embedding_size),nn.Sigmoid())


        self.cnn = nn.Sequential(nn.Conv1d(self.modality, self.modality, 1),nn.Sigmoid())
        self.linear3 = nn.Sequential(nn.Linear(self.embedding_size, embedding_size),nn.ReLU(inplace=True))


        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        for layer in self.linear:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.1)

        for layer in self.linear3:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.1)
        for layer in self.cnn:
            if isinstance(layer, nn.Conv1d):
                nn.init.xavier_uniform_(layer.weight)
        #SENet
        for layer in self.linearx:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.1)
        for layer in self.lineary:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.1)

    def forward(self, device,*attr):
        embs = torch.FloatTensor(self.num_nodes, self.modality, self.embedding_size).to(device)
        for j in range(len(attr[0])):
            if j==0:
                emb = torch.Tensor().float().to(device)
                t = 0
                for i, layer in enumerate(self.linear[self.modality * j:self.modality * (j + 1)]):
                    modal_emb = layer(attr[0][j][:, t:t + self.input_size[i]]).unsqueeze(1)
                    emb = torch.cat([emb, modal_emb], dim=1)
                    t += self.input_size[i]
                embs[attr[1][j]] = emb
            if j==1:
                embs[attr[1][j]] = self.embedding


        #SENet
        emb1 = self.lineary(torch.sum(embs, dim=1)).unsqueeze(1)  # (n,1,e_size)
        emb2 = self.linearx(torch.sum(embs, dim=2)).unsqueeze(2)  # (n,n_modal,1)
        attent=self.cnn(torch.matmul(emb2,emb1))  #(n,n_modal,e_size)

        final_embs=torch.max(self.linear3(torch.mul(attent,embs)),dim=1)[0]


        return final_embs,embs

class CSANLayer(nn.Module):
    def __init__(
        self, num_nodes, embedding_size, embedding_u_size,head_nums
    ):
        super(CSANLayer, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_size = embedding_size
        self.embedding_u_size = embedding_u_size
        self.num_heads=head_nums
        self.fc=nn.Linear(embedding_size,self.num_heads*embedding_u_size,bias=False)

        self.attn_l = Parameter(
            torch.FloatTensor(1, self.num_heads, self.embedding_u_size))
        self.attn_r = Parameter(
            torch.FloatTensor(1, self.num_heads, self.embedding_u_size))
        self.type_attn_l = Parameter(
            torch.FloatTensor(1, self.num_heads, self.embedding_u_size))
        self.type_attn_r = Parameter(
            torch.FloatTensor(1, self.num_heads, self.embedding_u_size))

        self.activate=nn.LeakyReLU(negative_slope=0.2,inplace=True)


        self.reset_parameters()
    def reset_parameters(self):
        gain=nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc.weight,gain=gain)
        nn.init.xavier_uniform_(self.attn_l,gain=gain)
        nn.init.xavier_uniform_(self.attn_r,gain=gain)
        nn.init.xavier_uniform_(self.type_attn_l,gain=gain)
        nn.init.xavier_uniform_(self.type_attn_r,gain=gain)


    def forward(self,g,feat):
        funcs = {}
        for c_etype in g.canonical_etypes:
            srctype, etype, dsttype = c_etype
            feat_src = feat_dst = self.fc(feat).view(-1, self.num_heads, self.embedding_u_size)  # (n,num_heads, output_size)\

            g[etype].srcdata['h'] = feat_src
            g[etype].update_all(fn.copy_src('h', 'm'), fn.mean('m', 'Wh_%s' % etype))
            mean_feat_dst = g[etype].dstdata['Wh_%s' % etype]
            el = (feat_src * self.type_attn_l).sum(dim=-1).unsqueeze(-1)
            er = (mean_feat_dst * self.type_attn_r).sum(dim=-1).unsqueeze(-1)
            g[etype].dstdata['type_att_%s' % etype] = torch.sigmoid(el + er)

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            g.srcdata.update({'ft_%s' % etype: feat_src, 'el_%s' % etype: el})
            g.dstdata.update({'er_%s' % etype: er})
            g[etype].apply_edges(fn.u_add_v('el_%s' % etype, 'er_%s' % etype, 'e_%s' % etype))
            g[etype].apply_edges(fn.v_mul_e('type_att_%s' % etype, 'e_%s' % etype, 'coff_%s' % etype))
            e = self.activate(g[etype].edata.pop('coff_%s' % etype))

            g[etype].edata['a_%s' % etype] = edge_softmax(g[etype], e)
            funcs[etype] = (fn.u_mul_e('ft_%s' % etype, 'a_%s' % etype, 'm_%s' % etype), fn.sum('m_%s' % etype, 'ft'))

        g.multi_update_all(funcs, 'mean')
        rst = g.dstdata['ft']


        last_embed = F.normalize(rst, dim=-1)

        return last_embed


class CSANModel(nn.Module):
    def __init__(
        self, num_nodes, embedding_size, embedding_u_size,head_nums,*dim_list
    ):
        super(CSANModel, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_size = embedding_size
        self.embedding_u_size = embedding_u_size

        self.model = nn.ModuleList([CSANLayer(num_nodes, dim_list[0], dim_list[1],head_nums[0]),
                                    CSANLayer(num_nodes, head_nums[0]*dim_list[1], dim_list[2], head_nums[1]),
        ])
        self.activate=nn.ReLU(inplace=True)
        self.linear = nn.Linear(embedding_size, embedding_u_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.1)
    def forward(self,g,feat):
        #node_embedd = self.linear(node_embed)
        embed = self.model[0](g,feat)
        embed=embed.reshape(self.num_nodes,-1)
        embed=self.activate(embed)
        embed = self.model[1](g,embed)
        embed=embed.reshape(self.num_nodes,-1)
        last_embed = F.normalize(embed, dim=1)

        return last_embed



class MHAGTModel(nn.Module):
    def __init__(
        self, num_nodes, embedding_size,edge_type_count, embedding_u_size,dim_a,k,modality
    ):
        super(MHAGTModel, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_size = embedding_size
        self.embedding_u_size = embedding_u_size
        self.edge_type_count = edge_type_count
        self.dim_a = dim_a
        self.k=k   #layer
        self.modality=modality
        self.alpha=0.5
        # self.HGATmodel=nn.ModuleList([CSANModel(
        # num_nodes, embedding_size,embedding_u_size,[8,1],*[embedding_size,embedding_size//2,embedding_u_size])  for  _ in range(self.modality)
        # ] )
        self.HGATmodel = nn.ModuleList([CSANModel(
            num_nodes, embedding_size, embedding_u_size, [8, 1],
            *[2*embedding_size, embedding_size, embedding_size]) for _ in range(self.modality)
        ])

        # self.trans_weights_s1 = Parameter(
        #     torch.FloatTensor(embedding_u_size, dim_a)
        # )
        self.trans_weights_s1 = Parameter(
            torch.FloatTensor(embedding_size, dim_a)
        )
        self.trans_weights_s2 = Parameter(torch.FloatTensor(dim_a, 1))
        # self.trans_weights_s3 = Parameter(
        #     torch.FloatTensor(embedding_u_size, embedding_size)
        # )
        self.trans_weights_s3 = Parameter(
            torch.FloatTensor(embedding_size, embedding_size)
        )
        #linear
        self.linear=nn.Linear(2*self.embedding_size,self.embedding_size)

        self.reset_parameters()
    def reset_parameters(self):
        self.trans_weights_s1.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.trans_weights_s2.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.trans_weights_s3.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

        #linear
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.1)

    def forward(self,G,fuse_emb,modality_embs,x_inputs,y_inputs):

        embed=torch.cat([self.HGATmodel[i](G,modality_embs[:,i,:]).unsqueeze(1) for i in range(self.modality)],
                             dim=1)     #(nodes,modailty_nums,embedding_u_size)

        trans_w_s1 = self.trans_weights_s1
        trans_w_s2 = self.trans_weights_s2
        trans_w_s3 = self.trans_weights_s3


        attention =F.softmax(
                torch.matmul(torch.tanh(torch.matmul(embed, trans_w_s1)), trans_w_s2
                             ).squeeze(2),
                dim=1).unsqueeze(1)
             # (nodes,1,modailty_nums)

        node_type_embed = torch.matmul(attention, embed)  # (nodes,1,embedding_u_size)
        del attention
        node_type_embed = self.linear(torch.cat([fuse_emb,torch.matmul(node_type_embed, trans_w_s3).squeeze(1)],dim=1))
        #node_type_embed = fuse_emb+torch.matmul(node_type_embed, trans_w_s3).squeeze()

        x_embed = node_type_embed[x_inputs]

        y_embed = node_type_embed[y_inputs]

        last_x_embed = F.normalize(x_embed, dim=1)
        last_y_embed = F.normalize(y_embed, dim=1)

        return [last_x_embed,last_y_embed,F.normalize(node_type_embed, dim=1)]


class StructureModel(nn.Module):
    def __init__(
        self, num_nodes, embedding_size,k
    ):
        super(StructureModel, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_size = embedding_size
        self.k=k

        self.alpha=0.5
         #linear

        #self.linear=nn.Sequential(nn.Linear(self.num_nodes,self.embedding_size))
        # self.linear = nn.Sequential(nn.Linear(self.num_nodes, 1200),nn.ReLU(inplace=True),
        #                             nn.Linear(1200, self.embedding_size),
        #                             )
        self.embedding = Parameter(torch.FloatTensor(self.num_nodes,self.embedding_size))

        self.reset_parameters()
    def reset_parameters(self):
        self.embedding.data.uniform_(-1.0,1.0)

        # for layer in self.linear:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.xavier_uniform_(layer.weight)
        #         nn.init.constant_(layer.bias, 0.1)
    def forward(self,x_inputs, y_inputs):

        #
        # x_embed = input_emb[x_inputs]
        # y_embed = input_emb[y_inputs]
        #
        # last_x_embed=self.linear(x_embed)
        # last_y_embed=self.linear(y_embed)

        last_embed = F.normalize(self.embedding,dim=1)
        last_x_embed=last_embed[x_inputs]
        last_y_embed=last_embed[y_inputs]

        # last_x_embed = F.normalize(last_x_embed, dim=1)
        # last_y_embed = F.normalize(last_y_embed, dim=1)



        # last_embed = F.normalize(self.embedding,dim=1)
        # last_x_embed=last_embed[x_inputs]
        # last_y_embed=last_embed[y_inputs]
        return [last_x_embed,last_y_embed,last_embed]

class NSLoss(nn.Module):
    def __init__(self, num_nodes, num_sampled, embedding_size):
        super(NSLoss, self).__init__()
        self.num_nodes = num_nodes
        self.num_sampled = num_sampled
        self.embedding_size = embedding_size
        self.weights = Parameter(torch.FloatTensor(num_nodes, embedding_size))
        self.sample_weights = F.normalize(
            torch.Tensor(
                [
                    (math.log(k + 2) - math.log(k + 1)) / math.log(num_nodes + 1)
                    for k in range(num_nodes)
                ]
            ),
            dim=0,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.weights.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    def forward(self, input, embs, label):
        n = input.shape[0]
        log_target = torch.log(
            torch.sigmoid(torch.sum(torch.mul(embs, self.weights[label]), 1))
        )
        negs = torch.multinomial(
            self.sample_weights, self.num_sampled * n, replacement=True
        ).view(n, self.num_sampled)
        noise = torch.neg(self.weights[negs])  #（n,num_sampled,embedding_size)
        sum_log_sampled = torch.sum(torch.log(torch.sigmoid(torch.bmm(noise, embs.unsqueeze(2)))), 1).squeeze()

        loss = log_target + sum_log_sampled
        return -loss.sum() / n

# class CSLoss(nn.Module):
#     def __init__(self, num_nodes, num_sampled, embedding_size):
#         super(CSLoss, self).__init__()
#         self.num_nodes = num_nodes
#         self.num_sampled = num_sampled
#         self.embedding_size = embedding_size
#
#
#         # linear
#
#         #self.linear = nn.Sequential(nn.Linear(self.embedding_size, self.self.num_nodes))
#         self.linear = nn.Sequential(nn.Linear(self.embedding_size, 1200),nn.ReLU(inplace=True),
#                                     nn.Linear(1200,self.num_nodes),
#                                     )
#         self.loss_function=nn.MSELoss()
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         for layer in self.linear:
#             if isinstance(layer, nn.Linear):
#                 nn.init.xavier_uniform_(layer.weight)
#                 nn.init.constant_(layer.bias, 0.1)
#     def forward(self, input, embs):
#         output=self.linear(embs)
#         output = F.normalize(output, dim=1)
#         loss=self.loss_function(input,output)
#
#         return loss
class CSLoss(nn.Module):
    def __init__(self, num_nodes, num_sampled, embedding_size):
        super(CSLoss, self).__init__()
        self.num_nodes = num_nodes
        self.num_sampled = num_sampled
        self.embedding_size = embedding_size
        self.sample_weights = F.normalize(
            torch.Tensor(
                [
                    (math.log(k + 2) - math.log(k + 1)) / math.log(num_nodes + 1)
                    for k in range(num_nodes)
                ]
            ),
            dim=0,
        )

    def forward(self, input, embs, weights,label):
        n = input.shape[0]
        log_target = torch.log(
            torch.sigmoid(torch.sum(torch.mul(embs, weights[label]), 1))
        )
        negs = torch.multinomial(
            self.sample_weights, self.num_sampled * n, replacement=True
        ).view(n, self.num_sampled)
        noise = torch.neg(weights[negs])  # （n,num_sampled,embedding_size)
        sum_log_sampled = torch.sum(torch.log(torch.sigmoid(torch.bmm(noise, embs.unsqueeze(2)))), 1).squeeze()

        loss = log_target + sum_log_sampled


        return -loss.sum() / n

def train_model(file_name,network_data):   #training_data_by_type：{‘edge_type'}=[[nodex,nodey]...]
    all_walks = generate_walks(network_data, args.num_walks, args.walk_length, args.schema, file_name)

    vocab, index2word = generate_vocab(all_walks)
    train_pairs = generate_pairs(all_walks, vocab, args.window_size)     #(x,y,r)
    print(len(vocab))
    print('train pairs:{}'.format(len(train_pairs)))

    attr_list = load_attr_data(file_name, vocab)
    if len(attr_list[0]) == 1:
        num_users = 0
        print("Users: {}".format(num_users))
    else:
        num_users = attr_list[0][1].size()[0]
        print("Users: {}".format(num_users))
    print("attr size", format(attr_list[0][0].size()))
    print("attr key size", format(attr_list[1][0].size()))


    k=args.k_layers
    modality=2
    edge_types = list(network_data.keys())
    num_nodes = len(index2word)
    edge_type_count = len(edge_types)
    epochs = args.epoch
    batch_size = args.batch_size
    embedding_size = args.dimensions   #256
    num_sampled = args.negative_samples
    dim_a = args.att_dim
    embedding_u_size = args.edge_dim

    neighbor_samples = args.neighbor_samples
    test_true_pairs=generate_test_pairs(testing_true_data_by_edge, vocab,edge_types)
    test_false_pairs=generate_test_pairs(testing_false_data_by_edge, vocab,edge_types)
    print(edge_types)
    G, edge_type = get_DGLHETEROGraph_from_edges(network_data, vocab)
    Adj_list=[G.adj(etype=edge_type) for edge_type in edge_types]
    for i in range(len(Adj_list)):
        if i==0:
            S_adj = Adj_list[i]
        else:
            S_adj += Adj_list[i]
    S_adj=S_adj.to_dense()
    P_khop=0.5*S_adj+0.25*torch.mm(S_adj,S_adj)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    neighbors = [[[] for __ in range(edge_type_count)] for _ in range(num_nodes)]
    for r in range(edge_type_count):  # 产生邻居列表
        g = network_data[edge_types[r]]
        for (x, y) in g:
            ix = vocab[x].index
            iy = vocab[y].index
            neighbors[ix][r].append(iy)
            neighbors[iy][r].append(ix)
        for i in range(num_nodes):
            if len(neighbors[i][r]) == 0:
                neighbors[i][r] = [i] * neighbor_samples
            elif len(neighbors[i][r]) < neighbor_samples:
                neighbors[i][r].extend(
                    list(
                        np.random.choice(
                            neighbors[i][r],
                            size=neighbor_samples - len(neighbors[i][r]),
                        )
                    )
                )
            elif len(neighbors[i][r]) > neighbor_samples:
                neighbors[i][r] = list(
                    np.random.choice(neighbors[i][r], size=neighbor_samples)
                )
    movie=['Movielens-100k plus','M2','M3','M4','M5']
    amazon=['amazon','A2']
    if args.input.split('/')[-1] in amazon:
        encoding_model = EncodingModel(num_nodes, embedding_size, edge_type_count, modality, num_users, *[3839, 1000])
    if args.input.split('/')[-1] in movie:
        encoding_model = EncodingModel(num_nodes, embedding_size, edge_type_count, modality, num_users, *[84, 1000])

    model = MHAGTModel(
        num_nodes, embedding_size ,edge_type_count, embedding_u_size,dim_a,k,modality
    )
    structure_model=StructureModel(num_nodes, embedding_size, k)
    nsloss1 = NSLoss(num_nodes, num_sampled, embedding_size)
    nsloss2=NSLoss(num_nodes, num_sampled, embedding_size)
    nsloss3=NSLoss(num_nodes, num_sampled, embedding_size*2)
    csloss  =CSLoss(num_nodes, num_sampled, embedding_size)

    encoding_model.to(device)
    model.to(device)
    nsloss1.to(device)
    nsloss2.to(device)
    nsloss3.to(device)
    csloss.to(device)
    structure_model.to(device)

    optimizer = torch.optim.Adam([{"params": structure_model.parameters()},{"params": model.parameters()},
                                  {"params": nsloss2.parameters()},{"params": nsloss1.parameters()},
                                   {"params": csloss.parameters()},{"params": nsloss3.parameters()},
                                 {"params": encoding_model.parameters()}], lr=1e-4)

    best_score = 0
    patience = 0

    for epoch in range(epochs):
        random.shuffle(train_pairs)
        batches = get_batches(train_pairs, batch_size)


        data_iter = tqdm.tqdm(
            batches,
            desc="epoch %d" % (epoch),
            total=(len(train_pairs) + (batch_size - 1)) // batch_size,
            bar_format="{l_bar}{r_bar}",
        )
        avg_loss = 0.0

        for i, data in enumerate(data_iter):
            optimizer.zero_grad()
            fuse_emb,modality_embs=encoding_model(device,*attr_list)
            content_embs = model(G.to(device),fuse_emb,modality_embs,data[0].to(device),data[1].to(device))   #[0]:x [1]:y   [2]edge_type   [3]x_neighbor   [4]y_neighbor                                                                                       # [3]:neigh


            structure_embs = structure_model(data[0].to(device),data[1].to(device))   #[0]:x [1]:y   [2]edge_type   [3]x_neighbor   [4]y_neighbor                                                                                       # [3]:neigh

            loss = nsloss1(data[0].to(device), content_embs[0], data[1].to(device))\
                   + nsloss2(data[0].to(device), structure_embs[0], data[1].to(device)) \
                    + nsloss3(data[0].to(device), torch.cat([content_embs[0], structure_embs[0]], dim=1),data[1].to(device))\
                    + csloss(data[0].to(device), content_embs[0], structure_embs[2], data[1].to(device))

            loss.backward()
            optimizer.step()



            avg_loss += loss.item()

            if i % 1000 == 0:
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "loss": loss.item(),
                }
                data_iter.write(str(post_fix))

        with torch.no_grad():
            valid_aucs, valid_f1s, valid_prs = [], [], []
            test_aucs, test_f1s, test_prs = [], [], []
            fuse_emb,modality_embs = encoding_model(device,*attr_list)
            data=get_test_batches(test_true_pairs)
            content_embs = model(G.to(device),fuse_emb,modality_embs,data[0].to(device),data[1].to(device))   #[0]:x [1]:y   [2]edge_type   [3]x_neighbor   [4]y_neighbor                                                                                       # [3]:neigh
            structure_embs = structure_model(data[0].to(device),data[1].to(device))   #[0]:x [1]:y   [2]edge_type   [3]x_neighbor   [4]y_neighbor                                                                                       # [3]:neigh
            pos_embs = ((torch.cat([content_embs[0], structure_embs[0]],dim=1)).cpu().detach().numpy(), (torch.cat([content_embs[1], structure_embs[1]],dim=1)).cpu().detach().numpy())
            #pos_embs = ((content_embs[0]).cpu().detach().numpy(),(content_embs[1]).cpu().detach().numpy())
            #pos_embs = ((structure_embs[0]).cpu().detach().numpy(),(structure_embs[1]).cpu().detach().numpy())

            data=get_test_batches(test_false_pairs)
            content_embs = model(G.to(device),fuse_emb,modality_embs,data[0].to(device),data[1].to(device))   #[0]:x [1]:y   [2]edge_type   [3]x_neighbor   [4]y_neighbor                                                                                       # [3]:neigh
            structure_embs = structure_model(data[0].to(device),data[1].to(device))   #[0]:x [1]:y   [2]edge_type   [3]x_neighbor   [4]y_neighbor                                                                                       # [3]:neigh
            neg_embs = ((torch.cat([content_embs[0], structure_embs[0]],dim=1)).cpu().detach().numpy(), (torch.cat([content_embs[1], structure_embs[1]],dim=1)).cpu().detach().numpy())
            #neg_embs = ((content_embs[0]).cpu().detach().numpy(), (content_embs[1]).cpu().detach().numpy())
            #neg_embs = ((structure_embs[0]).cpu().detach().numpy(),(structure_embs[1]).cpu().detach().numpy())



            valid_aucs, valid_f1s, valid_prs = [], [], []
            test_aucs, test_f1s, test_prs = [], [], []
            for i in range(edge_type_count):
                if args.eval_type == "all" or edge_types[i] in args.eval_type.split(","):
                    tmp_auc, tmp_f1, tmp_pr = evaluate(
                        pos_embs,neg_embs
                    )
                    valid_aucs.append(tmp_auc)
                    valid_f1s.append(tmp_f1)
                    valid_prs.append(tmp_pr)

                    tmp_auc, tmp_f1, tmp_pr = evaluate(
                        pos_embs, neg_embs
                    )
                    test_aucs.append(tmp_auc)
                    test_f1s.append(tmp_f1)
                    test_prs.append(tmp_pr)
            print("valid auc:", np.mean(valid_aucs))
            print("valid pr:", np.mean(valid_prs))
            print("valid f1:", np.mean(valid_f1s))

            average_auc = np.mean(test_aucs)
            average_f1 = np.mean(test_f1s)
            average_pr = np.mean(test_prs)

            cur_score = np.mean(valid_aucs)
            if cur_score > best_score:
                best_score = cur_score
                #filepath = 'model/' + args.input.split('/')[-1] + '/DMGI'

                #torch.save({"node embedding": torch.cat([content_embs[2], structure_embs[2]],dim=1), "vocab": vocab},
                #           os.path.join(filepath, 'model_result.pkl'))
                patience = 0
            else:
                patience += 1
                if patience > args.patience:
                    print("Early Stopping")
                    break
    return average_auc, average_f1, average_pr


if __name__ == "__main__":
    args = parse_args()
    file_name = args.input
    print(args)
    training_data = load_training_data(file_name + "/u1.base")
    valid_true_data_by_edge, valid_false_data_by_edge = load_testing_data(
        file_name + "/u1_negative.test"
    )
    testing_true_data_by_edge, testing_false_data_by_edge = load_testing_data(
        file_name + "/u1_negative.test"
    )

    average_auc, average_f1, average_pr = train_model(file_name,training_data)

    print("Overall ROC-AUC:", average_auc)
    print("Overall PR-AUC", average_pr)
    print("Overall F1:", average_f1)
