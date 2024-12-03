import copy
from torch import nn
import torch
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, n_users, n_items, text_feat, edge_p, user_nids, item_nids, args):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.layers = args.layers
        self.embedding_dim = args.dim
        self.reg_weight = args.reg_weight
        self.device = args.device
        self.ssl_reg = args.ssl_reg
        self.ssl_temp = args.ssl_temp
        self.edge_p = edge_p
        self.user_id_embedding = nn.Embedding(n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, self.embedding_dim)
        self.users, self.items = user_nids, item_nids
        self.original_graph = self.get_norm_adj_mat(self.users, self.items).to(self.device)
        self.train_u_i_graph = copy.deepcopy(self.original_graph)
        if args.use_text:
            self.text_feat = torch.tensor(text_feat).to(args.device)
            self.text_trans = nn.Linear(self.text_feat.shape[-1], self.embedding_dim)
        self.use_text = args.use_text
        self.Str_CL = args.Str_CL
        self.Sem_CL = args.Sem_CL
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
            nn.init.xavier_normal_(m.weight)

    def get_norm_adj_mat(self, users, items):
        row_data = np.array(users)
        col_data = np.array(items)
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        data_dict = dict(zip(zip(row_data, col_data + self.n_users), [1] * len(row_data)))
        data_dict.update(dict(zip(zip(col_data + self.n_users, row_data), [1] * len(row_data))))
        A._update(data_dict)
        sumArr = A.sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7 # avoid divide by zero
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D @ A @ D

        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def forward(self, graph, items=None):
        u_i_embedding = torch.cat([self.user_id_embedding.weight, self.item_id_embedding.weight], dim=0)
        u_i_embeddings_list = [u_i_embedding]

        for layer_idx in range(self.layers):
            u_i_embedding = torch.sparse.mm(graph, u_i_embedding)
            u_i_embeddings_list.append(u_i_embedding)

        u_i_embedding = torch.stack(u_i_embeddings_list, dim=1)
        u_i_embedding = torch.mean(u_i_embedding, dim=1)

        u_embedding, i_embedding = torch.split(u_i_embedding, [self.n_users, self.n_items])
        layer_u_embedding0, layer_i_embedding0 = torch.split(u_i_embeddings_list[0], [self.n_users, self.n_items])
        layer_u_embedding1, layer_i_embedding1 = torch.split(u_i_embeddings_list[1], [self.n_users, self.n_items])

        if items is not None:
            # training
            h = torch.zeros_like(i_embedding[items])
            if self.use_text: h += self.text_trans(self.text_feat[items])
            h = F.normalize(h, p=2, dim=-1)
        else:
            # testing
            h = torch.zeros_like(i_embedding)
            if self.use_text: h += self.text_trans(self.text_feat)
            h = F.normalize(h, p=2, dim=-1)

        return u_embedding, i_embedding, h, layer_u_embedding0, \
               layer_u_embedding1, layer_i_embedding0, layer_i_embedding1

    def ssl_layer_loss(self, side_embedding, fuse_embedding, idx=None):
        current_side_emb = side_embedding
        current_fuse_emb = fuse_embedding
        norm_side_emb = F.normalize(current_side_emb)
        norm_fuse_emb = F.normalize(current_fuse_emb)

        rand_idx = torch.randperm(len(fuse_embedding)).to(self.device)
        norm_all_fuse_emb = norm_fuse_emb[rand_idx]
        rand_idx = torch.randperm(len(side_embedding)).to(self.device)
        norm_all_side_emb = norm_side_emb[rand_idx]

        if idx is not None:
            # Str_CL
            norm_fuse_emb = norm_fuse_emb[idx]
            norm_side_emb = norm_side_emb[idx]
            pos_score = torch.exp(torch.mul(norm_side_emb, norm_fuse_emb).sum(dim=1) / self.ssl_temp)
            ttl_score = (torch.exp(torch.matmul(norm_side_emb, norm_all_fuse_emb.transpose(0, 1)) / self.ssl_temp)).sum(
                dim=-1)
        else:
            # Sem_CL
            pos_score = 2 * torch.exp(torch.mul(norm_side_emb, norm_fuse_emb).sum(dim=1) / self.ssl_temp)
            ttl_score = (torch.exp(torch.matmul(norm_side_emb, norm_all_fuse_emb.transpose(0, 1)) / self.ssl_temp)).sum(
                dim=-1) \
            + (torch.exp(torch.matmul(norm_fuse_emb, norm_all_side_emb.transpose(0, 1)) / self.ssl_temp)).sum(dim=-1)
        ssl_loss = -(torch.log(pos_score / ttl_score)).mean()
        return ssl_loss

    def reg_loss(self, *embeddings):
        return 0.5 * sum([(embeddings[i]**2).sum() / embeddings[i].shape[0] for i in range(len(embeddings))])

    def calculate_loss(self, interaction, gamma):
        user, pos_item, neg_item, neg_item2 = interaction
        all_items = torch.cat(interaction[1:], dim=-1)
        user_all_id_embeddings, item_all_id_embeddings, h, layer_u_embedding0, \
           layer_u_embedding1, layer_i_embedding0, layer_i_embedding1 = self.forward(self.train_u_i_graph, all_items)

        # CL_loss for fusion views and text views
        ssl_loss = 0
        if self.Str_CL:
            i_embeddings1 = layer_i_embedding1[pos_item]
            u_embeddings1 = layer_u_embedding1[user]
            i_embeddings0 = layer_i_embedding0[pos_item]
            u_embeddings0 = layer_u_embedding0[user]
            ssl_loss += self.ssl_layer_loss(i_embeddings1, u_embeddings0)
            ssl_loss += self.ssl_layer_loss(u_embeddings1, i_embeddings0)

        if self.Sem_CL and self.use_text:
            idx = torch.arange(len(pos_item) * 2, len(pos_item) * 3).to(self.device)  # provide more chance for tail part
            i_embeddings = item_all_id_embeddings[all_items]
            ssl_loss += self.ssl_layer_loss(i_embeddings, h, idx)

        # calculate BPR Loss
        u_embeddings = user_all_id_embeddings[user]
        pos_embeddings = item_all_id_embeddings[pos_item]
        neg_embeddings = item_all_id_embeddings[neg_item]
        neg_embeddings2 = item_all_id_embeddings[neg_item2]
        if self.use_text:
            pos_embeddings += h[:len(pos_item)]
            neg_embeddings += h[len(pos_item):len(pos_item) + len(neg_item)]
            neg_embeddings2 += h[len(pos_item) + len(neg_item):]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        weight = torch.mul(pos_embeddings, neg_embeddings).sum(dim=-1, keepdim=True)
        neg_scores2 = torch.mul(u_embeddings, neg_embeddings2).sum(dim=-1)
        weight2 = torch.mul(pos_embeddings, neg_embeddings2).sum(dim=-1, keepdim=True)

        bpr_loss = (1 - gamma) * (-((pos_scores - weight * neg_scores).sigmoid() + 1e-8).log()).mean()
        bpr_loss += gamma * (-((pos_scores - weight2 * neg_scores2).sigmoid() + 1e-8).log()).mean()

        # regularization loss
        u_ego_embeddings = self.user_id_embedding(user.to(self.device))
        pos_id_ego_embeddings = self.item_id_embedding(pos_item.to(self.device))
        neg_id_ego_embeddings = self.item_id_embedding(neg_item.to(self.device))
        neg_id_ego_embeddings2 = self.item_id_embedding(neg_item2.to(self.device))
        reg_loss = self.reg_loss(u_ego_embeddings, pos_id_ego_embeddings, neg_id_ego_embeddings, neg_id_ego_embeddings2)

        return bpr_loss + self.reg_weight * reg_loss + ssl_loss * self.ssl_reg, ssl_loss

    def predict(self):
        with torch.no_grad():
            user_all_embeddings, item_all_embeddings, h, *_ = self.forward(self.original_graph)
            if self.use_text:
                return user_all_embeddings, item_all_embeddings + h
            return user_all_embeddings, item_all_embeddings

    def edge_dropout(self, drop_rate):
        idx = torch.multinomial(self.edge_p, int((1 - drop_rate) * len(self.edge_p)))
        users, items = [self.users[i] for i in idx], [self.items[i] for i in idx]
        self.train_u_i_graph = self.get_norm_adj_mat(users, items).to(self.device)
