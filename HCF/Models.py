import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F

#This source file is based on the GRec published by Bo Li et al.
#We would like to thank and offer our appreciation to them.
#Original algorithm can be found in paper: Embedding App-Library Graph for Neural Third Party Library Recommendation. ESEC/FSE ’21

class HCF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, layer_num, dropout_list):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        self.n_layers = layer_num
        self.dropout_list = nn.ModuleList()
        #nn.ModuleList(),类似于容器，但是里面不一定按顺序存放，所以没有前向传播的功能，与list相比，它能识别到网络的参数

        torch.manual_seed(50)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self._init_weight_()
        for i in range(self.n_layers):
            self.dropout_list.append(nn.Dropout(dropout_list[i]))

    def _init_weight_(self):
        torch.manual_seed(50)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj_u1,adj_u2,adj_i1,adj_i2):

        hu = self.user_embedding.weight
        embedding=[hu]
        for i in range(self.n_layers):
            t=torch.sparse.mm(adj_u2,embedding[-1])
            # t=F.leaky_relu(torch.sparse.mm(adj_u2,embedding[-1]))
            t =torch.sparse.mm(adj_u1,t)
            # t=F.leaky_relu(torch.sparse.mm(adj_u1,t))
            # t=F.normalize(t, p=2, dim=1)
            # t = self.dropout_list[i](t)
            embedding.append(t)
        u_emb=torch.stack(embedding,dim=1)
        u_emb=torch.mean(u_emb, dim=1, keepdim=False)

        hi = self.item_embedding.weight
        embedding_i = [hi]
        for i in range(self.n_layers):
            t =torch.sparse.mm(adj_i2, embedding_i[-1])
            # t = F.leaky_relu(torch.sparse.mm(adj_i2, embedding_i[-1]))
            t = torch.sparse.mm(adj_i1, t)
            # t = F.leaky_relu(torch.sparse.mm(adj_i1, t))
            # t = F.normalize(t, p=2, dim=1)
            # t = self.dropout_list[i](t)
            embedding_i.append(t)
        i_emb = torch.stack(embedding_i, dim=1)
        i_emb = torch.mean(i_emb, dim=1, keepdim=False)
        return u_emb,i_emb
