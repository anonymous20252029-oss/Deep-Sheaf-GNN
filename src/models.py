import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SheafDiffusionConv(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh()
        )
        self.eps = nn.Parameter(torch.tensor(0.2))
        self.norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, h, edge_index, h0, alpha=0.2):
        row, col = edge_index
        edge_feat = torch.cat([h[row], h[col]], dim=-1)
        w = torch.sigmoid(self.edge_mlp(edge_feat))
        
        diff = w * (h[row] - h[col])
        out = torch.zeros_like(h)
        out.index_add_(0, row, diff)
        
        h_diff = h - self.eps * out
        h_next = (1.0 - alpha) * h_diff + alpha * h0
        return F.elu(self.norm(h_next))

class SheafGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth, dropout=0.5):
        super().__init__()
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.conv = SheafDiffusionConv(hidden_dim)
        self.depth = depth
        self.dropout = dropout
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, return_latent=False):
        h0 = F.dropout(F.elu(self.encoder(x)), p=self.dropout, training=self.training)
        h = h0
        for _ in range(self.depth):
            h = self.conv(h, edge_index, h0, alpha=0.2)
            h = F.dropout(h, p=self.dropout, training=self.training)
            
        latent = h.clone()
        out = self.classifier(h)
        return (out, latent) if return_latent else out

class BaselineGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(depth - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.dropout = dropout
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, return_latent=False):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        latent = x.clone()
        out = self.classifier(x)
        return (out, latent) if return_latent else out
