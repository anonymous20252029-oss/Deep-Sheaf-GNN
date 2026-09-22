import torch
import torch.nn.functional as F

def compute_normalized_dirichlet_energy(edge_index, X):
    """
    Computes normalized Dirichlet energy E(X) to quantify over-smoothing:
    E(X) = (1 / |E|) * sum_{(u,v) in E} || X_u / ||X_u||_2 - X_v / ||X_v||_2 ||_2^2
    """
    row, col = edge_index
    X_norm = F.normalize(X, p=2, dim=-1)
    diff = X_norm[row] - X_norm[col]
    energy = torch.sum(diff ** 2, dim=-1).mean().item()
    return max(float(energy), 1e-6)
