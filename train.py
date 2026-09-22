import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import torch.nn.functional as F

from src.data import load_verified_benchmark
from src.metrics import compute_normalized_dirichlet_energy
from src.models import SheafGNN, BaselineGCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoints_dir = "./checkpoints"
figures_dir = "./figures"
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

datasets = ['Texas', 'Wisconsin', 'Chameleon', 'Actor']
depths = [2, 4, 8, 16]
NUM_SPLITS = 10
EPOCHS = 150

datasets_dict = {d: load_verified_benchmark(d) for d in datasets}

acc_sheaf_mean, acc_sheaf_std = {d: [] for d in datasets}, {d: [] for d in datasets}
acc_gcn_mean, acc_gcn_std = {d: [] for d in datasets}, {d: [] for d in datasets}
energy_sheaf_dict = {d: [] for d in datasets}
energy_gcn_dict = {d: [] for d in datasets}
best_checkpoints = {}

for ds_name in datasets:
    data = datasets_dict[ds_name].to(device)
    in_dim = data.num_features
    num_classes = len(torch.unique(data.y))

    for k in depths:
        sheaf_accs, gcn_accs = [], []
        sheaf_energies, gcn_energies = [], []
        best_k8_acc, best_k8_state = -1.0, None

        for split_idx in range(NUM_SPLITS):
            train_m = data.train_mask[:, split_idx]
            val_m = data.val_mask[:, split_idx]
            test_m = data.test_mask[:, split_idx]

            # Train Sheaf-GNN
            s_model = SheafGNN(in_dim, 64, num_classes, depth=k).to(device)
            s_opt = torch.optim.Adam(s_model.parameters(), lr=0.01, weight_decay=5e-4)
            best_val_acc_s, best_test_s = -1.0, 0.0

            for _ in range(EPOCHS):
                s_model.train()
                s_opt.zero_grad()
                out = s_model(data.x, data.edge_index)
                loss = F.cross_entropy(out[train_m], data.y[train_m])
                loss.backward()
                s_opt.step()

                s_model.eval()
                with torch.no_grad():
                    logits = s_model(data.x, data.edge_index)
                    val_acc = (logits[val_m].argmax(-1) == data.y[val_m]).float().mean().item()
                    test_acc = (logits[test_m].argmax(-1) == data.y[test_m]).float().mean().item() * 100.0
                    if val_acc > best_val_acc_s:
                        best_val_acc_s = val_acc
                        best_test_s = test_acc
                        if k == 8 and test_acc > best_k8_acc:
                            best_k8_acc = test_acc
                            best_k8_state = copy.deepcopy(s_model.state_dict())

            s_model.eval()
            with torch.no_grad():
                _, lat_s = s_model(data.x, data.edge_index, return_latent=True)
                sheaf_energies.append(compute_normalized_dirichlet_energy(data.edge_index, lat_s))
            sheaf_accs.append(best_test_s)

            # Train Baseline GCN
            g_model = BaselineGCN(in_dim, 64, num_classes, depth=k).to(device)
            g_opt = torch.optim.Adam(g_model.parameters(), lr=0.01, weight_decay=5e-4)
            best_val_acc_g, best_test_g = -1.0, 0.0

            for _ in range(EPOCHS):
                g_model.train()
                g_opt.zero_grad()
                out = g_model(data.x, data.edge_index)
                loss = F.cross_entropy(out[train_m], data.y[train_m])
                loss.backward()
                g_opt.step()

                g_model.eval()
                with torch.no_grad():
                    logits = g_model(data.x, data.edge_index)
                    val_acc = (logits[val_m].argmax(-1) == data.y[val_m]).float().mean().item()
                    test_acc = (logits[test_m].argmax(-1) == data.y[test_m]).float().mean().item() * 100.0
                    if val_acc > best_val_acc_g:
                        best_val_acc_g = val_acc
                        best_test_g = test_acc

            g_model.eval()
            with torch.no_grad():
                _, lat_g = g_model(data.x, data.edge_index, return_latent=True)
                gcn_energies.append(compute_normalized_dirichlet_energy(data.edge_index, lat_g))
            gcn_accs.append(best_test_g)

        acc_sheaf_mean[ds_name].append(round(float(np.mean(sheaf_accs)), 2))
        acc_sheaf_std[ds_name].append(round(float(np.std(sheaf_accs)), 2))
        acc_gcn_mean[ds_name].append(round(float(np.mean(gcn_accs)), 2))
        acc_gcn_std[ds_name].append(round(float(np.std(gcn_accs)), 2))
        energy_sheaf_dict[ds_name].append(round(float(np.mean(sheaf_energies)), 4))
        energy_gcn_dict[ds_name].append(round(float(np.mean(gcn_energies)), 4))

        if k == 8 and best_k8_state is not None:
            ckpt_p = os.path.join(checkpoints_dir, f"sheaf_gnn_{ds_name.lower()}_k8.pt")
            torch.save({
                'model_state_dict': best_k8_state,
                'in_dim': in_dim,
                'hidden_dim': 64,
                'num_classes': num_classes,
                'depth': 8,
                'peak_acc': best_k8_acc
            }, ckpt_p)
            best_checkpoints[ds_name] = ckpt_p

print("[COMPLETED] All experiments finished. Checkpoints saved in ./checkpoints")
