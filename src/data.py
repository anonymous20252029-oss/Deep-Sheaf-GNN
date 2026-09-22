import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor as ActorDS

def load_verified_benchmark(name, root_dir="/tmp/pyg_data"):
    key = name.lower()
    dataset_dir = f"{root_dir}/{key}"

    if key == 'texas':
        ds = WebKB(root=dataset_dir, name='Texas')
    elif key == 'wisconsin':
        ds = WebKB(root=dataset_dir, name='Wisconsin')
    elif key == 'chameleon':
        try:
            ds = WikipediaNetwork(root=dataset_dir, name='chameleon', geom_gcn_preprocess=True)
        except Exception:
            ds = WikipediaNetwork(root=dataset_dir, name='chameleon', geom_gcn_preprocess=False)
    elif key == 'actor':
        ds = ActorDS(root=dataset_dir)
    else:
        raise ValueError(f"Unrecognized dataset: {name}")

    d = ds[0]

    # Standardize train/val/test masks to [N, 10] boolean tensors
    if hasattr(d, 'train_mask') and d.train_mask is not None and d.train_mask.dim() > 1:
        train_masks = d.train_mask.bool()
        val_masks = d.val_mask.bool()
        test_masks = d.test_mask.bool()
    else:
        n = d.num_nodes
        tm, vm, tsm = [], [], []
        for s in range(10):
            torch.manual_seed(42 + s)
            perm = torch.randperm(n)
            t_m = torch.zeros(n, dtype=torch.bool)
            v_m = torch.zeros(n, dtype=torch.bool)
            te_m = torch.zeros(n, dtype=torch.bool)
            t_m[perm[:int(0.6 * n)]] = True
            v_m[perm[int(0.6 * n):int(0.8 * n)]] = True
            te_m[perm[int(0.8 * n):]] = True
            tm.append(t_m)
            vm.append(v_m)
            tsm.append(te_m)
        train_masks = torch.stack(tm, dim=1)
        val_masks = torch.stack(vm, dim=1)
        test_masks = torch.stack(tsm, dim=1)

    edges, _ = add_self_loops(d.edge_index, num_nodes=d.num_nodes)
    return Data(x=d.x, edge_index=edges, y=d.y,
                train_mask=train_masks, val_mask=val_masks, test_mask=test_masks)
