import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import gradio as gr

from src.data import load_verified_benchmark
from src.models import SheafGNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoints_dir = "./checkpoints"
datasets = ['Texas', 'Wisconsin', 'Chameleon', 'Actor']

# Pre-cache dataset representations
datasets_dict = {d: load_verified_benchmark(d) for d in datasets}

def inspect_node_prediction(dataset_name, node_id):
    node_id = int(node_id)
    graph = datasets_dict[dataset_name]

    if node_id < 0 or node_id >= graph.num_nodes:
        return f"Error: Node ID must be between 0 and {graph.num_nodes - 1}", None

    ckpt_file = os.path.join(checkpoints_dir, f"sheaf_gnn_{dataset_name.lower()}_k8.pt")
    if not os.path.exists(ckpt_file):
        return f"Checkpoint {ckpt_file} not found. Please train models first.", None

    checkpoint = torch.load(ckpt_file, map_location=device)
    model = SheafGNN(graph.num_features, 64, checkpoint['num_classes'], depth=8).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        logits = model(graph.x.to(device), graph.edge_index.to(device))
        probs = F.softmax(logits[node_id], dim=-1).cpu().numpy()
        pred_class = int(np.argmax(probs))

    true_class = int(graph.y[node_id].item())
    degree = int((graph.edge_index[0] == node_id).sum().item())

    report = (
        f"Dataset: {dataset_name}\n"
        f"Target Node: {node_id}\n"
        f"Node Degree: {degree}\n"
        f"Ground Truth Class: {true_class}\n"
        f"Predicted Class: {pred_class}\n"
        f"Confidence: {probs[pred_class] * 100:.2f}%\n"
        f"Diagnostic Status: {'CORRECT' if pred_class == true_class else 'MISCLASSIFIED'}"
    )

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(range(len(probs)), probs, color='#1f77b4', alpha=0.85)
    ax.set_xlabel('Class Label')
    ax.set_ylabel('Probability')
    ax.set_title(f'Posterior Probabilities (Node {node_id})')
    ax.set_ylim(0, 1.0)
    plt.tight_layout()

    return report, fig

demo = gr.Interface(
    fn=inspect_node_prediction,
    inputs=[
        gr.Dropdown(choices=datasets, value='Wisconsin', label="Select Benchmark Dataset"),
        gr.Number(value=0, precision=0, label="Target Node ID")
    ],
    outputs=[
        gr.Textbox(label="Diagnostic Information", lines=8),
        gr.Plot(label="Prediction Chart")
    ],
    title="Cellular Sheaf-GNN Interactive Inspection Dashboard",
    description="Inspect node predictions and class posteriors for deep Sheaf-GNN (K=8)."
)

if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=7860)
