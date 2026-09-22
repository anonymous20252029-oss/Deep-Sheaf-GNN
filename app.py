import os
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.data import load_verified_benchmark
from src.models import SheafGNN

# Page setup
st.set_page_config(
    page_title="Deep Cellular Sheaf-GNN Dashboard",
    page_icon="🕸️",
    layout="wide"
)

st.title("🕸️ Deep Cellular Sheaf Neural Networks: Interactive Demo")
st.caption("ACIIDS 2027 Research Demo: Bounded Diffusion & Over-Smoothing Resilience on Heterophilic Graphs")

checkpoints_dir = "./checkpoints"
figures_dir = "./figures"
datasets = ['Wisconsin', 'Texas', 'Chameleon', 'Actor']

@st.cache_resource
def get_dataset(name):
    return load_verified_benchmark(name)

# Sidebar selector
selected_dataset = st.sidebar.selectbox("Select Heterophilic Benchmark:", datasets)

tab1, tab2, tab3 = st.tabs(["🔬 Interactive Node Inference", "📊 Experimental Results", "🖼️ Publication Figures"])

# TAB 1: Live Model Inference using Saved Checkpoints
with tab1:
    st.subheader(f"Node-Level Diagnostics on {selected_dataset} (K = 8)")
    
    ckpt_path = os.path.join(checkpoints_dir, f"sheaf_gnn_{selected_dataset.lower()}_k8.pt")
    
    if not os.path.exists(ckpt_path):
        st.error(f"Checkpoint not found at `{ckpt_path}`. Please verify your `checkpoints/` folder.")
    else:
        graph = get_dataset(selected_dataset)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        model = SheafGNN(graph.num_features, 64, checkpoint['num_classes'], depth=8)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        col_in, col_res = st.columns([1, 2])
        
        with col_in:
            node_id = st.number_input(
                f"Enter Node ID (0 to {graph.num_nodes - 1}):",
                min_value=0,
                max_value=graph.num_nodes - 1,
                value=0,
                step=1
            )
            
            with torch.no_grad():
                logits = model(graph.x, graph.edge_index)
                probs = F.softmax(logits[node_id], dim=-1).numpy()
                pred_class = int(np.argmax(probs))
            
            true_class = int(graph.y[node_id].item())
            degree = int((graph.edge_index[0] == node_id).sum().item())
            is_correct = (pred_class == true_class)
            
            st.metric("Prediction Status", "CORRECT" if is_correct else "MISCLASSIFIED", delta=f"Confidence: {probs[pred_class]*100:.1f}%")
            st.write(f"- **Ground Truth Class:** `{true_class}`")
            st.write(f"- **Predicted Class:** `{pred_class}`")
            st.write(f"- **Local Graph Degree:** `{degree}`")
        
        with col_res:
            fig, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
            bars = ax.bar(range(len(probs)), probs, color='#1f77b4', alpha=0.85, edgecolor='#0f4c81')
            ax.set_xlabel('Class Index', fontsize=10)
            ax.set_ylabel('Probability', fontsize=10)
            ax.set_title(f'Posterior Distribution (Node {node_id})', fontsize=11, fontweight='bold')
            ax.set_ylim(0, 1.0)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval*100:.1f}%', ha='center', va='bottom', fontsize=8)
                
            st.pyplot(fig)
            plt.close()

# TAB 2: Table Summary
with tab2:
    st.subheader("Benchmark Accuracy across Depths (K ∈ {2, 4, 8, 16})")
    benchmark_data = {
        "Dataset": ["Texas", "Texas", "Wisconsin", "Wisconsin", "Chameleon", "Chameleon", "Actor", "Actor"],
        "Model": ["Sheaf-GNN (Ours)", "Baseline GCN", "Sheaf-GNN (Ours)", "Baseline GCN",
                  "Sheaf-GNN (Ours)", "Baseline GCN", "Sheaf-GNN (Ours)", "Baseline GCN"],
        "K = 2": ["74.32 ± 5.57%", "57.30 ± 3.59%", "82.94 ± 3.83%", "53.73 ± 6.02%", "53.82 ± 1.88%", "39.19 ± 2.34%", "34.09 ± 0.79%", "28.43 ± 1.35%"],
        "K = 4": ["75.95 ± 6.45%", "57.57 ± 3.83%", "80.20 ± 5.07%", "53.92 ± 3.64%", "53.31 ± 1.92%", "33.99 ± 2.68%", "36.63 ± 0.87%", "27.20 ± 0.78%"],
        "K = 8": ["77.03 ± 5.16%", "58.11 ± 4.40%", "82.55 ± 4.59%", "53.92 ± 5.28%", "49.52 ± 1.66%", "31.69 ± 2.43%", "35.93 ± 1.14%", "25.82 ± 1.12%"],
        "K = 16": ["79.19 ± 4.69%", "58.65 ± 4.53%", "81.76 ± 3.62%", "50.59 ± 5.17%", "49.98 ± 2.29%", "29.98 ± 2.05%", "35.12 ± 1.37%", "25.11 ± 1.09%"]
    }
    st.dataframe(benchmark_data, use_container_width=True)

# TAB 3: Figures Inspector
with tab3:
    st.subheader("Research Figures (300 DPI)")
    fig_choice = st.selectbox("Choose Figure to Display:", [
        "Figure 1: Node Classification Accuracy vs. Depth",
        "Figure 2: Normalized Dirichlet Energy Spectrum (Log Scale)",
        "Figure 3: Deep Benchmark at K=8 Bar Chart",
        "Figure 4: Wisconsin Latent Embeddings (t-SNE)"
    ])
    
    mapping = {
        "Figure 1: Node Classification Accuracy vs. Depth": "Figure1_Accuracy_vs_Depth.png",
        "Figure 2: Normalized Dirichlet Energy Spectrum (Log Scale)": "Figure2_Dirichlet_Decay_Spectrum.png",
        "Figure 3: Deep Benchmark at K=8 Bar Chart": "Figure3_Benchmark_K8_BarChart.png",
        "Figure 4: Wisconsin Latent Embeddings (t-SNE)": "Figure4_Wisconsin_tSNE.png"
    }
    
    img_path = os.path.join(figures_dir, mapping[fig_choice])
    if os.path.exists(img_path):
        st.image(Image.open(img_path), caption=fig_choice, use_container_width=True)
    else:
        st.warning(f"File `{img_path}` not found.")
