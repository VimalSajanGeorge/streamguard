#!/usr/bin/env python3
"""
Script to add Transformer feature/attention analysis and GNN embeddings visualization cells
to the StreamGuard Production Training notebook.
"""
import json
import sys

def create_transformer_viz_cells():
    """Create cells for Transformer feature & attention analysis"""
    cells = []

    # Cell 1: Transformer Feature & Attention Analysis Header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 1.5: Transformer Feature & Attention Analysis\n",
            "\n",
            "Visualize transformer model internals:\n",
            "- Feature importance from trained models\n",
            "- Attention weights and patterns\n",
            "- Embedding space visualization (t-SNE/UMAP)\n",
            "- Token-level attention heatmaps"
        ]
    })

    # Cell 2: Load Transformer Model & Data
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 15.6: Load Transformer Model for Analysis\n",
            "import torch\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from pathlib import Path\n",
            "from sklearn.manifold import TSNE\n",
            "from umap import UMAP\n",
            "import json\n",
            "\n",
            "# Configuration\n",
            "TRANSFORMER_OUTPUT = Path('training/outputs/transformer_v17')\n",
            "BEST_SEED = 42  # Change to best performing seed\n",
            "\n",
            "# Load best model\n",
            "model_path = TRANSFORMER_OUTPUT / f'seed_{BEST_SEED}' / 'checkpoints' / 'best_model.pt'\n",
            "\n",
            "if model_path.exists():\n",
            "    print(f'[+] Loading transformer model from {model_path}')\n",
            "    checkpoint = torch.load(model_path, map_location='cpu')\n",
            "    print(f'[+] Model from epoch {checkpoint.get(\"epoch\", \"?\")}')\n",
            "    print(f'[+] Best F1: {checkpoint.get(\"best_f1\", \"?\"):.4f}')\n",
            "else:\n",
            "    print(f'[!] Model not found at {model_path}')\n",
            "    print('[!] Please run transformer training first')"
        ]
    })

    # Cell 3: Extract Embeddings
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 15.7: Extract Embeddings from Validation Set\n",
            "from torch.utils.data import DataLoader\n",
            "from training.datasets.sql_transformer_dataset import SQLTransformerDataset\n",
            "\n",
            "def extract_embeddings(model, dataloader, device='cpu', max_samples=1000):\n",
            "    \"\"\"\n",
            "    Extract final layer embeddings from transformer model.\n",
            "    \n",
            "    Args:\n",
            "        model: Trained transformer model\n",
            "        dataloader: DataLoader for validation data\n",
            "        device: Device to run on\n",
            "        max_samples: Maximum samples to extract (for memory efficiency)\n",
            "    \n",
            "    Returns:\n",
            "        embeddings: numpy array of shape (n_samples, embedding_dim)\n",
            "        labels: numpy array of shape (n_samples,)\n",
            "        predictions: numpy array of shape (n_samples,)\n",
            "    \"\"\"\n",
            "    model.eval()\n",
            "    model.to(device)\n",
            "    \n",
            "    embeddings_list = []\n",
            "    labels_list = []\n",
            "    predictions_list = []\n",
            "    \n",
            "    total = 0\n",
            "    with torch.no_grad():\n",
            "        for batch in dataloader:\n",
            "            if total >= max_samples:\n",
            "                break\n",
            "            \n",
            "            input_ids = batch['input_ids'].to(device)\n",
            "            attention_mask = batch['attention_mask'].to(device)\n",
            "            labels = batch['labels']\n",
            "            \n",
            "            # Forward pass to get embeddings (access last hidden state)\n",
            "            outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)\n",
            "            \n",
            "            # Use CLS token embedding from last layer\n",
            "            cls_embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()\n",
            "            \n",
            "            # Get predictions\n",
            "            logits = model(input_ids=input_ids, attention_mask=attention_mask)\n",
            "            preds = torch.argmax(logits, dim=1).cpu().numpy()\n",
            "            \n",
            "            embeddings_list.append(cls_embeddings)\n",
            "            labels_list.append(labels.numpy())\n",
            "            predictions_list.append(preds)\n",
            "            \n",
            "            total += len(labels)\n",
            "    \n",
            "    embeddings = np.vstack(embeddings_list)\n",
            "    labels = np.concatenate(labels_list)\n",
            "    predictions = np.concatenate(predictions_list)\n",
            "    \n",
            "    print(f'[+] Extracted {len(embeddings)} embeddings')\n",
            "    print(f'    Shape: {embeddings.shape}')\n",
            "    print(f'    Vulnerable: {(labels == 1).sum()} ({(labels == 1).mean()*100:.1f}%)')\n",
            "    print(f'    Safe: {(labels == 0).sum()} ({(labels == 0).mean()*100:.1f}%)')\n",
            "    \n",
            "    return embeddings, labels, predictions\n",
            "\n",
            "# Try to extract embeddings if model is loaded\n",
            "if 'checkpoint' in locals() and model_path.exists():\n",
            "    # Load validation dataset\n",
            "    val_data_path = Path('data/processed/sql_injection/val/data.jsonl')\n",
            "    if val_data_path.exists():\n",
            "        val_dataset = SQLTransformerDataset(\n",
            "            data_file=str(val_data_path),\n",
            "            tokenizer_name='microsoft/graphcodebert-base',\n",
            "            max_seq_len=512\n",
            "        )\n",
            "        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)\n",
            "        \n",
            "        # Extract embeddings\n",
            "        embeddings, labels, predictions = extract_embeddings(\n",
            "            model=checkpoint['model'],  # Adjust based on checkpoint structure\n",
            "            dataloader=val_loader,\n",
            "            max_samples=1000\n",
            "        )\n",
            "    else:\n",
            "        print(f'[!] Validation data not found at {val_data_path}')\n",
            "else:\n",
            "    print('[!] Skipping embedding extraction - model not loaded')"
        ]
    })

    # Cell 4: t-SNE/UMAP Visualization
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 15.8: Embedding Space Visualization (t-SNE & UMAP)\n",
            "\n",
            "if 'embeddings' in locals():\n",
            "    fig, axes = plt.subplots(1, 2, figsize=(16, 7))\n",
            "    \n",
            "    # t-SNE\n",
            "    print('[*] Computing t-SNE...')\n",
            "    tsne = TSNE(n_components=2, random_state=42, perplexity=30)\n",
            "    embeddings_tsne = tsne.fit_transform(embeddings)\n",
            "    \n",
            "    # UMAP\n",
            "    print('[*] Computing UMAP...')\n",
            "    umap = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)\n",
            "    embeddings_umap = umap.fit_transform(embeddings)\n",
            "    \n",
            "    # Plot t-SNE\n",
            "    ax1 = axes[0]\n",
            "    scatter1 = ax1.scatter(\n",
            "        embeddings_tsne[:, 0],\n",
            "        embeddings_tsne[:, 1],\n",
            "        c=labels,\n",
            "        cmap='RdYlGn_r',\n",
            "        alpha=0.6,\n",
            "        s=20,\n",
            "        edgecolors='none'\n",
            "    )\n",
            "    ax1.set_title('t-SNE: Transformer Embeddings', fontsize=14, fontweight='bold')\n",
            "    ax1.set_xlabel('t-SNE Dimension 1')\n",
            "    ax1.set_ylabel('t-SNE Dimension 2')\n",
            "    ax1.legend(*scatter1.legend_elements(), title='Labels', labels=['Safe', 'Vulnerable'])\n",
            "    ax1.grid(True, alpha=0.3)\n",
            "    \n",
            "    # Plot UMAP\n",
            "    ax2 = axes[1]\n",
            "    scatter2 = ax2.scatter(\n",
            "        embeddings_umap[:, 0],\n",
            "        embeddings_umap[:, 1],\n",
            "        c=labels,\n",
            "        cmap='RdYlGn_r',\n",
            "        alpha=0.6,\n",
            "        s=20,\n",
            "        edgecolors='none'\n",
            "    )\n",
            "    ax2.set_title('UMAP: Transformer Embeddings', fontsize=14, fontweight='bold')\n",
            "    ax2.set_xlabel('UMAP Dimension 1')\n",
            "    ax2.set_ylabel('UMAP Dimension 2')\n",
            "    ax2.legend(*scatter2.legend_elements(), title='Labels', labels=['Safe', 'Vulnerable'])\n",
            "    ax2.grid(True, alpha=0.3)\n",
            "    \n",
            "    plt.tight_layout()\n",
            "    plt.savefig(TRANSFORMER_OUTPUT / 'embeddings_visualization.png', dpi=300, bbox_inches='tight')\n",
            "    plt.show()\n",
            "    \n",
            "    print(f'[+] Saved visualization to {TRANSFORMER_OUTPUT / \"embeddings_visualization.png\"}')\n",
            "    \n",
            "    # Calculate separation metrics\n",
            "    from sklearn.metrics import silhouette_score, calinski_harabasz_score\n",
            "    \n",
            "    tsne_silhouette = silhouette_score(embeddings_tsne, labels)\n",
            "    umap_silhouette = silhouette_score(embeddings_umap, labels)\n",
            "    tsne_ch = calinski_harabasz_score(embeddings_tsne, labels)\n",
            "    umap_ch = calinski_harabasz_score(embeddings_umap, labels)\n",
            "    \n",
            "    print('\\n[*] Embedding Space Separation Metrics:')\n",
            "    print(f'    t-SNE Silhouette Score: {tsne_silhouette:.4f}')\n",
            "    print(f'    UMAP Silhouette Score: {umap_silhouette:.4f}')\n",
            "    print(f'    t-SNE Calinski-Harabasz: {tsne_ch:.2f}')\n",
            "    print(f'    UMAP Calinski-Harabasz: {umap_ch:.2f}')\n",
            "else:\n",
            "    print('[!] Embeddings not available - run previous cell first')"
        ]
    })

    # Cell 5: Attention Weights (if accessible)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 15.9: Attention Weights Visualization (Example)\n",
            "# Note: This requires the model to output attention weights during forward pass\n",
            "\n",
            "def visualize_attention(tokens, attention_weights, layer=0, head=0):\n",
            "    \"\"\"\n",
            "    Visualize attention weights for a specific layer and head.\n",
            "    \n",
            "    Args:\n",
            "        tokens: List of tokens\n",
            "        attention_weights: Attention tensor of shape (layers, heads, seq_len, seq_len)\n",
            "        layer: Which layer to visualize\n",
            "        head: Which attention head to visualize\n",
            "    \"\"\"\n",
            "    att = attention_weights[layer, head].cpu().numpy()\n",
            "    \n",
            "    plt.figure(figsize=(12, 10))\n",
            "    sns.heatmap(\n",
            "        att,\n",
            "        xticklabels=tokens,\n",
            "        yticklabels=tokens,\n",
            "        cmap='YlOrRd',\n",
            "        cbar_kws={'label': 'Attention Weight'},\n",
            "        square=True\n",
            "    )\n",
            "    plt.title(f'Attention Weights - Layer {layer}, Head {head}', fontsize=14, fontweight='bold')\n",
            "    plt.xlabel('Key Tokens')\n",
            "    plt.ylabel('Query Tokens')\n",
            "    plt.xticks(rotation=45, ha='right')\n",
            "    plt.yticks(rotation=0)\n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "\n",
            "print('[*] Attention visualization function defined')\n",
            "print('[!] Note: To use this, you need to modify the model forward pass to return attention weights')\n",
            "print('[!] Add output_attentions=True to the model call and extract from outputs.attentions')\n",
            "print()\n",
            "print('Example usage:')\n",
            "print('  outputs = model(input_ids=ids, attention_mask=mask, output_attentions=True)')\n",
            "print('  attentions = outputs.attentions  # Tuple of attention tensors')\n",
            "print('  visualize_attention(tokens, attentions[0], layer=0, head=0)')"
        ]
    })

    return cells

def create_gnn_viz_cells():
    """Create cells for GNN embeddings visualization"""
    cells = []

    # Cell 1: GNN Embeddings Analysis Header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 2.5: GNN Embeddings Visualization\n",
            "\n",
            "Visualize GNN graph-level embeddings:\n",
            "- Graph embedding space (t-SNE/UMAP)\n",
            "- Vulnerable vs Safe separation\n",
            "- Cluster analysis\n",
            "- Multi-seed embedding comparison"
        ]
    })

    # Cell 2: Load GNN Models & Extract Embeddings
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 17.5: Extract GNN Graph Embeddings\n",
            "import torch\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from pathlib import Path\n",
            "from torch_geometric.loader import DataLoader as GeoDataLoader\n",
            "from torch_geometric.data import Data\n",
            "from sklearn.manifold import TSNE\n",
            "from umap import UMAP\n",
            "import json\n",
            "\n",
            "def load_graph_dataset(data_dir, max_graphs=None):\n",
            "    \"\"\"Load graph dataset from directory.\"\"\"\n",
            "    data_dir = Path(data_dir)\n",
            "    graph_files = sorted(list(data_dir.glob('*.pt')))\n",
            "    \n",
            "    if max_graphs:\n",
            "        graph_files = graph_files[:max_graphs]\n",
            "    \n",
            "    graphs = []\n",
            "    for f in graph_files:\n",
            "        try:\n",
            "            g = torch.load(f, map_location='cpu')\n",
            "            graphs.append(g)\n",
            "        except Exception as e:\n",
            "            print(f'[!] Failed to load {f}: {e}')\n",
            "    \n",
            "    print(f'[+] Loaded {len(graphs)} graphs from {data_dir}')\n",
            "    return graphs\n",
            "\n",
            "def extract_gnn_embeddings(model, dataloader, device='cpu'):\n",
            "    \"\"\"\n",
            "    Extract graph-level embeddings from GNN model.\n",
            "    \n",
            "    Args:\n",
            "        model: Trained GNN model\n",
            "        dataloader: GeoDataLoader for graph data\n",
            "        device: Device to run on\n",
            "    \n",
            "    Returns:\n",
            "        embeddings: numpy array of shape (n_graphs, embedding_dim)\n",
            "        labels: numpy array of shape (n_graphs,)\n",
            "        predictions: numpy array of shape (n_graphs,)\n",
            "    \"\"\"\n",
            "    model.eval()\n",
            "    model.to(device)\n",
            "    \n",
            "    embeddings_list = []\n",
            "    labels_list = []\n",
            "    predictions_list = []\n",
            "    \n",
            "    with torch.no_grad():\n",
            "        for batch in dataloader:\n",
            "            batch = batch.to(device)\n",
            "            \n",
            "            # Forward pass to get embeddings\n",
            "            # Assuming model has a method to get graph embeddings before classification\n",
            "            # You may need to modify this based on your model architecture\n",
            "            try:\n",
            "                # Try to get embeddings from model\n",
            "                if hasattr(model, 'get_embeddings'):\n",
            "                    graph_emb = model.get_embeddings(batch.x, batch.edge_index, batch.batch)\n",
            "                else:\n",
            "                    # Fallback: run full forward and extract from last layer\n",
            "                    logits = model(batch.x, batch.edge_index, batch.batch)\n",
            "                    # Use logits as embeddings (not ideal but works)\n",
            "                    graph_emb = logits\n",
            "                \n",
            "                graph_emb = graph_emb.cpu().numpy()\n",
            "                \n",
            "                # Get predictions\n",
            "                logits = model(batch.x, batch.edge_index, batch.batch)\n",
            "                preds = torch.argmax(logits, dim=1).cpu().numpy()\n",
            "                \n",
            "                labels = batch.y.cpu().numpy()\n",
            "                \n",
            "                embeddings_list.append(graph_emb)\n",
            "                labels_list.append(labels)\n",
            "                predictions_list.append(preds)\n",
            "            except Exception as e:\n",
            "                print(f'[!] Error extracting embeddings: {e}')\n",
            "                continue\n",
            "    \n",
            "    embeddings = np.vstack(embeddings_list)\n",
            "    labels = np.concatenate(labels_list)\n",
            "    predictions = np.concatenate(predictions_list)\n",
            "    \n",
            "    print(f'[+] Extracted {len(embeddings)} graph embeddings')\n",
            "    print(f'    Shape: {embeddings.shape}')\n",
            "    print(f'    Vulnerable: {(labels == 1).sum()} ({(labels == 1).mean()*100:.1f}%)')\n",
            "    print(f'    Safe: {(labels == 0).sum()} ({(labels == 0).mean()*100:.1f}%)')\n",
            "    \n",
            "    return embeddings, labels, predictions\n",
            "\n",
            "# Configuration\n",
            "GNN_OUTPUT = Path('training/outputs/gnn_v17')\n",
            "BEST_SEED = 42  # Change to best performing seed\n",
            "\n",
            "# Load best GNN model\n",
            "model_path = GNN_OUTPUT / f'seed_{BEST_SEED}' / 'checkpoints' / 'best_model.pt'\n",
            "\n",
            "if model_path.exists():\n",
            "    print(f'[+] Loading GNN model from {model_path}')\n",
            "    checkpoint = torch.load(model_path, map_location='cpu')\n",
            "    print(f'[+] Model from epoch {checkpoint.get(\"epoch\", \"?\")}')\n",
            "    print(f'[+] Best F1: {checkpoint.get(\"best_f1\", \"?\"):.4f}')\n",
            "    \n",
            "    # Load validation graphs\n",
            "    val_graphs = load_graph_dataset('data/processed/graphs/val', max_graphs=500)\n",
            "    val_loader = GeoDataLoader(val_graphs, batch_size=32, shuffle=False)\n",
            "    \n",
            "    # Extract embeddings\n",
            "    try:\n",
            "        gnn_embeddings, gnn_labels, gnn_predictions = extract_gnn_embeddings(\n",
            "            model=checkpoint['model'],  # Adjust based on checkpoint structure\n",
            "            dataloader=val_loader\n",
            "        )\n",
            "    except Exception as e:\n",
            "        print(f'[!] Failed to extract embeddings: {e}')\n",
            "        print('[!] You may need to modify extract_gnn_embeddings() based on your model architecture')\n",
            "else:\n",
            "    print(f'[!] Model not found at {model_path}')\n",
            "    print('[!] Please run GNN training first')"
        ]
    })

    # Cell 3: GNN Embeddings Visualization
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 17.6: GNN Embedding Space Visualization (t-SNE & UMAP)\n",
            "\n",
            "if 'gnn_embeddings' in locals():\n",
            "    fig, axes = plt.subplots(1, 2, figsize=(16, 7))\n",
            "    \n",
            "    # t-SNE\n",
            "    print('[*] Computing t-SNE for GNN embeddings...')\n",
            "    tsne = TSNE(n_components=2, random_state=42, perplexity=30)\n",
            "    gnn_tsne = tsne.fit_transform(gnn_embeddings)\n",
            "    \n",
            "    # UMAP\n",
            "    print('[*] Computing UMAP for GNN embeddings...')\n",
            "    umap = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)\n",
            "    gnn_umap = umap.fit_transform(gnn_embeddings)\n",
            "    \n",
            "    # Plot t-SNE\n",
            "    ax1 = axes[0]\n",
            "    scatter1 = ax1.scatter(\n",
            "        gnn_tsne[:, 0],\n",
            "        gnn_tsne[:, 1],\n",
            "        c=gnn_labels,\n",
            "        cmap='RdYlGn_r',\n",
            "        alpha=0.6,\n",
            "        s=30,\n",
            "        edgecolors='black',\n",
            "        linewidth=0.5\n",
            "    )\n",
            "    ax1.set_title('t-SNE: GNN Graph Embeddings', fontsize=14, fontweight='bold')\n",
            "    ax1.set_xlabel('t-SNE Dimension 1')\n",
            "    ax1.set_ylabel('t-SNE Dimension 2')\n",
            "    ax1.legend(*scatter1.legend_elements(), title='Labels', labels=['Safe', 'Vulnerable'])\n",
            "    ax1.grid(True, alpha=0.3)\n",
            "    \n",
            "    # Plot UMAP\n",
            "    ax2 = axes[1]\n",
            "    scatter2 = ax2.scatter(\n",
            "        gnn_umap[:, 0],\n",
            "        gnn_umap[:, 1],\n",
            "        c=gnn_labels,\n",
            "        cmap='RdYlGn_r',\n",
            "        alpha=0.6,\n",
            "        s=30,\n",
            "        edgecolors='black',\n",
            "        linewidth=0.5\n",
            "    )\n",
            "    ax2.set_title('UMAP: GNN Graph Embeddings', fontsize=14, fontweight='bold')\n",
            "    ax2.set_xlabel('UMAP Dimension 1')\n",
            "    ax2.set_ylabel('UMAP Dimension 2')\n",
            "    ax2.legend(*scatter2.legend_elements(), title='Labels', labels=['Safe', 'Vulnerable'])\n",
            "    ax2.grid(True, alpha=0.3)\n",
            "    \n",
            "    plt.tight_layout()\n",
            "    plt.savefig(GNN_OUTPUT / 'graph_embeddings_visualization.png', dpi=300, bbox_inches='tight')\n",
            "    plt.show()\n",
            "    \n",
            "    print(f'[+] Saved visualization to {GNN_OUTPUT / \"graph_embeddings_visualization.png\"}')\n",
            "    \n",
            "    # Calculate separation metrics\n",
            "    from sklearn.metrics import silhouette_score, calinski_harabasz_score\n",
            "    \n",
            "    tsne_silhouette = silhouette_score(gnn_tsne, gnn_labels)\n",
            "    umap_silhouette = silhouette_score(gnn_umap, gnn_labels)\n",
            "    tsne_ch = calinski_harabasz_score(gnn_tsne, gnn_labels)\n",
            "    umap_ch = calinski_harabasz_score(gnn_umap, gnn_labels)\n",
            "    \n",
            "    print('\\n[*] GNN Embedding Space Separation Metrics:')\n",
            "    print(f'    t-SNE Silhouette Score: {tsne_silhouette:.4f}')\n",
            "    print(f'    UMAP Silhouette Score: {umap_silhouette:.4f}')\n",
            "    print(f'    t-SNE Calinski-Harabasz: {tsne_ch:.2f}')\n",
            "    print(f'    UMAP Calinski-Harabasz: {umap_ch:.2f}')\n",
            "    \n",
            "    # Prediction accuracy\n",
            "    accuracy = (gnn_predictions == gnn_labels).mean()\n",
            "    print(f'\\n[*] Prediction Accuracy: {accuracy*100:.2f}%')\n",
            "else:\n",
            "    print('[!] GNN embeddings not available - run previous cell first')"
        ]
    })

    # Cell 4: Misclassification Analysis
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 17.7: GNN Misclassification Analysis in Embedding Space\n",
            "\n",
            "if 'gnn_embeddings' in locals() and 'gnn_umap' in locals():\n",
            "    # Identify misclassified samples\n",
            "    correct = gnn_predictions == gnn_labels\n",
            "    misclassified = ~correct\n",
            "    \n",
            "    print(f'[*] Total samples: {len(gnn_labels)}')\n",
            "    print(f'[*] Correct: {correct.sum()} ({correct.mean()*100:.2f}%)')\n",
            "    print(f'[*] Misclassified: {misclassified.sum()} ({misclassified.mean()*100:.2f}%)')\n",
            "    \n",
            "    # Visualize misclassified samples in embedding space\n",
            "    fig, axes = plt.subplots(1, 2, figsize=(16, 7))\n",
            "    \n",
            "    # UMAP with correct/incorrect labels\n",
            "    ax1 = axes[0]\n",
            "    ax1.scatter(\n",
            "        gnn_umap[correct, 0],\n",
            "        gnn_umap[correct, 1],\n",
            "        c=gnn_labels[correct],\n",
            "        cmap='RdYlGn_r',\n",
            "        alpha=0.3,\n",
            "        s=20,\n",
            "        label='Correct'\n",
            "    )\n",
            "    ax1.scatter(\n",
            "        gnn_umap[misclassified, 0],\n",
            "        gnn_umap[misclassified, 1],\n",
            "        c='red',\n",
            "        marker='x',\n",
            "        s=100,\n",
            "        alpha=0.8,\n",
            "        linewidth=2,\n",
            "        label='Misclassified'\n",
            "    )\n",
            "    ax1.set_title('Misclassified Samples in Embedding Space', fontsize=14, fontweight='bold')\n",
            "    ax1.set_xlabel('UMAP Dimension 1')\n",
            "    ax1.set_ylabel('UMAP Dimension 2')\n",
            "    ax1.legend()\n",
            "    ax1.grid(True, alpha=0.3)\n",
            "    \n",
            "    # Confusion by class\n",
            "    ax2 = axes[1]\n",
            "    \n",
            "    # False positives and false negatives\n",
            "    fp = (gnn_predictions == 1) & (gnn_labels == 0)  # Predicted vuln, actually safe\n",
            "    fn = (gnn_predictions == 0) & (gnn_labels == 1)  # Predicted safe, actually vuln\n",
            "    \n",
            "    ax2.scatter(\n",
            "        gnn_umap[~(fp | fn), 0],\n",
            "        gnn_umap[~(fp | fn), 1],\n",
            "        c='gray',\n",
            "        alpha=0.2,\n",
            "        s=20,\n",
            "        label='Correct'\n",
            "    )\n",
            "    ax2.scatter(\n",
            "        gnn_umap[fp, 0],\n",
            "        gnn_umap[fp, 1],\n",
            "        c='orange',\n",
            "        marker='^',\n",
            "        s=80,\n",
            "        alpha=0.8,\n",
            "        label=f'False Pos ({fp.sum()})'\n",
            "    )\n",
            "    ax2.scatter(\n",
            "        gnn_umap[fn, 0],\n",
            "        gnn_umap[fn, 1],\n",
            "        c='red',\n",
            "        marker='v',\n",
            "        s=80,\n",
            "        alpha=0.8,\n",
            "        label=f'False Neg ({fn.sum()})'\n",
            "    )\n",
            "    ax2.set_title('False Positives vs False Negatives', fontsize=14, fontweight='bold')\n",
            "    ax2.set_xlabel('UMAP Dimension 1')\n",
            "    ax2.set_ylabel('UMAP Dimension 2')\n",
            "    ax2.legend()\n",
            "    ax2.grid(True, alpha=0.3)\n",
            "    \n",
            "    plt.tight_layout()\n",
            "    plt.savefig(GNN_OUTPUT / 'misclassification_analysis.png', dpi=300, bbox_inches='tight')\n",
            "    plt.show()\n",
            "    \n",
            "    print(f'[+] Saved to {GNN_OUTPUT / \"misclassification_analysis.png\"}')\n",
            "else:\n",
            "    print('[!] Run previous cells first to extract embeddings')"
        ]
    })

    return cells

def add_cells_to_notebook(notebook_path, transformer_insert_after, gnn_insert_after):
    """Add visualization cells to the notebook."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Find insertion points by searching for cell content markers
    transformer_cells = create_transformer_viz_cells()
    gnn_cells = create_gnn_viz_cells()

    # Search for insertion points
    cells = notebook['cells']

    # Find where to insert transformer cells (after "Transformer Metrics Summary")
    transformer_idx = None
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell.get('source', []))
            if 'Transformer Metrics Summary' in source or 'Cell 15.5' in source:
                transformer_idx = i + 1
                break

    # Find where to insert GNN cells (after "GNN v1.7 Production Training")
    gnn_idx = None
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if 'GNN TRAINING COMPLETE' in source:
                gnn_idx = i + 1
                break

    if transformer_idx is None:
        print('[!] Could not find insertion point for transformer cells')
        print('[!] Appending at end instead')
        transformer_idx = len(cells)

    if gnn_idx is None:
        print('[!] Could not find insertion point for GNN cells')
        print('[!] Appending at end instead')
        gnn_idx = len(cells)

    # Adjust GNN index if transformer cells were inserted before it
    if gnn_idx > transformer_idx:
        gnn_idx += len(transformer_cells)

    # Insert cells
    print(f'[+] Inserting {len(transformer_cells)} transformer analysis cells at position {transformer_idx}')
    cells[transformer_idx:transformer_idx] = transformer_cells

    print(f'[+] Inserting {len(gnn_cells)} GNN visualization cells at position {gnn_idx}')
    cells[gnn_idx:gnn_idx] = gnn_cells

    notebook['cells'] = cells

    # Save updated notebook
    backup_path = notebook_path.replace('.ipynb', '_backup.ipynb')
    print(f'[+] Creating backup at {backup_path}')
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    print(f'[+] Saving updated notebook to {notebook_path}')
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    print(f'[+] Successfully added {len(transformer_cells) + len(gnn_cells)} visualization cells')

if __name__ == '__main__':
    notebook_path = r'C:\Users\Vimal Sajan\streamguard\StreamGuard_Production_Training.ipynb'
    add_cells_to_notebook(notebook_path, transformer_insert_after=None, gnn_insert_after=None)
