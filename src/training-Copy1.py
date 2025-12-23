"""
SFNET - Training Module
Implements training procedures for the Graph Autoencoder model.
"""
import sys
import os
import numpy as np
import torch
import torch_geometric.transforms as T
import pandas as pd
import torch.nn as nn
import random
import argparse
from torch_geometric.data import HeteroData, Data
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import normalize
from scipy.linalg import block_diag
from tqdm import trange

from EEB import EEB_stage2, EEB_stage1
from SEB import Local_stage2, Local_stage1, Global
from CECB import CECB
from supervised_loss import ContrastiveLoss


def add_fake_edges(num_vertices, num_old_edges, fp_ratio):
    """
    Add false positive edges for data augmentation.
    
    Args:
        num_vertices (int): Total number of vertices in the graph.
        num_old_edges (int): Original number of edges.
        fp_ratio (float): Ratio of false positive edges to add.
        
    Returns:
        torch.Tensor: Edge indices for the fake edges (shape: 2 x num_add_edges).
    """
    num_add_edges = int(fp_ratio * num_old_edges)
    fake_edges = torch.from_numpy(
        np.random.randint(0, high=num_vertices, size=(2, num_add_edges))
    )
    print(f"Added {fp_ratio * 100}% false edges")
    return fake_edges


def remove_real_edges(old_edge_indices, fn_ratio):
    """
    Remove real edges to simulate false negatives.
    
    Args:
        old_edge_indices (np.ndarray): Indices of existing edges.
        fn_ratio (float): Ratio of edges to remove.
        
    Returns:
        torch.Tensor: Indices of edges to be removed.
    """
    num_remove_edges = int(fn_ratio * old_edge_indices.shape[0])
    removed_indices = torch.from_numpy(
        np.random.choice(old_edge_indices, size=num_remove_edges, replace=False)
    ).int()
    print(f"Removed {fn_ratio * 100}% real edges")
    return removed_indices


def create_pyg_data(preprocessing_output_path, split=0.1, false_edges=None):
    """
    Create PyTorch Geometric Data objects from preprocessed numpy files.
    
    Args:
        preprocessing_output_path (str): Path to the preprocessing output folder.
        split (float): Fraction of edges to use for testing. Default: 0.1
        false_edges (dict): Optional dict with 'fp' and 'fn' keys for false edge simulation.
        
    Returns:
        tuple: (cell_level_data, gene_level_data) as PyG Data objects.
    """
    required_files = {
        "celllevel_adjacencylist.npy",
        "celllevel_adjacencymatrix.npy",
        "celllevel_edgelist.npy",
        "genelevel_edgelist.npy",
        "celllevel_features.npy",
        "genelevel_features.npy"
    }
    
    if not os.path.exists(preprocessing_output_path) or \
            not required_files.issubset(set(os.listdir(preprocessing_output_path))):
        raise Exception("Proper preprocessing files not found. Please run the 'preprocessing' step.")

    # Load preprocessed data
    celllevel_adj_matrix = torch.from_numpy(
        np.load(os.path.join(preprocessing_output_path, "celllevel_adjacencymatrix.npy"))
    ).type(torch.LongTensor)
    
    celllevel_features = torch.from_numpy(
        normalize(np.load(os.path.join(preprocessing_output_path, "celllevel_features.npy")))
    ).type(torch.float32)
    
    celllevel_edgelist = torch.from_numpy(
        np.load(os.path.join(preprocessing_output_path, "celllevel_edgelist.npy"))
    ).type(torch.LongTensor)
    
    genelevel_edgelist = torch.from_numpy(
        np.load(os.path.join(preprocessing_output_path, "genelevel_edgelist.npy"))
    ).type(torch.LongTensor)
    
    genelevel_features = torch.from_numpy(
        normalize(np.load(os.path.join(preprocessing_output_path, "genelevel_features.npy")))
    ).type(torch.float32)
    
    genelevel_grns_flat = torch.from_numpy(
        np.load(os.path.join(preprocessing_output_path, "initial_grns.npy"))
    ).type(torch.float32).flatten()

    # Create PyG Data objects
    cell_level_data = Data(
        x=celllevel_features, 
        edge_index=celllevel_edgelist, 
        y=celllevel_adj_matrix
    )
    gene_level_data = Data(
        x=genelevel_features, 
        edge_index=genelevel_edgelist, 
        y=genelevel_grns_flat
    )
    
    # Apply train/test split
    if split is not None:
        print(f"{1 - split} training edges | {split} testing edges")
        
        transform = T.RandomLinkSplit(
            num_test=split,
            num_val=0,
            is_undirected=True, 
            add_negative_train_samples=True,
            neg_sampling_ratio=1.0,
            key="edge_label",
            disjoint_train_ratio=0,
        )
        train_cell_data, _, test_cell_data = transform(cell_level_data)
        cell_level_data = (train_cell_data, test_cell_data)

    # Apply false edge modifications if specified
    if false_edges is not None:
        fp_ratio = false_edges.get("fp", 0)
        fn_ratio = false_edges.get("fn", 0)
        
        if fn_ratio != 0:
            new_indices = train_cell_data.edge_label.clone()
            old_edge_indices = np.argwhere(train_cell_data.edge_label == 1).squeeze()
            new_neg_indices = remove_real_edges(old_edge_indices, fn_ratio).long()
            new_indices[new_neg_indices] = 0
            train_cell_data.edge_label = new_indices
        
        if fp_ratio != 0:
            pos_mask = train_cell_data.edge_label == 1
            new_edges = add_fake_edges(
                train_cell_data.x.size()[0], 
                train_cell_data.edge_label_index[:, pos_mask].shape[1], 
                fp_ratio
            )
            train_cell_data.edge_label = torch.cat([
                train_cell_data.edge_label, 
                torch.ones(new_edges.shape[1])
            ])
            train_cell_data.edge_label_index = torch.cat([
                train_cell_data.edge_label_index, 
                new_edges
            ], dim=1)
            
    return cell_level_data, gene_level_data


def train(data, model, hyperparameters):
    """
    Basic training loop for the model.
    
    Args:
        data (tuple): Training data (cell_level_data, gene_level_data).
        model: The GAE model to train.
        hyperparameters (dict): Training hyperparameters.
    """
    num_epochs = hyperparameters["num_epochs"]
    optimizer = hyperparameters["optimizer"][0]
    criterion = hyperparameters["criterion"]

    with trange(num_epochs, desc="") as pbar:
        for epoch in pbar:
            pbar.set_description(f"Epoch {epoch}")
            model.train()
            optimizer.zero_grad()
            
            recon_cell, recon_gene = model(
                data[0].x, data[1].x, 
                data[0].edge_index, data[1].edge_index
            )
            
            loss = criterion(recon_cell, data[0].y.float())
            loss.backward() 
            optimizer.step()  
            
            pbar.set_postfix(loss=loss.item())


def create_intracellular_gene_mask(num_cells, num_genes_per_cell):
    """
    Create a block-diagonal mask for intracellular gene interactions.
    
    This mask identifies which gene-gene interactions are within the same cell,
    which is used for computing the intracellular penalty loss.
    
    Args:
        num_cells (int): Number of cells in the dataset.
        num_genes_per_cell (int): Number of genes per cell.
        
    Returns:
        np.ndarray: Boolean mask of shape (num_cells * num_genes_per_cell, num_cells * num_genes_per_cell).
    """
    identity_block = np.ones(shape=(num_genes_per_cell, num_genes_per_cell))
    block_list = [identity_block for _ in range(num_cells)]
    return block_diag(*block_list).astype(bool)


def compute_adaptive_learning_rate(epoch, base_lr):
    """
    Compute adaptive learning rate with warm-up and decay schedule.
    
    Learning rate schedule:
    - Epochs 0-49: lr = 0.01 (warm-up phase)
    - Epochs 50+: lr = 0.001 (decay phase)
    
    Args:
        epoch (int): Current training epoch.
        base_lr (float): Base learning rate (currently not used in schedule).
        
    Returns:
        float: Learning rate for the current epoch.
    """
    if epoch < 50:
        return 0.01  # Warm-up phase
    else:
        return 0.001  # Decay phase


def train_gae(data, model, hyperparameters, lr):
    """
    Train the Graph Autoencoder with multi-view enhancement blocks.
    
    This function implements the full SFNET training pipeline including:
    1. Cell-level encoding and enhancement (CECB + EEB)
    2. Gene-level encoding and enhancement (CECB + SEB)
    3. Triple enhancement fusion
    4. Contrastive learning loss
    5. Reconstruction loss with intracellular penalty
    
    Args:
        data (tuple): Training data ((train_cell, test_cell), gene_data).
        model: The GAE model to train.
        hyperparameters (dict): Training hyperparameters.
        lr (float): Base learning rate.
        
    Returns:
        tuple: (trained_model, metrics_df) with training metrics over epochs.
    """
    num_epochs = hyperparameters["num_epochs"]
    optimizer = hyperparameters["optimizer"][0]
    criterion = hyperparameters["criterion"]
    split = hyperparameters["split"]
    num_genes_per_cell = hyperparameters["num_genespercell"]
    
    # Unpack cell-level data
    if split is not None:
        cell_train_data = data[0][0]
        cell_test_data = data[0][1]
    else:
        cell_train_data = data[0]
    gene_train_data = data[1]

    num_cells = cell_train_data.x.shape[0]
    intracellular_mask = create_intracellular_gene_mask(num_cells, num_genes_per_cell)
    mse_loss = torch.nn.MSELoss()

    # Metrics storage
    test_roc_scores = []
    test_ap_scores = []
    test_auprc_scores = []

    with trange(num_epochs, desc="") as pbar:
        for epoch in pbar:
            pbar.set_description(f"Epoch {epoch}")
            model.train()
            optimizer.zero_grad()

            # Get positive edge mask for training
            pos_mask = cell_train_data.edge_label == 1
            
            # Encode cell and gene features
            z_combined, _, _, z_gene = model.encode(
                cell_train_data.x, 
                gene_train_data.x, 
                cell_train_data.edge_label_index[:, pos_mask], 
                gene_train_data.edge_index
            )
            
            # Update optimizer with adaptive learning rate
            hyperparameters["optimizer"] = torch.optim.Adam(
                model.parameters(), 
                lr=compute_adaptive_learning_rate(epoch, lr)
            )
            print("=" * 72)

            # ============== CECB: Cell-level Enhancement Block ==============
            # Store original embeddings for contrastive learning
            z_cell_original = z_combined
            z_gene_original = z_gene
            
            # Apply Channel Enhancement Convolution Block
            cecb_cell = CECB(z_combined.shape[1])
            cecb_gene = CECB(z_gene.shape[1])
            z_combined = cecb_cell(z_combined)
            z_gene = cecb_gene(z_gene)
            z_gene_cecb = z_gene  # Store CECB output for fusion

            # ============== SEB: Spatial Enhancement Block ==============
            in_channel = z_gene.shape[1]
            out_channel = z_gene.shape[1]
            seb_gene = Global(in_channel, out_channel)
            gene_graph_data = Data(x=z_gene, edge_index=gene_train_data.edge_index)
            z_gene, z_gene_seb = seb_gene(gene_graph_data)

            # ============== EEB: Embedding Enhancement Block ==============
            eeb_cell = EEB_stage2(z_combined.shape[1], z_combined.shape[1])
            z_combined = eeb_cell(z_combined)

            # ============== Triple Enhancement Fusion ==============
            # Fuse multi-scale gene representations
            linear1 = nn.Linear(z_gene.shape[1], 16)
            relu = nn.ReLU()
            linear2 = nn.Linear(16, z_gene.shape[1])
            linear_align = nn.Linear(z_gene_seb.shape[1], z_gene_cecb.shape[1])
            
            z_gene_seb_aligned = linear_align(z_gene_seb)
            z_gene = z_gene_original + z_gene_cecb + z_gene_seb_aligned + z_gene
            z_gene = linear1(z_gene)
            z_gene = relu(z_gene)
            z_gene = linear2(z_gene)

            # ============== Loss Computation ==============
            # Reconstruct gene-level adjacency
            recon_gene_adj = model.decoder.forward_all(z_gene)
            
            # Intracellular penalty: encourage accurate reconstruction of within-cell GRN
            intracellular_loss = mse_loss(
                recon_gene_adj[intracellular_mask], 
                gene_train_data.y
            )
            
            # Contrastive loss: encourage consistent cell representations
            contrastive_loss_fn = ContrastiveLoss(799)
            supervised_loss = contrastive_loss_fn(z_cell_original, z_combined)
            
            # Reconstruction loss for cell-level graph
            recon_loss = model.recon_loss(
                z_combined, 
                cell_train_data.edge_label_index[:, pos_mask]
            )
            
            # Total loss
            total_loss = recon_loss + intracellular_loss + supervised_loss
            
            # Compute training metrics
            train_auroc, train_ap = model.test(
                z_combined, 
                cell_train_data.edge_label_index[:, pos_mask], 
                cell_train_data.edge_label_index[:, ~pos_mask]
            )

            # Backpropagation
            total_loss.backward()  
            optimizer.step()

            # ============== Evaluation on Test Set ==============
            model.eval()
            test_pos_mask = cell_test_data.edge_label == 1
            
            test_recon_loss = model.recon_loss(
                z_combined, 
                cell_test_data.edge_label_index[:, test_pos_mask]
            )
            test_auroc, test_ap = model.test(
                z_combined, 
                cell_test_data.edge_label_index[:, test_pos_mask], 
                cell_test_data.edge_label_index[:, ~test_pos_mask]
            )
            test_precision, test_recall, _ = compute_precision_recall(
                model, z_combined, 
                cell_test_data.edge_label_index[:, test_pos_mask], 
                cell_test_data.edge_label_index[:, ~test_pos_mask]
            )
            test_auprc = auc(test_recall, test_precision)
            
            # Store metrics
            test_roc_scores.append(test_auroc)
            test_auprc_scores.append(test_auprc)
            test_ap_scores.append(test_ap)

            # Log progress
            if (epoch + 1) % 5 == 0:
                print(f"Test AUROC: {test_auroc:.4f}, Test AP: {test_ap:.4f}")
                
            pbar.set_postfix(
                train_loss=total_loss.item(), 
                train_recon_loss=recon_loss.item(),
                test_recon_loss=test_recon_loss.item()
            )

    # Create metrics DataFrame
    metrics_df = pd.DataFrame({
        "Epoch": range(num_epochs), 
        "SFNET Test AP": test_ap_scores, 
        "SFNET Test ROC": test_roc_scores
    })

    return model, metrics_df


def compute_precision_recall(model, z, pos_edge_index, neg_edge_index):
    """
    Compute precision-recall curve for link prediction evaluation.
    
    Args:
        model: The trained GAE model.
        z (torch.Tensor): Node embeddings.
        pos_edge_index (torch.Tensor): Positive edge indices.
        neg_edge_index (torch.Tensor): Negative edge indices.
        
    Returns:
        tuple: (precision, recall, thresholds) arrays from sklearn.
    """
    # Create ground truth labels
    pos_y = z.new_ones(pos_edge_index.size(1))
    neg_y = z.new_zeros(neg_edge_index.size(1))
    y = torch.cat([pos_y, neg_y], dim=0)

    # Get model predictions
    pos_pred = model.decode(z, pos_edge_index, sigmoid=True)
    neg_pred = model.decode(z, neg_edge_index, sigmoid=True)
    pred = torch.cat([pos_pred, neg_pred], dim=0)

    # Convert to numpy for sklearn
    y = y.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()

    return precision_recall_curve(y, pred)
