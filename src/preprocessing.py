"""
SFNET - Preprocessing Module
Implements data preprocessing with Shared Factor Neighborhood (SFN) algorithm
for constructing cell-level and gene-level graphs from spatial transcriptomics data.
"""
import numpy as np
import pandas as pd
import networkx as nx
import time
import random
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from collections import Counter
from node2vec import Node2Vec
from rich.progress import track
from rich.console import Console

# Directory configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')


def convert_adjacencylist_to_edgelist(adj_list):
    """
    Convert adjacency list representation to edge list format.
    
    Args:
        adj_list (np.ndarray): Adjacency list of shape (num_nodes, k_neighbors).
        
    Returns:
        np.ndarray: Edge list of shape (2, num_edges).
    """
    edge_list = []
    
    for node, neighbors in enumerate(adj_list):
        for neighbor in neighbors:
            edge_list.append([node, neighbor])
            
    return np.array(edge_list).T


def convert_adjacencylist_to_adjacencymatrix(adj_list):
    """
    Convert adjacency list to adjacency matrix representation.
    
    Args:
        adj_list (np.ndarray): Adjacency list of shape (num_nodes, k_neighbors).
        
    Returns:
        np.ndarray: Symmetric adjacency matrix of shape (num_nodes, num_nodes).
    """
    num_vertices = len(adj_list)
    adj_matrix = np.zeros(shape=(num_vertices, num_vertices))
    
    for i in range(num_vertices):
        for j in adj_list[i]:
            adj_matrix[i][j] = 1
            adj_matrix[j][i] = 1  # Ensure symmetry
    
    return adj_matrix


def select_lr_genes(data_df, num_genes_per_cell, lr_database=0):
    """
    Select ligand-receptor genes and random genes for GRN construction.
    
    This function prioritizes ligand-receptor pairs from the database,
    then fills remaining slots with randomly selected genes.
    
    Args:
        data_df (pd.DataFrame): Spatial transcriptomics data with columns
            ["Cell_ID", "X", "Y", "Cell_Type", gene1, gene2, ...].
        num_genes_per_cell (int): Total number of genes to select per cell.
        lr_database (int): Database selector (0: CellTalk, 2: scMultiSim).
        
    Returns:
        tuple: (selected_data_df, lr_gene_to_id_mapping)
    """
    if lr_database == 0:
        # Use CellTalk mouse ligand-receptor database
        sample_counts = data_df.drop(["Cell_ID", "X", "Y", "Cell_Type"], axis=1)
        lr_df = pd.read_csv(os.path.join(DATA_DIR, "celltalk_mouse_lr_pair.txt"), sep="\t")
        
        # Extract receptor and ligand gene sets
        receptors = set(lr_df["receptor_gene_symbol"].str.upper().to_list())
        ligands = set(lr_df["ligand_gene_symbol"].str.upper().to_list())

        # Create case-insensitive gene name mappings
        real_to_upper = {x: x.upper() for x in sample_counts.columns}
        upper_to_real = {upper: real for real, upper in real_to_upper.items()}
        candidate_genes = set(np.vectorize(real_to_upper.get)(sample_counts.columns.to_numpy()))
        
        # Select genes that exist in both data and LR database
        selected_ligands = candidate_genes.intersection(ligands)
        selected_receptors = candidate_genes.intersection(receptors)
        selected_lrs = selected_ligands | selected_receptors
        
        # Limit LR genes if too many
        if len(selected_lrs) > num_genes_per_cell // 2 + 1:
            selected_lrs = set(random.sample(tuple(selected_lrs), num_genes_per_cell // 2 + 1))
            selected_ligands = selected_lrs.intersection(selected_ligands)
            selected_receptors = selected_lrs.intersection(selected_receptors)
        
        # Fill remaining slots with random genes
        num_genes_remaining = num_genes_per_cell - len(selected_ligands) - len(selected_receptors)
        candidate_genes_remaining = candidate_genes - selected_ligands - selected_receptors
        selected_random_genes = set(random.sample(tuple(candidate_genes_remaining), num_genes_remaining))
        
        selected_genes = list(selected_random_genes | selected_ligands | selected_receptors)
        
        # Build final dataframe with selected genes
        selected_columns = ["Cell_ID", "X", "Y", "Cell_Type"] + \
                          np.vectorize(upper_to_real.get)(selected_genes).tolist()
        selected_df = data_df[selected_columns]
        
        # Create LR gene to column index mapping
        lr_to_id = {
            gene: list(selected_df.columns).index(gene) - 4 
            for gene in np.vectorize(upper_to_real.get)(list(selected_ligands | selected_receptors))
        }
        
        return selected_df, lr_to_id
    
    elif lr_database == 2:
        # Use scMultiSim simulated LR pairs
        sample_counts = data_df.drop(["Cell_ID", "X", "Y", "Cell_Type"], axis=1)
        candidate_genes = sample_counts.columns.to_numpy()
        
        scmultisim_lrs = pd.read_csv(
            os.path.join(DATA_DIR, "scMultiSim/simulated/cci_gt.csv")
        )[["ligand", "receptor"]]
        
        selected_ligands = np.unique(scmultisim_lrs["ligand"])
        selected_receptors = np.unique(scmultisim_lrs["receptor"])
        selected_lrs = np.concatenate((selected_ligands, selected_receptors), axis=0)
        
        num_genes_remaining = num_genes_per_cell - len(selected_ligands) - len(selected_receptors)
        indices = np.argwhere(candidate_genes == selected_lrs)
        candidate_genes_remaining = np.delete(candidate_genes, indices)
        selected_random_genes = random.sample(set(candidate_genes_remaining), num_genes_remaining)
        
        selected_genes = np.concatenate((selected_lrs, selected_random_genes), axis=0)
        new_columns = ["Cell_ID", "X", "Y", "Cell_Type"] + list(selected_genes)
        selected_df = data_df[new_columns]
        
        lr_to_id = {gene: list(selected_df.columns).index(gene) - 4 for gene in selected_genes}
        
        return selected_df, lr_to_id
    
    else:
        raise ValueError(f"Invalid lr_database type: {lr_database}. Use 0 (CellTalk) or 2 (scMultiSim).")


def infer_initial_grns(data_df, cespgrn_hyperparams):
    """
    Infer initial gene regulatory networks using CeSpGRN.
    
    CeSpGRN (Cell-Specific Gene Regulatory Network) uses graphical lasso
    with spatial kernels to infer cell-specific GRNs.
    
    Args:
        data_df (pd.DataFrame): Spatial transcriptomics data.
        cespgrn_hyperparams (dict): Hyperparameters for CeSpGRN including:
            - bandwidth: Kernel bandwidth
            - n_neigh: Number of neighbors for truncation
            - lamb: Regularization parameter
            - max_iters: Maximum iterations
            
    Returns:
        np.ndarray: Inferred GRNs of shape (num_cells, num_genes, num_genes).
    """
    console = Console()
    
    from submodules.CeSpGRN.src import kernel
    from submodules.CeSpGRN.src import g_admm as CeSpGRN
    
    with console.status("[cyan] Preparing CeSpGRN ...") as status:
        status.update(spinner="aesthetic", spinner_style="cyan")
        
        counts = data_df.drop(["Cell_ID", "X", "Y", "Cell_Type"], axis=1).values
        
        # PCA for dimensionality reduction
        pca = PCA(n_components=20)
        x_pca = pca.fit_transform(counts)
        
        # Extract hyperparameters
        bandwidth = cespgrn_hyperparams["bandwidth"]
        n_neigh = cespgrn_hyperparams["n_neigh"]
        lamb = cespgrn_hyperparams["lamb"]
        max_iters = cespgrn_hyperparams["max_iters"]
        
        # Compute spatial kernel
        kernel_matrix, kernel_truncated = kernel.calc_kernel_neigh(
            x_pca, k=5, bandwidth=bandwidth, 
            truncate=True, truncate_param=n_neigh
        )
        
        # Estimate covariance
        empirical_cov = CeSpGRN.est_cov(X=counts, K_trun=kernel_truncated, weighted_kt=True)
        
        # Initialize CeSpGRN model
        cespgrn_model = CeSpGRN.G_admm_minibatch(
            X=counts[:, None, :], 
            K=kernel_matrix, 
            pre_cov=empirical_cov, 
            batchsize=120
        )
    
    # Train and get GRNs
    grns = cespgrn_model.train(max_iters=max_iters, n_intervals=100, lamb=lamb)
    
    return grns


def construct_celllevel_graph(data_df, k, get_edges=False):
    """
    Construct cell-level graph using Shared Factor Neighborhood (SFN) algorithm.
    
    This is the core innovation of SFNET, combining SFN-based functional similarity with
    spatial proximity to construct a hybrid cell-level graph that captures both:
    1. Expression-based similarity (via PCA factor analysis)
    2. Spatial proximity (Euclidean distance)
    
    SFN Algorithm Steps:
    1. PCA dimensionality reduction (retain 90% variance)
    2. Factor extraction: Each cell's dominant PCA component
    3. Expression-based kNN in PCA space
    4. Factor Neighborhood (FN) vector: Distribution of factors in neighborhood
    5. SFN similarity: Manhattan distance between FN vectors
    6. Fusion: Combine SFN and spatial similarities
    
    Args:
        data_df (pd.DataFrame): Spatial transcriptomics data with columns
            ["Cell_ID", "X", "Y", "Cell_Type", gene1, gene2, ...].
        k (int): Number of nearest neighbors per cell.
        get_edges (bool): If True, also return edge coordinates for visualization.
        
    Returns:
        tuple: (adjacency_list, edge_coordinates) where edge_coordinates is None
            if get_edges is False.
    """
    num_cells = len(data_df)
    coords = np.vstack([data_df["X"].values, data_df["Y"].values]).T
    
    # ===== Step 1: SFN Algorithm - Expression-based Similarity =====
    exp_matrix = data_df.drop(["Cell_ID", "X", "Y", "Cell_Type"], axis=1).values
    
    # PCA dimensionality reduction (retain 90% variance)
    pca = PCA(n_components=0.9, svd_solver='full')
    pca_features = pca.fit_transform(exp_matrix)
    
    # Scale PCA features and determine dominant 'Factor' for each cell
    # The factor represents which PCA component contributes most to the cell's profile
    scaler = StandardScaler()
    pca_scaled = scaler.fit_transform(pca_features)
    cell_factors = np.argmax(pca_scaled, axis=1) + 1  # 1-based factor index
    
    # Expression-based kNN in PCA space
    k_neighbors = min(10, num_cells - 1)  # Ensure k doesn't exceed available cells
    nbrs_expression = NearestNeighbors(n_neighbors=k_neighbors).fit(pca_scaled)
    _, neighbor_indices = nbrs_expression.kneighbors(pca_scaled)
    
    # Build Factor Neighborhood (FN) vectors
    # FN vector: Distribution of factors among a cell's expression neighbors
    fn_vectors = []
    num_factors = pca_scaled.shape[1]
    
    for i in range(num_cells):
        neighbor_factors = cell_factors[neighbor_indices[i]]
        factor_counts = Counter(neighbor_factors)
        vector = [factor_counts.get(f, 0) for f in range(1, num_factors + 1)]
        fn_vectors.append(vector)
    
    # Compute SFN similarity using Manhattan distance between FN vectors
    # Lower distance = higher similarity
    fn_distance = squareform(pdist(np.array(fn_vectors), 'cityblock'))
    sfn_similarity = 1 / (1 + fn_distance)
    
    # ===== Step 2: Spatial Similarity =====
    spatial_distance = squareform(pdist(coords, 'euclidean'))
    # Normalize to [0, 1] similarity scale
    spatial_similarity = 1 / (1 + spatial_distance / (np.max(spatial_distance) + 1e-9))
    
    # ===== Step 3: Fuse Similarities and Select Top-k Neighbors =====
    # Equal weighting of SFN and spatial similarities
    combined_score = sfn_similarity + spatial_similarity
    
    adjacency = np.zeros(shape=(num_cells, k), dtype=int)
    edge_x = []
    edge_y = []

    for i in track(range(num_cells), description="[cyan]2. SFNET: Constructing Integrated Cell-Level Graph"):
        x0, y0 = coords[i]
        
        # Exclude self-connection and select top-k neighbors
        scores = combined_score[i].copy()
        scores[i] = -1  # Mask self
        neighbors = np.argsort(scores)[-k:]
        adjacency[i] = neighbors
        
        if get_edges:
            for neighbor_cell in neighbors:
                x1, y1 = coords[neighbor_cell]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
    
    edges = [edge_x, edge_y] if get_edges else None
    
    return adjacency, edges


def construct_genelevel_graph(disjoint_grns, celllevel_adj_list, node_type="int", lrgenes=None):
    """
    Construct gene-level super-graph from cell-specific GRNs.
    
    This function creates a unified gene graph by:
    1. Creating disjoint subgraphs for each cell's GRN
    2. Connecting ligand-receptor genes between neighboring cells
    
    Args:
        disjoint_grns (np.ndarray): Cell-specific GRNs of shape (num_cells, num_genes, num_genes).
        celllevel_adj_list (np.ndarray): Cell adjacency list from construct_celllevel_graph.
        node_type (str): "int" for integer node labels, "str" for string labels.
        lrgenes (iterable): Indices of ligand-receptor genes to connect across cells.
        
    Returns:
        tuple: (gene_level_graph, num_to_gene_dict, gene_to_num_dict, grn_union)
    """
    num_genes = disjoint_grns[0].shape[0]
    num_cells = disjoint_grns.shape[0]
    
    # Node label mappings
    num_to_gene = {}  # Maps integer node ID to "Cell{i}_Gene{j}" format
    gene_to_num = {}  # Inverse mapping
    
    assert max(lrgenes) <= num_genes

    # Create individual GRN graphs
    grn_graph_list = []
    for cell_idx, grn in enumerate(track(disjoint_grns, description="[cyan]3a. Combining individual GRNs")):
        
        graph = nx.from_numpy_array(grn)
        grn_graph_list.append(graph)
        
        for gene_idx in range(num_genes):
            node_id = cell_idx * num_genes + gene_idx
            node_name = f"Cell{cell_idx}_Gene{gene_idx}"
            num_to_gene[node_id] = node_name
            gene_to_num[node_name] = node_id

    # Create disjoint union of all GRN graphs
    if not grn_graph_list:
        raise ValueError("grn_graph_list is empty!")
    grn_union = nx.disjoint_union_all(grn_graph_list)
    gene_level_graph = nx.relabel_nodes(grn_union, num_to_gene)
    
    # Connect LR genes between neighboring cells
    for cell_idx, neighborhood in enumerate(track(
        celllevel_adj_list, 
        description="[cyan]3b. Constructing Gene-Level Graph"
    )):
        for neighbor_cell in neighborhood:
            if neighbor_cell != -1:
                for lr_gene1 in lrgenes:
                    for lr_gene2 in lrgenes:
                        node1 = f"Cell{cell_idx}_Gene{lr_gene1}"
                        node2 = f"Cell{neighbor_cell}_Gene{lr_gene2}"
                        
                        if not gene_level_graph.has_node(node1) or not gene_level_graph.has_node(node2):
                            raise ValueError(
                                f"Nodes {node1} or {node2} not found. "
                                "Debug the Gene-Level Graph creation."
                            )
                        
                        gene_level_graph.add_edge(node1, node2)

    # Convert node labels to integers if requested
    if node_type == "int":
        gene_level_graph = nx.convert_node_labels_to_integers(gene_level_graph)

    assert len(gene_level_graph.nodes()) == num_cells * num_genes

    return gene_level_graph, num_to_gene, gene_to_num, grn_union


def get_gene_features(graph, feature_type="node2vec"):
    """
    Generate gene node features using graph embedding methods.
    
    Args:
        graph (nx.Graph): Gene-level graph.
        feature_type (str): Embedding method ("node2vec" supported).
        
    Returns:
        tuple: (feature_vectors, trained_model)
    """
    if feature_type == "node2vec":
        node2vec = Node2Vec(
            graph, 
            dimensions=64, 
            walk_length=15, 
            num_walks=100, 
            workers=4
        )
        model = node2vec.fit()
        gene_feature_vectors = model.wv.vectors
        
    return gene_feature_vectors, model


# Legacy function aliases for backward compatibility
def convert_adjacencylist2edgelist(adj_list):
    """Deprecated: Use convert_adjacencylist_to_edgelist instead."""
    return convert_adjacencylist_to_edgelist(adj_list)


def convert_adjacencylist2adjacencymatrix(adj_list):
    """Deprecated: Use convert_adjacencylist_to_adjacencymatrix instead."""
    return convert_adjacencylist_to_adjacencymatrix(adj_list)


def select_LRgenes(data_df, num_genespercell, lr_database=0):
    """Deprecated: Use select_lr_genes instead."""
    return select_lr_genes(data_df, num_genespercell, lr_database)
