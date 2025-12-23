"""
SFNET: Spatial Factor Network
Integrated Graph Autoencoder for Spatial Transcriptomics Analysis

This module provides the main entry point for SFNET, combining:
- Shared Factor Neighborhood (SFN) algorithm for cell graph construction
- Multi-view Graph Autoencoder architecture
- Enhanced attention mechanisms for cross-view fusion

Author: SFNET Team
"""
import numpy as np
import pandas as pd
import networkx as nx
import os
import argparse
import torch
import time
import preprocessing
import training
import models
from torch_geometric.nn import GAE
from rich.console import Console
from rich.text import Text


# Debug mode flag
DEBUG_MODE = False

# SFNET brand color
SFNET_COLOR = "#004ac9"


def print_logo():
    """
    Print SFNET ASCII art logo with custom color.
    Color: #CEE505 (Lime Yellow)
    """
    console = Console()
    
    logo = r"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   ███████╗███████╗███╗   ██╗███████╗████████╗                 ║
    ║   ██╔════╝██╔════╝████╗  ██║██╔════╝╚══██╔══╝                 ║
    ║   ███████╗█████╗  ██╔██╗ ██║█████╗     ██║                    ║
    ║   ╚════██║██╔══╝  ██║╚██╗██║██╔══╝     ██║                    ║
    ║   ███████║██║     ██║ ╚████║███████╗   ██║                    ║
    ║   ╚══════╝╚═╝     ╚═╝  ╚═══╝╚══════╝   ╚═╝                    ║
    ║                                                               ║
    ║   GRN-Integrated Heterogeneous Attentive Graph Autoencoder    ║
    ║   for Cell-Cell Interaction Reconstruction from Spatial       ║
    ║   Transcriptomics                                             ║
    ║                                                               ║
    ║ ┌───────────────────────────────────────────────────────────┐ ║
    ║ │  SFN Algorithm  +  Heterogeneous Attention  +  Enhancement│ ║
    ║ └───────────────────────────────────────────────────────────┘ ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    
    # Print logo with custom color #CEE505
    styled_logo = Text(logo)
    styled_logo.stylize(f"{SFNET_COLOR} bold")
    console.print(styled_logo)
    console.print()


def parse_arguments():
    """
    Parse command-line arguments for SFNET.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='SFNET: Spatial Factor Network - Integrated GAE for ST data'
    )
    
    parser.add_argument(
        "-m", "--mode", 
        type=str, 
        default="train",
        help="SFNET mode: preprocess, train, or preprocess,train"
    )
    parser.add_argument(
        "-i", "--inputdirpath", 
        type=str,
        help="Input directory path where ST data is stored"
    )
    parser.add_argument(
        "-o", "--outputdirpath", 
        type=str, 
        default="../output/seqfish",
        help="Output directory path where results will be stored"
    )
    parser.add_argument(
        "-s", "--studyname", 
        type=str, 
        default='SFNET',
        help="Study name for output files"
    )
    parser.add_argument(
        "-t", "--split", 
        type=float, 
        default=0.3,
        help="Fraction of edges for testing [0,1)"
    )
    parser.add_argument(
        "-n", "--numgenespercell", 
        type=int, 
        default=45,
        help="Number of genes in each gene regulatory network"
    )
    parser.add_argument(
        "-k", "--nearestneighbors", 
        type=int, 
        default=2,
        help="Number of nearest neighbors for each cell"
    )
    parser.add_argument(
        "--fp", 
        type=float, 
        default=0,
        help="(Experimental) Fraction of false positive test edges [0,1)"
    )
    parser.add_argument(
        "--fn", 
        type=float, 
        default=0,
        help="(Experimental) Fraction of false negative test edges [0,1)"
    )
    parser.add_argument(
        "-a", "--ownadjacencypath", 
        type=str, 
        default=None,
        help="Path to custom cell-level adjacency matrix"
    )
    parser.add_argument(
        "-lr", 
        type=float, 
        default=0.8,
        help="Base learning rate"
    )
    parser.add_argument(
        "-l", "--lrdatabase", 
        type=int, 
        default=0,
        help="Ligand-Receptor database: 0=CellTalk, 2=scMultiSim"
    )
    
    return parser.parse_args()


def run_preprocessing(st_data, num_nearest_neighbors, lr_gene_ids, 
                      cespgrn_hyperparams, custom_adjacency_path=None):
    """
    Execute the preprocessing pipeline.
    
    This function performs:
    1. CeSpGRN inference for initial GRNs
    2. Cell-level graph construction using SFN algorithm
    3. Gene-level super-graph construction
    4. Node feature generation using Node2Vec
    
    Args:
        st_data (pd.DataFrame): Spatial transcriptomics data.
        num_nearest_neighbors (int): Number of neighbors for cell graph.
        lr_gene_ids (iterable): Ligand-receptor gene indices.
        cespgrn_hyperparams (dict): Hyperparameters for CeSpGRN.
        custom_adjacency_path (str): Optional path to custom adjacency matrix.
        
    Returns:
        tuple: (cell_adj, gene_graph, num2gene, gene2num, grns, gene_features, model)
    """
    # Infer initial GRNs
    if DEBUG_MODE:
        print("1. Skipping CeSpGRN inference (debug mode)")
        grns = np.load("../output/scmultisim/1_preprocessing_output/initial_grns.npy")
    else:
        grns = preprocessing.infer_initial_grns(st_data, cespgrn_hyperparams)
    
    # Construct cell-level graph
    if custom_adjacency_path is not None:
        celllevel_adj = np.load(custom_adjacency_path)
    else:
        celllevel_adj, _ = preprocessing.construct_celllevel_graph(
            st_data, 
            num_nearest_neighbors, 
            get_edges=False
        )
    
    # Construct gene-level super-graph
    gene_level_graph, num_to_gene, gene_to_num, grn_components = \
        preprocessing.construct_genelevel_graph(
            grns, 
            celllevel_adj, 
            node_type="int", 
            lrgenes=lr_gene_ids
        )
    
    # Generate node features using Node2Vec
    gene_features, gene_feature_model = preprocessing.get_gene_features(
        grn_components, 
        feature_type="node2vec"
    )
    
    return (celllevel_adj, gene_level_graph, num_to_gene, gene_to_num, 
            grns, gene_features, gene_feature_model)


def build_model(data, hyperparams):
    """
    Build the SFNET Graph Autoencoder model.
    
    Constructs the multi-view encoder with:
    - CellEncoder for cell-level graph
    - GeneEncoder for gene-level graph
    - HeteroLayer for cross-view attention
    
    Args:
        data (tuple): (cell_level_data, gene_level_data) PyG Data objects.
        hyperparams (dict): Model hyperparameters.
        
    Returns:
        GAE: Graph Autoencoder model.
    """
    num_cells, num_cell_features = data[0][0].x.shape[0], data[0][0].x.shape[1]
    num_genes, num_gene_features = data[1].x.shape[0], data[1].x.shape[1]
    hidden_dim = hyperparams["concat_hidden_dim"] // 2
    num_genes_per_cell = hyperparams["num_genespercell"]

    # Build encoders
    cell_encoder = models.CellEncoder(num_cell_features, hidden_dim)
    gene_encoder = models.GeneEncoder(
        num_features=num_gene_features, 
        hidden_dim=hidden_dim,
        num_vertices=num_cells, 
        num_subvertices=num_genes_per_cell
    )

    # Build heterogeneous attention layer
    hetero_layer = models.HeteroLayer(hidden_dim)

    # Build multi-view encoder
    multiview_encoder = models.MultiviewEncoder(
        GeneEncoder=gene_encoder, 
        CellEncoder=cell_encoder,
        heterolayer=hetero_layer, 
        hidden_dim=hidden_dim
    )

    return GAE(multiview_encoder)


def main():
    """
    Main execution function for SFNET.
    
    Orchestrates the full pipeline:
    1. Argument parsing and setup
    2. Preprocessing (if requested)
    3. Training (if requested)
    4. Model and metrics saving
    """
    # Display SFNET logo
    print_logo()
    
    args = parse_arguments()
    
    # Print configuration
    console = Console()
    console.print("=" * 60, style=SFNET_COLOR)
    console.print("SFNET Configuration:", style=f"{SFNET_COLOR} bold")
    print(args)
    print("=" * 60)
    
    # Extract arguments
    mode = args.mode
    input_path = args.inputdirpath
    output_path = args.outputdirpath
    num_nearest_neighbors = args.nearestneighbors
    num_genes_per_cell = args.numgenespercell
    lr_database = args.lrdatabase
    study_name = args.studyname
    custom_adjacency_path = args.ownadjacencypath
    
    # Define output directories
    preprocess_output_path = os.path.join(output_path, "preprocessing_output")
    model_output_path = os.path.join(output_path, "model_pth")
    metrics_output_path = os.path.join(output_path, "training_output")

    # ==================== PREPROCESSING ====================
    if "preprocess" in mode:
        os.makedirs(preprocess_output_path, exist_ok=True)
        
        # Load spatial transcriptomics data
        st_data = pd.read_csv(input_path, index_col=None)
        required_columns = {"Cell_ID", "X", "Y", "Cell_Type"}
        assert required_columns.issubset(set(st_data.columns.to_list())), \
            f"Missing required columns: {required_columns - set(st_data.columns.to_list())}"
        
        num_cells, total_num_genes = st_data.shape[0], st_data.shape[1] - 4
        print(f"\n{num_cells} Cells & {total_num_genes} Total Genes\n")
        
        # CeSpGRN hyperparameters
        cespgrn_hyperparams = {
            "bandwidth": 0.1,
            "n_neigh": 30,
            "lamb": 0.1,
            "max_iters": 1000
        }
        
        print(f"Hyperparameters:")
        print(f"  - Nearest Neighbors: {num_nearest_neighbors}")
        print(f"  - Genes per Cell: {num_genes_per_cell}\n")
        
        # Select ligand-receptor genes
        selected_st_data, lr_gene_to_id = preprocessing.select_lr_genes(
            st_data, 
            num_genes_per_cell, 
            lr_database
        )
        
        # Extract cell-level features
        celllevel_features = st_data.drop(
            ["Cell_ID", "Cell_Type", "X", "Y"], 
            axis=1
        ).values
        
        # Run preprocessing pipeline
        (celllevel_adj, genelevel_graph, num_to_gene, gene_to_num, 
         grns, genelevel_features, genelevel_feature_model) = run_preprocessing(
            selected_st_data, 
            num_nearest_neighbors,
            lr_gene_to_id.values(), 
            cespgrn_hyperparams, 
            custom_adjacency_path
        )
        
        # Convert representations
        celllevel_edgelist = preprocessing.convert_adjacencylist_to_edgelist(celllevel_adj)
        genelevel_edgelist = nx.to_pandas_edgelist(genelevel_graph).drop(
            ["weight"], 
            axis=1
        ).to_numpy().T
        genelevel_adjmatrix = nx.adjacency_matrix(genelevel_graph, weight=None)
        
        # Validate edge list shape
        expected_edges = celllevel_adj.shape[0] * celllevel_adj.shape[1]
        assert celllevel_edgelist.shape == (2, expected_edges), \
            f"Edge list shape mismatch: expected (2, {expected_edges}), got {celllevel_edgelist.shape}"
        
        # Save preprocessing outputs
        np.save(os.path.join(preprocess_output_path, "celllevel_adjacencylist.npy"), celllevel_adj)
        np.save(os.path.join(preprocess_output_path, "celllevel_adjacencymatrix.npy"), 
                preprocessing.convert_adjacencylist_to_adjacencymatrix(celllevel_adj))
        np.save(os.path.join(preprocess_output_path, "celllevel_edgelist.npy"), celllevel_edgelist)
        np.save(os.path.join(preprocess_output_path, "celllevel_features.npy"), celllevel_features)
        np.save(os.path.join(preprocess_output_path, "genelevel_edgelist.npy"), genelevel_edgelist)
        np.save(os.path.join(preprocess_output_path, "genelevel_adjmatrix.npy"), genelevel_adjmatrix)
        np.save(os.path.join(preprocess_output_path, "initial_grns.npy"), grns)
        np.save(os.path.join(preprocess_output_path, "genelevel_features.npy"), genelevel_features)
        genelevel_feature_model.save(os.path.join(preprocess_output_path, "genelevel_feature_model"))
        
        print(f"Preprocessing outputs saved to: {preprocess_output_path}")

    # ==================== TRAINING ====================
    if "train" in mode:
        hyperparameters = {
            "num_genespercell": num_genes_per_cell,
            "concat_hidden_dim": 64,
            "optimizer": "adam",
            "criterion": torch.nn.BCELoss(),
            "num_epochs": 120, # 120 as default on A100 GPU, 30 used for test on 4090 GPU
            "split": args.split,
        }

        # Configure false edge simulation
        false_edges = None
        if args.fp != 0 or args.fn != 0:
            false_edges = {"fp": args.fp, "fn": args.fn}

        # Load preprocessed data
        celllevel_data, genelevel_data = training.create_pyg_data(
            preprocess_output_path, 
            hyperparameters["split"],
            false_edges
        )
        print(f"Gene-level edge index shape: {genelevel_data['edge_index'].shape}")
        
        os.makedirs(model_output_path, exist_ok=True)

        print("\n" + "=" * 60)
        print("Starting Training Process")
        print("=" * 60)

        # Prepare data tuple
        data = (celllevel_data, genelevel_data)

        # Build model
        model = build_model(data, hyperparameters)
        
        # Configure optimizer
        if hyperparameters["optimizer"] == "adam":
            hyperparameters["optimizer"] = (
                torch.optim.Adam(model.parameters(), lr=0.01),
            )
        
        # Train model
        trained_model, metrics_df = training.train_gae(
            model=model, 
            data=data, 
            hyperparameters=hyperparameters, 
            lr=args.lr
        )
        
        # Save trained model
        model_save_path = os.path.join(
            model_output_path, 
            f'{study_name}_trained_gae_model.pth'
        )
        torch.save(trained_model.state_dict(), model_save_path)
        print(f"Model saved to: {model_save_path}")
        
        # Save metrics
        os.makedirs(metrics_output_path, exist_ok=True)
        metrics_save_path = os.path.join(
            metrics_output_path, 
            f"{study_name}_metrics_{args.split}.csv"
        )
        metrics_df.to_csv(metrics_save_path, index=False)
        print(f"Metrics saved to: {metrics_save_path}")

    console = Console()
    console.print("\n✓ SFNET execution completed successfully!", style=f"{SFNET_COLOR} bold")


if __name__ == "__main__":
    main()
