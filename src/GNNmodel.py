import torch
from torch import nn
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader, Batch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.init import xavier_uniform_, zeros_
from torch_geometric.utils import subgraph
import numpy as np
from torch import Tensor
from torch.nn import Linear, Parameter
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
import networkx as nx
from torch_geometric.typing import Adj, OptTensor, Size
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import add_self_loops
from sklearn.metrics import accuracy_score
from typing import Union, Tuple, Callable, Optional
from matplotlib import pyplot as plt

class GATEncoder(nn.Module):
    """GAT-based model layer with attention value extraction"""
    def __init__(self, in_dim, out_dim, heads=1):
        super().__init__()
        self.conv1 = GATConv(in_dim, out_dim)

    def forward(self, x, edge_index):
        # Returns node features and attention weights
        x_out, (edge_index, attn_weights) = self.conv1(x, edge_index,
                                                     return_attention_weights=True)
        return F.elu(x_out), edge_index, attn_weights.mean(dim=1)  # Average multi-head attention

class GNNEncoder(nn.Module):
    """GCN model layer."""
    def __init__(self, in_dim, num_communities):
        super().__init__()
        self.conv1 = GCNConv(in_dim, num_communities)
        self.norm = nn.LayerNorm(num_communities)

    def forward(self, x, edge_index, edge_weight=None):

        x1 = F.leaky_relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x1 = self.norm(x1)  # normalize the output
        return x1

class OntologyCommunityDetection(nn.Module):
    """
    PyTorch graph neural network model using tabular data and an ontology graph.
    This model integrates feature data with graph propagation through GAT layer, whose node activations are used for classification
    The GAT node activations and attention values are used in a separate GNN layer to assign nodes to communities.
    The model is trained for optimizing cross entropy loss (from classfication) + modularity loss (to ensure meaningful community detection)

    Args:
        n_features (int): Number of input features (number of gene expressions).
        n_nodes (int): Total number of nodes in the ontology graph (ontology class terms).
        node_embedding_dim (int): Number of channels for the first propagation layer (dimension of node embeddings).
        feature_node_map (torch.Tensor): Adjacency matrix for feature-to-class node mapping in the ontology.
        node_list (list): List of ontology class terms.
        out_channels (int, optional): Number of output classes. Defaults to 1.
        heads (int, optional): Number of attention heads. Defaults to 1.
        num_communities (int, optional): Number of communities. Defaults to 3.
        task (str): Task type ('classification' or 'regression'). Defaults to 'classification'.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.3.
    """

    def __init__(
        self,
        n_features,
        n_nodes,
        node_embedding_dim,
        GNN_output_dim,
        node_list,
        out_channels=1,
        heads=1,
        num_communities=3,
        task='classification',
        dropout_rate=0.3
    ):
        super(OntologyCommunityDetection, self).__init__()

        # Store the input parameters
        self.n_features = n_features
        self.n_nodes = n_nodes
        self.node_embedding_dim = node_embedding_dim
        self.node_list = node_list
        self.out_channels = out_channels
        self.task = task
        self.heads = heads
        self.num_communities = num_communities
        self.GNN_output_dim = GNN_output_dim


        # Define the model to detect communities (GCN based model)
        self.CommunityDetection = GNNEncoder(self.GNN_output_dim, self.num_communities)

        # Define the model to propagate ontology features
        self.OntologyEncoder = GATEncoder(self.node_embedding_dim, self.GNN_output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # final NN layer for classification (aggregated node activations from OntologyEncoder layer and gives final outputs)
        self.fc2 = nn.Linear(self.n_nodes*self.GNN_output_dim, self.out_channels)


    def forward(self, data, mask=None):
        """Runs the forward pass of the module."""

        # processing the graphs with OntologyEncoder and CommunityDetection models
        all_x_enc = []
        all_comm_assn = []
        #for i, data in enumerate(loader):

        x_enc, edge_index, attn_weights = self.OntologyEncoder(data.x, data.edge_index)

        comm_assn = self.CommunityDetection(x_enc, edge_index, edge_weight=attn_weights)
        comm_assn = F.softmax(comm_assn, dim=-1)
        # print(comm_assn.shape)

        # # Get the number of nodes for each graph in the batch
        batch_size = data.batch_size#len(data_list)  # or whatever your batch size is
        num_nodes = comm_assn.shape[0] // batch_size
        all_comm_assn = comm_assn.reshape(batch_size, num_nodes, self.num_communities)

        out = x_enc
        out = out.view(batch_size, num_nodes, self.node_embedding_dim)
        
        # reshape for next model layer
        out = out.view(out.size(0), -1)

        x = out

        # final aggregation of node activations
        x = self.fc2(x)
        # print(x.shape)

        # final outputs (if task = classification, give probabilities via sigmoid/softmax)
        if self.task == 'classification':
          if self.out_channels == 1:
              x = torch.sigmoid(x)
              return x, all_comm_assn
          else:
              x = F.softmax(x, dim=1)
              return x, all_comm_assn

        elif self.task == 'regression':
            return F.relu(x), all_comm_assn

    # function to compute target task accuracy
    def evaluate_model(self, train_loader, test_loader, device):
        self.eval()
        train_preds, train_labels = [], []
        test_preds, test_labels = [], []

        with torch.no_grad():
            # Evaluation on training data
            for data in train_loader:
                data = data.to(device)
                output = self(data)  # model forward

                if self.task == 'classification':
                    if self.out_channels == 1:
                        pred = (output > 0.5).long()
                    else:
                        pred, _ = output  # unpack if returning (x, all_comm_assn)
                        pred = torch.argmax(pred, dim=1)
                    train_preds.append(pred.cpu())
                    train_labels.append(data.y.cpu())

                elif self.task == 'regression':
                    output = output.squeeze()
                    train_preds.append(output.cpu())
                    train_labels.append(data.y.cpu())

            # Evaluation on test data
            for data in test_loader:
                data = data.to(device)
                output = self(data)

                if self.task == 'classification':
                    if self.out_channels == 1:
                        pred = (output > 0.5).long()
                    else:
                        pred, _ = output
                        pred = torch.argmax(pred, dim=1)
                    test_preds.append(pred.cpu())
                    test_labels.append(data.y.cpu())

                elif self.task == 'regression':
                    output = output.squeeze()
                    test_preds.append(output.cpu())
                    test_labels.append(data.y.cpu())

        # Compute metrics
        if self.task == 'classification':
            train_preds = torch.cat(train_preds).numpy()
            train_labels = torch.cat(train_labels).numpy()
            test_preds = torch.cat(test_preds).numpy()
            test_labels = torch.cat(test_labels).numpy()

            train_accuracy = accuracy_score(train_labels, train_preds)
            test_accuracy = accuracy_score(test_labels, test_preds)

        elif self.task == 'regression':
            train_preds = torch.cat(train_preds)
            train_labels = torch.cat(train_labels)
            test_preds = torch.cat(test_preds)
            test_labels = torch.cat(test_labels)

            train_accuracy = F.mse_loss(train_preds, train_labels).item()
            test_accuracy = F.mse_loss(test_preds, test_labels).item()

        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")


    # function to get detected communities for all individual datapoints
    def get_trained_communities(self, data_loader, device, print_stats=True):
        self.eval()
        all_preds = []
        all_communities = []
        all_labels = []

        with torch.no_grad():
            sample_counter = 1
            for i, data in enumerate(data_loader):
                data = data.to(device)

                # Forward pass through the model
                # Ensure model forward pass returns the expected outputs
                model_output = self(data)

                if isinstance(model_output, tuple) and len(model_output) == 2:
                    predictions, comm_assign_batch = model_output # comm_assign_batch shape: (batch_size, num_nodes, num_communities)
                else:
                    # Handle case where model might not return communities
                    predictions = model_output
                    comm_assign_batch = None # No communities detected by model

                if self.task == 'classification' and self.out_channels > 1:
                    pred_labels = torch.argmax(predictions, dim=1)
                elif self.task == 'classification' and self.out_channels == 1:
                    pred_labels = (predictions > 0.5).long().squeeze()
                else:
                    pred_labels = predictions.squeeze()

                all_preds.append(pred_labels.cpu())
                all_labels.append(data.y.cpu())

                if comm_assign_batch is not None:
                    # Iterate through each graph in the batch
                    batch_size = data.num_graphs
                    # Find the start index of each graph in the concatenated batch tensors
                    # This is needed to convert global indices back to local per-graph indices
                    batch_starts = torch.cat([torch.tensor([0], device=device), torch.bincount(data.batch).cumsum(dim=0)[:-1]])

                    for graph_idx in range(batch_size):
                        # Get community assignments for the current graph
                        comm = comm_assign_batch[graph_idx] # Shape: (num_nodes_in_graph, num_communities)
                        comm_ids = torch.argmax(comm, dim=1) # Shape: (num_nodes_in_graph,)

                        all_communities.append(comm_ids.cpu())

                        if print_stats:
                            print('sample:', sample_counter)
                            #print(f"\nBatch {i}, Graph {graph_idx}:")
                            print(f"  Predicted Label = {pred_labels[graph_idx].item()}, Actual Label = {data.y[graph_idx].item()}")
                            print(f"  Node Community assignments = {comm_ids.tolist()}")
                            unique_comms = torch.unique(comm_ids)
                            print(f"  Unique communities = {unique_comms.tolist()}")

                            # Community-wise node count
                            comm_counts = torch.bincount(comm_ids, minlength=self.num_communities)
                            for cid in range(self.num_communities):
                                print(f"    Community {cid}: {comm_counts[cid].item()} nodes")

                            # Community edge analysis (intra/inter) for the current graph
                            # Filter edge_index to get edges only within this graph
                            node_mask = data.batch == graph_idx
                            # Find edges where both source and destination are in this graph
                            graph_edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
                            graph_edge_index = data.edge_index[:, graph_edge_mask] # Global indices

                            # Convert global edge indices to local indices within the current graph
                            graph_edge_index_local = graph_edge_index - batch_starts[graph_idx]


                            intra = torch.zeros(self.num_communities, dtype=torch.long, device=device)
                            inter = torch.zeros((self.num_communities, self.num_communities), dtype=torch.long, device=device)

                            # Use local indices to access comm_ids
                            for edge_j in range(graph_edge_index_local.size(1)):
                                src_local, dst_local = graph_edge_index_local[:, edge_j]
                                src_c = comm_ids[src_local]
                                dst_c = comm_ids[dst_local]
                                if src_c == dst_c:
                                    intra[src_c] += 1
                                else:
                                    inter[src_c, dst_c] += 1

                            print("\n  Intra-community edges:")
                            for cid in range(self.num_communities):
                                print(f"    Community {cid}: {intra[cid].item()} edges")

                            print("\n  Inter-community edges:")
                            for src_c in range(self.num_communities):
                                for dst_c in range(self.num_communities):
                                    if src_c != dst_c and inter[src_c, dst_c] > 0:
                                        print(f"    Community {src_c} <-> {dst_c}: {inter[src_c, dst_c].item()} edges")
                        sample_counter += 1

        return all_communities

    # function to compute importance of communities by masking each community and checking change in prediction loss per node
    def compute_community_importance(
    self,
    data_loader,
    criterion,
    device,
    num_communities,
    print_stats=True
    ):
        self.eval()
        community_importance = torch.zeros(num_communities, device=device)
        counts = torch.zeros(num_communities, device=device)

        with torch.no_grad():
            for i, data in enumerate(data_loader):
                data = data.to(device)

                # Original prediction and loss
                output_batch, comm_assign_batch = self(data)
                targets = data.y
                original_loss = criterion(output_batch.squeeze(1), targets)

                if print_stats:
                    print(f"\nSample {i + 1}")
                    print(f"Original loss: {original_loss.item():.4f}")

                # Determine dominant community per node
                comm_ids = comm_assign_batch.argmax(dim=-1)  # shape: [batch_size, num_nodes]

                for comm_idx in range(num_communities):
                    # Create mask for nodes belonging to current community
                    comm_mask = (comm_ids == comm_idx)  # bool mask: [batch_size, num_nodes]

                    if comm_mask.sum() == 0:
                        continue  # skip if no nodes in this community

                    # Clone data to avoid modifying original batch
                    masked_data = data.clone()

                    # Zero-out node features of the masked community nodes
                    # data.x shape: [total_nodes_in_batch, num_features]
                    # comm_mask shape: [batch_size, num_nodes]
                    # Need to convert comm_mask to a flat mask for all nodes in batch

                    # First, get a mask of shape [total_nodes_in_batch]
                    # data.batch: tensor assigning node index to graph in batch
                    batch = data.batch  # shape: [total_nodes_in_batch]
                    masked_nodes = torch.zeros(data.x.size(0), dtype=torch.bool, device=device)

                    # For each graph in batch, zero nodes belonging to community comm_idx
                    for graph_idx in range(data.num_graphs):
                        node_mask_graph = (batch == graph_idx)
                        # Nodes of current graph in flat indices
                        node_indices = node_mask_graph.nonzero(as_tuple=False).squeeze()

                        # comm_mask[graph_idx] shape: [num_nodes_in_graph]
                        if comm_mask[graph_idx].sum() == 0:
                            continue
                        # comm_mask for current graph, convert to bool tensor
                        comm_mask_graph = comm_mask[graph_idx]

                        # Find nodes in graph belonging to community comm_idx
                        nodes_to_mask = node_indices[comm_mask_graph]

                        masked_nodes[nodes_to_mask] = True

                    # Apply mask: zero-out features of masked nodes
                    masked_data.x = masked_data.x.clone()
                    masked_data.x[masked_nodes] = 0

                    # Forward pass on masked data
                    masked_output, _ = self(masked_data)

                    masked_loss = criterion(masked_output.squeeze(1), targets)

                    delta_loss = (masked_loss - original_loss) / comm_mask.sum()

                    if print_stats:
                        print(f"  Community {comm_idx}:")
                        print(f"    Masked nodes: {int(comm_mask.sum().item())}")
                        print(f"    Loss change per node: {delta_loss.item():.6f}")

                    community_importance[comm_idx] += delta_loss
                    counts[comm_idx] += 1

            # Normalize by how many times each community appeared
            importance_normalized = community_importance / (counts + 1e-8)

        return importance_normalized.cpu()


    def important_community_nodes(self, data_loader, device, criterion, print_stats=True):
        """
        Identifies and counts how often each node belongs to the most important community across all samples in the dataloader.
        """
        self.eval()
        num_nodes = self.n_nodes
        node_frequency = torch.zeros(num_nodes, dtype=torch.long, device=device)

        # Compute community importance across all samples
        comm_importance = self.compute_community_importance(
            data_loader, criterion, device, self.num_communities, print_stats=print_stats
        )
        most_imp_comm = torch.argmax(comm_importance).item()

        with torch.no_grad():
            sample_counter = 0
            for data in data_loader:
                data = data.to(device)
                _, communities = self(data)
                # communities shape: [batch_size, num_nodes, num_communities]
                comm_ids_batch = communities.argmax(dim=-1)  # [batch_size, num_nodes]

                for comm_ids in comm_ids_batch:
                    # Count nodes belonging to most important community
                    node_frequency += (comm_ids == most_imp_comm).long()

        # Map node indices to names and frequencies
        node_name_to_frequency = {self.node_list[i]: node_frequency[i].item() for i in range(num_nodes)}

        return node_name_to_frequency


    def important_community_edges(self, data_loader, edge_index, device, criterion, print_stats=True):
        """
        Counts how often each edge connects nodes in the most important community across all samples in the dataloader.
        """
        self.eval()
        edge_counts = {}

        # Compute community importance over all samples
        comm_importance = self.compute_community_importance(
            data_loader, criterion, device, self.num_communities, print_stats=print_stats
        )
        most_imp_comm = torch.argmax(comm_importance).item()

        with torch.no_grad():
            # sample_counter = 0
            for data in data_loader:
                data = data.to(device)
                _, communities = self(data)
                comm_ids_batch = communities.argmax(dim=-1)  # [batch_size, num_nodes]

                for comm_ids in comm_ids_batch:
                    # Iterate over all edges (fixed graph)
                    for j in range(edge_index.size(1)):
                        src = edge_index[0, j].item()
                        dst = edge_index[1, j].item()

                        if comm_ids[src] == most_imp_comm and comm_ids[dst] == most_imp_comm:
                            edge = tuple(sorted((src, dst)))  # treat undirected edges equivalently
                            edge_counts[edge] = edge_counts.get(edge, 0) + 1

        return edge_counts

    def print_gradients(self):
        """
        Prints the gradients of all parameters in the model.
        """
        for name, param in self.named_parameters():
            if param.grad is not None:
                print(f"{name}: {param.grad.abs().mean()}")
            else:
                print(f"{name}: No gradient")