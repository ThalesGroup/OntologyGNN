import torch
from torch import nn
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader, Batch
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
from typing import Union, Tuple, Callable, Optional
from matplotlib import pyplot as plt

class GATEncoder(nn.Module):
    """GAT-based encoder with attention value extraction"""
    def __init__(self, in_dim, out_dim, heads=1):
        super().__init__()
        self.conv1 = GATConv(in_dim, out_dim, heads=heads)

    def forward(self, x, edge_index):
        # Returns node features and attention weights
        x_out, (edge_index, attn_weights) = self.conv1(x, edge_index,
                                                     return_attention_weights=True)
        return F.elu(x_out), attn_weights.mean(dim=1)  # Average multi-head attention

class GNNEncoder(nn.Module):
    """GNN backbone for node embeddings."""
    def __init__(self, in_dim, hidden_dim, num_communities):
        super().__init__()
        self.conv1 = GCNConv(in_dim, num_communities)
        self.conv2 = GCNConv(hidden_dim, num_communities)
        self.act = nn.LeakyReLU(0.2),
        self.norm = nn.LayerNorm(num_communities)

    def forward(self, x, edge_index, edge_weight=None):

        edge_index, _ = add_self_loops(
        edge_index, edge_weight, fill_value=1e-10, num_nodes=x.size(0)
        )

        x1 = F.leaky_relu(self.conv1(x, edge_index, edge_weight=edge_weight))#self.norm(F.relu(self.conv1(x, edge_index)))
        #x1 = F.leaky_relu(self.conv2(x1, edge_index, edge_weight=edge_weight))
        x1 = self.norm(x1)
        return x1

class OntologyNNC(nn.Module):
    """
    PyTorch neural network model using tabular data and an ontology graph.
    This model integrates feature data with graph propagation through a GAT layer and detects communities in the ontology through another GNN layer using node activations and attention weights from GAT layer

    Args:
        n_features (int): Number of input features (e.g., F1, F3, F4, F5).
        n_nodes (int): Total number of nodes in the ontology graph (features + ontology terms).
        n_nodes_annot (int): Number of nodes with initial embeddings (features).
        n_nodes_emb (int): Number of nodes with embeddings after propagation.
        n_prop1 (int): Number of channels for the first propagation layer.
        adj_mat_fc1 (torch.Tensor): Adjacency matrix for feature-to-node connections.
        propagation (str): Propagation method for graph convolution (default: 'GCNConv').
    """

    def __init__(
        self,
        n_features,
        n_nodes,
        n_nodes_annot,
        n_nodes_emb,
        n_prop1,
        adj_mat_fc1,
        propagation="GCNPropagation",
        selection=None,
        ratio=1.0,
        out_channels=1,
        heads=1,
        num_communities=3,
        out_activation=None,
        task='regression',
        dropout_rate=0.3
    ):
        super(OntologyNNC, self).__init__()

        # Store the input parameters
        self.n_features = n_features
        self.n_nodes = n_nodes
        self.n_nodes_annot = n_nodes_annot
        self.n_nodes_emb = n_nodes_emb
        self.n_prop1 = n_prop1
        self.out_channels = out_channels
        self.out_activation = out_activation
        self.task = task
        self.propagation = propagation
        self.heads = heads
        self.num_communities = num_communities
        self.ratio = ratio


        # Convert adjacency matrix to a non-trainable torch tensor
        adj_mat_fc1 = torch.tensor(adj_mat_fc1, dtype=torch.float).t()
        self.adj_mat_fc1 = Parameter(adj_mat_fc1, requires_grad=False)

        # Define the first fully connected layer (feature to node mapping)
        self.fc1 = Linear(in_features=n_features, out_features=n_nodes_annot)

        self.GNNencoder = GNNEncoder(1, 16, self.num_communities)
        self.encoder = GATEncoder(n_prop1, 1)

        #Apply the mask to the weights of the first layer
        with torch.no_grad():
            self.fc1.weight.mul_(self.adj_mat_fc1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate)


        self.classifier = nn.Linear(self.n_nodes, self.out_channels)


    def forward(self, feature_data, graph_data, mask=None):
        """Runs the forward pass of the module."""

        #x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        edge_index, batch = graph_data.edge_index, graph_data.batch

        # Initialize the node embeddings from the feature data using the fc1 layer
        initial_embedding = self.fc1(feature_data)

        #Added ReLU activation after fc1 layer
        x = initial_embedding

        data_list = []
        for sample in x:
            # Reshape sample to (num_nodes, 1)
            sam = torch.tensor(sample, dtype=torch.float32).view(-1, 1)
            data_obj = Data(x=sam, edge_index=edge_index)

            #print(data_obj.x.shape)
            if mask is not None:
                data_obj.x = data_obj.x * (1 - mask)
                #print(data_obj.x.shape)

            data_list.append(data_obj)

        # # Create a DataLoader to batch the data with graphs
        loader = torch_geometric.loader.DataLoader(data_list, batch_size=len(x), shuffle=False)

        pred = []
        #print(x.shape)
        out = x.new_zeros(x.shape)
        all_comm_assn = []
        #print(out.shape)
        for i, data in enumerate(loader):
            #print(i)
            x_enc, attn_weights = self.encoder(data.x, data.edge_index)
            #print(x_enc.shape, attn_weights.shape, data.edge_index.shape)
            out = x_enc.squeeze(1)

            comm_assn = self.GNNencoder(x_enc, data.edge_index, edge_weight=attn_weights)
            comm_assn = F.softmax(comm_assn, dim=-1)

            # Get the number of nodes for each graph in the batch
            batch_size = len(data_list)  # or whatever your batch size is
            num_nodes = comm_assn.shape[0] // batch_size
            all_comm_assn = comm_assn.reshape(batch_size, num_nodes, self.num_communities)

        x = out.reshape(x.shape)

        x = self.classifier(x)

        if self.task == 'classification':
          if self.out_channels == 1:
            #x = self.relu(x)
            x = torch.sigmoid(x)
          else:
            #x = self.relu(x)
            x = F.softmax(x, dim=1), all_comm_assn#torch.stack(comm_assn_split)

        return x
