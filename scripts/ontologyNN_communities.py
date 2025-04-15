import torch
from torch import nn
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader, Batch
from torch.nn.init import xavier_uniform_, zeros_
from torch_geometric.utils import subgraph
import numpy as np
#from torch_scatter import scatter, scatter_add
from torch import Tensor
#from torch_scatter import scatter_add, scatter, scatter_mean
from torch.nn import Linear, Parameter
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
import networkx as nx
#import pycombo as combo
from torch_geometric.typing import Adj, OptTensor, Size
from torch_geometric.utils.num_nodes import maybe_num_nodes
from typing import Union, Tuple, Callable, Optional
from matplotlib import pyplot as plt

class GNNEncoder(nn.Module):
    """GNN backbone for node embeddings."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, out_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index):
        x1 = F.relu(self.conv1(x, edge_index))#self.norm(F.relu(self.conv1(x, edge_index)))
        #x = x1+x
        #x = F.tanh(self.conv1(x, edge_index))
        #x = self.conv2(x, edge_index)
        return x1

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class CommunityAwareAttention(nn.Module):
    """Combines node- and community-level attention."""
    def __init__(self, in_dim, num_communities):
        super().__init__()
        self.num_communities = num_communities

        # Community assignment (structure-aware)
        self.community_assign = GCNConv(in_dim, num_communities)

        #xavier_uniform_(self.community_assign.lin.weight)

        # Node-level attention parameters
        self.node_query = nn.Parameter(torch.randn(num_communities, in_dim))
        self.node_key = nn.Linear(in_dim, in_dim)

        # In CommunityAwareAttention.__init__()
        # self.node_attn_heads = nn.ModuleList([
        #     nn.Linear(in_dim, in_dim) for _ in range(4)
        # ])

        # Community-level attention parameters
        self.comm_query = nn.Parameter(torch.randn(in_dim))
        self.comm_key = nn.Linear(in_dim, in_dim)

        self.attn_dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch):
        """
        Args:
            x: Node features [N, in_dim]
            edge_index: Graph connectivity [2, E]
            batch: Batch vector [N]
        Returns:
            graph_embs: Graph embeddings [batch_size, in_dim]
        """
        graph_embs = []
        all_comm_assign = []  # Store community assignments for all graphs

        #print(batch.unique())
        for graph_idx in batch.unique():
            mask = (batch == graph_idx)
            x_sub = x[mask]
            edge_index_sub, _ = subgraph(mask, edge_index, num_nodes=x.size(0), relabel_nodes=True)

            # Step 1: Detect communities (soft assignments)
            #x_sub = x_sub.unsqueeze(-1) if x_sub.dim() == 1 else x_sub
            #print(x_sub.shape)
            comm_logits = self.community_assign(x_sub, edge_index_sub)
            comm_assign = F.softmax(comm_logits, dim=-1)  # [N_sub, K]
            #print(comm_assign.shape)
            all_comm_assign.append(comm_assign)

            # Step 2: Node-Level Attention (within communities)
            # Compute keys for all nodes
            keys = self.node_key(x_sub)  # [N_sub, in_dim]

            # Compute attention scores per community
            node_scores = torch.einsum('kd,nd->kn', self.node_query, keys)  # [K, N_sub]
            #print(node_scores.shape)
            node_scores = node_scores * comm_assign.T  # Mask by community assignments
            node_attn = F.softmax(node_scores, dim=1)  # [K, N_sub]
            #print(node_attn)

            # Aggregate node features per community
            comm_emb = torch.einsum('kn,nd->kd', node_attn, x_sub)  # [K, in_dim]

            # Step 3: Community-Level Attention (across communities)
            comm_keys = self.comm_key(comm_emb)  # [K, in_dim]
            comm_scores = torch.einsum('d,kd->k', self.comm_query, comm_keys)  # [K]

            #comm_attn = self.attn_dropout(F.softmax(comm_scores, dim=0))
            comm_attn = F.softmax(comm_scores, dim=0)  # [K]
            #print(comm_attn)

            # Aggregate community features
            graph_emb = torch.einsum('k,kd->d', comm_attn, comm_emb)  # [in_dim]
            graph_embs.append(graph_emb)

        return torch.stack(graph_embs), all_comm_assign  # [batch_size, in_dim]


class OntologyNNC(nn.Module):
    """
    PyTorch neural network model using tabular data and an ontology graph.
    This model integrates feature data with graph propagation through GNN layers.

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
        num_communities=1,
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

        # if selection:
        #     self.ratio = ratio
        #     if selection == "top":
        #         self.selection = TopSelection(in_channels=n_nodes, ratio=ratio)
        # else:
        #     self.selection = None


        # Convert adjacency matrix to a non-trainable torch tensor
        adj_mat_fc1 = torch.tensor(adj_mat_fc1, dtype=torch.float).t()
        self.adj_mat_fc1 = Parameter(adj_mat_fc1, requires_grad=False)

        # Define the first fully connected layer (feature to node mapping)
        self.fc1 = Linear(in_features=n_features, out_features=n_nodes_annot)

        self.encoder = GNNEncoder(n_prop1, 32, 32)

        #Apply the mask to the weights of the first layer
        with torch.no_grad():
            self.fc1.weight.mul_(self.adj_mat_fc1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate)

        self.attention = CommunityAwareAttention(32, self.num_communities)
        self.classifier = nn.Linear(32, self.out_channels)


    def forward(self, feature_data, graph_data):
        """Runs the forward pass of the module."""

        #x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        edge_index, batch = graph_data.edge_index, graph_data.batch

        # Initialize the node embeddings from the feature data using the fc1 layer
        initial_embedding = self.fc1(feature_data)

        #Added ReLU activation after fc1 layer
        x = initial_embedding
        #print(x.shape)
        # x = self.relu(x)

        data_list = []
        for sample in x:
            # Reshape sample to (num_nodes, 1)
            sam = torch.tensor(sample, dtype=torch.float32).view(-1, 1)
            data_obj = Data(x=sam, edge_index=edge_index)
            data_list.append(data_obj)

        #print(len(data_list))
        # # Create a DataLoader to batch the data with graphs
        loader = torch_geometric.loader.DataLoader(data_list, batch_size=1, shuffle=False)

        pred = []
        for data in loader:
            x = self.encoder(data.x, data.edge_index)
            #print(x.shape)
            #print(x.unsqueeze(0).shape)
            x = self.dropout(x)

            # x = x.unsqueeze(0)
            # x = x.view(x.size(0), -1)
            #print(x.shape)
            graph_embs, _ = self.attention(x, data.edge_index, data.batch)
            pr = self.classifier(graph_embs) #self.classifier(x)
            #print(pr.shape)
            pred.append(pr)
            # print(self.classifier(graph_embs).shape)

        x = torch.stack(pred)
        #print(x.shape)
        #print(F.softmax(x, dim=2).shape)

        if self.task == 'classification':
          if self.out_channels == 1:
            #x = self.relu(x)
            x = torch.sigmoid(x)
          else:
            #x = self.relu(x)
            x = F.softmax(x, dim=2)

        return x

