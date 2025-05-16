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
%matplotlib inline

class GATEncoder(nn.Module):
    """GAT-based model layer with attention value extraction"""
    def __init__(self, in_dim, out_dim, heads=1):
        super().__init__()
        self.conv1 = GATConv(in_dim, out_dim)

    def forward(self, x, edge_index):
        # Returns node features and attention weights
        x_out, (edge_index, attn_weights) = self.conv1(x, edge_index,
                                                     return_attention_weights=True)
        return F.elu(x_out), attn_weights.mean(dim=1)  # Average multi-head attention

class GNNEncoder(nn.Module):
    """GCN model layer."""
    def __init__(self, in_dim, num_communities):
        super().__init__()
        self.conv1 = GCNConv(in_dim, num_communities)
        self.norm = nn.LayerNorm(num_communities)

    def forward(self, x, edge_index, edge_weight=None):

        edge_index, _ = add_self_loops(
        edge_index, edge_weight, fill_value=1e-10, num_nodes=x.size(0)
        )

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
        feature_node_map,
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
        self.out_channels = out_channels
        self.task = task
        self.heads = heads
        self.num_communities = num_communities


        # Convert adjacency matrix to a non-trainable torch tensor
        feature_node_map = torch.tensor(feature_node_map, dtype=torch.float).t()
        self.feature_node_map= Parameter(feature_node_map, requires_grad=False)

        # Define the first fully connected layer (feature to node mapping)
        self.fc1 = Linear(in_features=n_features, out_features=n_nodes)

        # Define the model to detect communities (GCN based model)
        self.CommunityDetection = GNNEncoder(self.node_embedding_dim, self.num_communities)

        # Define the model to propagate ontology features
        self.OntologyEncoder = GATEncoder(self.node_embedding_dim, 1)

        #Apply the mask to the weights of the first layer
        with torch.no_grad():
            self.fc1.weight.mul_(self.feature_node_map)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # final NN layer for classification (aggregated node activations from OntologyEncoder layer and gives final outputs)
        self.classifier = nn.Linear(self.n_nodes, self.out_channels)


    def forward(self, feature_data, graph_data, mask=None):
        """Runs the forward pass of the module."""

        edge_index, batch = graph_data.edge_index, graph_data.batch

        # Initialize the node embeddings from the feature data using the fc1 layer (this step populates the gene ontology classes with data)
        initial_embedding = self.fc1(feature_data)

        x = initial_embedding

        # create graphs for each sample
        data_list = []
        for sample in x:
            # Reshape sample to (num_nodes, 1)
            sam = torch.tensor(sample, dtype=torch.float32).view(-1, 1)
            data_obj = Data(x=sam, edge_index=edge_index)
            
            # mask when evaluating the communities for their importance (by masking communities and checking loss)
            if mask is not None:
                data_obj.x = data_obj.x * (1 - mask)

            data_list.append(data_obj)

        # Create a torch geometric DataLoader to batch the data with graphs
        loader = torch_geometric.loader.DataLoader(data_list, batch_size=len(x), shuffle=False)

        # processing the graphs with OntologyEncoder and CommunityDetection models
        pred = []
        out = x.new_zeros(x.shape)
        all_comm_assn = []
        for i, data in enumerate(loader):

            x_enc, attn_weights = self.OntologyEncoder(data.x, data.edge_index)

            out = x_enc.squeeze(1)

            comm_assn = self.CommunityDetection(x_enc, data.edge_index, edge_weight=attn_weights)
            comm_assn = F.softmax(comm_assn, dim=-1)

            # # Get the number of nodes for each graph in the batch
            batch_size = len(data_list)  # or whatever your batch size is
            num_nodes = comm_assn.shape[0] // batch_size
            all_comm_assn = comm_assn.reshape(batch_size, num_nodes, self.num_communities)

        x = out.reshape(x.shape)

        x = self.classifier(x)

        # final outputs (if task = classification, give probabilities via sigmoid/softmax)
        if self.task == 'classification':
          if self.out_channels == 1:
            x = torch.sigmoid(x) 
          else:
            x = F.softmax(x, dim=1), all_comm_assn

        return x
