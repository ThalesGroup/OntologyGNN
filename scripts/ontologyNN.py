import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader, Batch
import numpy as np
from torch import Tensor
from torch_scatter import scatter_add
from torch.nn import Linear, Parameter
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
import networkx as nx
from torch_geometric.typing import Adj, OptTensor, Size
from torch_geometric.utils.num_nodes import maybe_num_nodes
from typing import Union, Tuple, Callable, Optional
from matplotlib import pyplot as plt

Adj = Tensor
OptTensor = Optional[Tensor]
Size = Optional[Tuple[int, int]]

class DAGProp(nn.Module):
    def __init__(self, in_channels, out_channels, root_weight=True,
                 bias=True, nonlinearity=torch.tanh, aggr="mean", **kwargs):
        super(DAGProp, self).__init__(**kwargs)
        
        self.in_channels = in_channels if isinstance(in_channels, tuple) else (in_channels, in_channels)
        self.out_channels = out_channels
        self.root_weight = root_weight
        self.nonlinearity = nonlinearity
        self.aggr = aggr
        
        self.lin_l = Linear(self.in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(self.in_channels[1], out_channels, bias=False)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def is_cyclic(self, edge_index: Adj) -> bool:
        graph = nx.DiGraph()
        graph.add_edges_from(edge_index.t().tolist())
        return not nx.is_directed_acyclic_graph(graph)

    def forward(self, x: torch.Tensor, edge_index: Adj, batch: OptTensor = None,
                size: Size = None) -> torch.Tensor:
        device = x.device
        
        if self.is_cyclic(edge_index):
            raise ValueError("The graph is cyclic")
        
        batch_size = int(batch.max().item() + 1) if batch is not None else 1
        num_nodes = torch.bincount(batch, minlength=batch_size).to(device) if batch is not None else torch.tensor([x.size(0)], device=device)
        max_num_nodes = num_nodes.max().item()
        
        out = torch.zeros_like(x)
        visited = torch.zeros(x.size(0), dtype=torch.bool, device=device)
        
        incoming = scatter_add(torch.ones(edge_index.size(1), dtype=torch.int8, device=device), edge_index[1], dim=0, dim_size=x.size(0))
        leaves = (incoming == 0).nonzero(as_tuple=True)[0]
        out[leaves] = self.nonlinearity(x[leaves])
        visited[leaves] = True
        
        if self.root_weight:
            mask = (x != 0).any(dim=1)
            out[mask] = self.lin_r(x[mask])
        
        previous_visits = leaves
        
        while not torch.all(visited):
            mask = torch.isin(edge_index[0], previous_visits)
            fathers = torch.unique(edge_index[1, mask])
            
            all_children_visited = scatter_add(visited[edge_index[0, mask]].to(torch.int), edge_index[1, mask], dim=0, dim_size=x.size(0))
            total_children = scatter_add(torch.ones_like(visited[edge_index[0, mask]], dtype=torch.int), edge_index[1, mask], dim=0, dim_size=x.size(0))
            ref_next_visits = fathers[all_children_visited[fathers] == total_children[fathers]]
            
            mask = torch.isin(edge_index[1], ref_next_visits)
            aggregated = scatter_add(out[edge_index[0, mask]], edge_index[1, mask], dim=0, dim_size=x.size(0))
            out[ref_next_visits] += self.lin_l(aggregated[ref_next_visits])
            out[ref_next_visits] = self.nonlinearity(out[ref_next_visits])
            
            visited[ref_next_visits] = True
            previous_visits = ref_next_visits
        
        return out

    def __repr__(self):
        return '{}({}, {}, aggr={}, nonlinearity={})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels,self.aggr,self.nonlinearity.__name__)

class TopSelection(nn.Module):
    """Top selection for pooling layer used in GraphGONet with differentiable feature selection.

    Instead of hard top‑k selection (which converts the ratio to an integer),
    this implementation uses a differentiable soft mask. The learnable ratio 
    parameter controls the threshold for keeping features as a fraction of the 
    total number of features.

    Args:
        in_channels (int): number of input channels.
        ratio (Union[float, int], optional): Initial fraction of features to keep.
            This will be learned during training. Defaults to 0.5.
        beta (float, optional): Sharpness of the sigmoid gating function.
            Larger beta makes the soft mask closer to a hard selection. Defaults to 10.0.
    """
    def __init__(self, in_channels: int, ratio: Union[float, int] = 0.5, beta: float = 10.0, **kwargs):
        super(TopSelection, self).__init__()
        self.in_channels = in_channels
        # Make ratio a learnable parameter.
        self.ratio = Parameter(torch.tensor(ratio, dtype=torch.float32), requires_grad=True)
        self.beta = beta

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Args:
            x (torch.Tensor): Node feature matrix of shape (num_nodes, num_features).
            edge_index (torch.Tensor): Edge indices (unused in this layer).
            edge_attr (torch.Tensor, optional): Edge attributes (unused).
            batch (torch.Tensor, optional): Batch vector (unused).
        Returns:
            torch.Tensor: Modified node feature matrix.
        """
        
        num_features = x.shape[1]
        # We'll use the absolute values of features as a “score” per feature.
        x_abs = x.abs()

        # Compute per-node min and max over features.
        # (Here, each row of x corresponds to a node.)
        x_min, _ = x_abs.min(dim=1, keepdim=True)
        x_max, _ = x_abs.max(dim=1, keepdim=True)

        # Instead of computing an integer number of features to keep, we use the ratio
        # to define a threshold on the “score.” When ratio is 1, threshold = x_min (i.e. nearly all features pass),
        # and when ratio is 0, threshold = x_max (i.e. almost no features pass).
        threshold = x_min + (1 - self.ratio) * (x_max - x_min)

        # Now, build a differentiable soft mask using a sigmoid.
        # The steepness parameter beta controls how close the mask is to a hard 0/1 decision.
        mask = torch.sigmoid(self.beta * (x_abs - threshold))

        # Apply the mask to the original features.
        x = x * mask
        return x

    def __repr__(self):
        return "{}({}, ratio={:.4f})".format(
            self.__class__.__name__, self.in_channels, self.ratio.item()
        )


class GCNPropagation(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) propagation layer. - for aggregating the information in the ontology classes graph

    Args:
        in_channels (int): Number of input channels (node feature dimensionality).
        out_channels (int): Number of output channels (node feature dimensionality after propagation).
        aggr (str): Aggregation method for neighborhood information (default: 'mean').
    """

    def __init__(self, in_channels, out_channels, aggr="mean"):
        super(GCNPropagation, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        #self.conv2 = GCNConv(in_channels*12, out_channels)

    def forward(self, data):
        """
        Forward pass for the GCN propagation layer.

        Args:
            x (torch.Tensor): Node feature matrix of shape [num_nodes, in_channels].
            edge_index (torch.Tensor): Graph connectivity in COO format of shape [2, num_edges].
            batch (torch.Tensor): Batch indices for graph-level tasks (optional).

        Returns:
            torch.Tensor: Updated node features of shape [num_nodes, out_channels].
        """
        # Apply GCN convolution
        x = self.conv1(data.x, data.edge_index)
        # Apply tanh activation
        x = F.tanh(x)

        return x
        
class GATPropagation(nn.Module):
    """
    Graph Attention Network (GAT) propagation layer for aggregating information in the ontology classes graph.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        heads (int, optional): Number of attention heads (default: 1).
        concat (bool, optional): Whether to concatenate the outputs from the heads (default: False).
        dropout (float, optional): Dropout probability (default: 0.0).
    """
    def __init__(self, in_channels, out_channels, heads=1, concat=False, dropout=0.0):
        super(GATPropagation, self).__init__()
        self.conv = GATConv(in_channels, out_channels, heads=heads, concat=concat, dropout=dropout)

    def forward(self, data, batch=None):
        x = self.conv(data.x, data.edge_index)
        x = F.elu(x)
        return x


class OntologyNN(nn.Module):
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
        out_activation=None,
        task='regression',
        dropout_rate=0.3
    ):
        super(OntologyNN, self).__init__()

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

        if selection:
            self.ratio = ratio
            if selection == "top":
                self.selection = TopSelection(in_channels=n_nodes, ratio=ratio)
        else:
            self.selection = None


        # Convert adjacency matrix to a non-trainable torch tensor
        adj_mat_fc1 = torch.tensor(adj_mat_fc1, dtype=torch.float).t()
        self.adj_mat_fc1 = Parameter(adj_mat_fc1, requires_grad=False)

        # Define the first fully connected layer (feature to node mapping)
        self.fc1 = Linear(in_features=n_features, out_features=n_nodes_annot)

        #Apply the mask to the weights of the first layer
        with torch.no_grad():
            self.fc1.weight.mul_(self.adj_mat_fc1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.dag_prop = DAGProp(in_channels=n_prop1, out_channels=n_prop1)

        self.gcn_prop = GCNPropagation(in_channels=self.n_prop1, out_channels=self.n_prop1)
        self.gat_prop = GATPropagation(in_channels=n_nodes, out_channels=n_nodes, heads=1, concat=False)
        # Define the graph propagation layer
        # self.propagation = eval(propagation)(
        #     in_channels=n_prop1, out_channels=n_prop1
        # )

        # Define the final fully connected layer for income prediction
        self.fc2 = Linear(in_features=n_nodes, out_features=out_channels)

    def forward(self, feature_data, graph_data):
        """Runs the forward pass of the module."""
        
        #x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        edge_index, batch = graph_data.edge_index, graph_data.batch

        # Initialize the node embeddings from the feature data using the fc1 layer
        initial_embedding = self.fc1(feature_data)

        #Added ReLU activation after fc1 layer
        x = initial_embedding
        # x = self.relu(x)

        if self.propagation == "DAGProp":
            num_data_samples, num_nodes = x.shape
            out = x.new_zeros(x.shape)
            for i in range(num_data_samples):
                # Extract features for the current sample (shape: (num_nodes,))
                x_sample = x[i].unsqueeze(1)
                # # Apply DAG propagation on the current sample
                out_sample = self.dag_prop(x_sample, edge_index, batch)
                out[i] = out_sample.squeeze(1)

            x = out
        elif self.propagation == "GCNPropagation":
            data_list = []
            for sample in x:
                # Reshape sample to (num_nodes, 1)
                sam = torch.tensor(sample, dtype=torch.float32).view(-1, 1)
                data_obj = Data(x=sam, edge_index=edge_index)
                data_list.append(data_obj)
            
            # Create a DataLoader to batch the data with graphs
            loader = DataLoader(data_list, batch_size=len(x), shuffle=False)
            for batch in loader:
                out = self.gcn_prop(batch)
            
            x = out.reshape(x.shape)#self.gcn_prop(x, edge_index, batch)
        
        elif self.propagation == "GATPropagation":
            data_list = []
            for sample in x:
                # Reshape sample to (num_nodes, 1)
                sam = torch.tensor(sample, dtype=torch.float32).view(-1, 1)
                data_obj = Data(x=sam, edge_index=edge_index)
                data_list.append(data_obj)
            
            # Create a DataLoader to batch the data with graphs
            loader = DataLoader(data_list, batch_size=len(x), shuffle=False)
            for batch in loader:
                out = self.gcn_prop(batch)
            
            x = out.reshape(x.shape)#x = self.gat_prop(x, edge_index, batch)

        if self.selection:
            x = self.selection(x, edge_index)

        # Aggregate all information with the last fc layer
        x = self.fc2(x)

        if self.task == 'classification':
          if self.out_channels == 1:
            #x = self.relu(x)
            x = torch.sigmoid(x)
          else:
            #x = self.relu(x)
            x = F.softmax(x, dim=1)
        elif self.task == 'regression':
            x = self.relu(x)
        if self.out_channels >=2:
          return x
        else:
          return x.view(-1)  # Return a 1D tensor of predictions

def interpret(model, edge_index, feature_data_test, ontology_keys, pred_label=0, sample_index=None):

    """
    Returns relevance scores for the ontology classes for a sample data prediction, or the +/- frequency of each class if all data is evaluated.
    
    Args:
        model: the trained model.
        edge_index: edge index for the ontology graph.
        feature_data_test: feature data for the test set
        ontology_keys (list): list of ontology classes.
        pred_label = interpretation classification category. Defaults to 1.
        sample_index (int, optional): index of the sample data point to interpret. If None, interpret all samples from test set. Defaults to None.
    
    """

    # if sample data point, Get model activations for it, otherwise consider full test set as sample
    if sample_index:
        sample_index = sample_index  # Choose the index of the sample data point
        sample_data = feature_data_test[sample_index].unsqueeze(0)
    else:
        sample_data = feature_data_test

    # Create the graph data for the sample
    graph_data = Data(
        x=torch.ones(model.n_nodes, model.n_nodes_emb),
        edge_index=edge_index,
        batch=torch.zeros(model.n_nodes, dtype=torch.int64)
    )
    
    #print(sample_data.shape)
    # Perform a forward pass to get activations
    with torch.no_grad():
        model.eval()
        activations = model(sample_data, graph_data)
        
        if sample_index:
            pred_label = torch.argmax(activations)

        # Access the activations of specific layers
        # Example: Get the activations of the first fully connected layer
        # Replace 'model.fc1' with the desired layer name from your model architecture
        fc1_activations = model.fc1(sample_data)
        # print("FC1 activations:", fc1_activations)

        # Example: Get the output of the propagation layer
        x = model.fc1(sample_data)
        
        if model.propagation == 'DAGProp':
            out = x.new_zeros(x.shape)
            for i in range(sample_data.shape[0]):
                # Extract features for the current sample
                x_sample = x[i].unsqueeze(1)
                # Apply DAG propagation on the current sample
                out_sample = model.dag_prop(x_sample, edge_index, graph_data.batch)
                out[i] = out_sample.squeeze(1)
        
        elif model.propagation == 'GCNPropagation':
            data_list = []
            for sample in x:
                # Reshape sample to (num_nodes, 1)
                sam = torch.tensor(sample, dtype=torch.float32).view(-1, 1)
                data_obj = Data(x=sam, edge_index=edge_index)
                data_list.append(data_obj)
            
            # Create a DataLoader to batch the graphs
            #print(data_list)
            loader = DataLoader(data_list, batch_size=len(x), shuffle=False)
            for batch in loader:
                out = model.gcn_prop(batch)
            
            out = out.reshape(x.shape)
            #out = model.gcn_prop(x, edge_index)
        
        elif model.propagation == 'GATPropagation':
            out = model.gat_prop(x, edge_index)
        
        if model.selection:
            out = model.selection(out, edge_index)
        
        fc2_activations = model.fc2(out)

    ontology_dict = {}
    for i, key in enumerate(ontology_keys):
      ontology_dict[key] = {
          'count': 0,
          'positive': 0,
          'negative': 0
      }
     
    print('predicted label:', torch.argmax(fc2_activations, axis=1).shape)
    
    if not sample_index:
        # filter the samples for the specified label
        out = out[torch.argmax(fc2_activations, axis=1)==pred_label]

    for i in range(len(out)):
        
        if sample_index:
            scores = out[i]*model.fc2.weight[torch.argmax(fc2_activations, axis=1)[0]]
        else:
            scores = out[i]*model.fc2.weight[pred_label]

        for j, key in enumerate(ontology_keys):
            ontology_dict[key] = {
                'count': ontology_dict[key]['count']+1 if scores[j] != 0 else ontology_dict[key]['count']+0,
                'positive': ontology_dict[key]['positive']+1 if scores[j] > 0 else ontology_dict[key]['positive']+0,
                'negative': ontology_dict[key]['negative']+1 if scores[j] < 0 else ontology_dict[key]['negative']+0
            }

    # Sort the ontology keys based on relevance scores in descending order
    if sample_index:
        sorted_ontology_mapping = sorted(zip(ontology_keys, scores), key=lambda x: x[1], reverse=True)

        for key, relevance in sorted_ontology_mapping:
            print(f"{key}: {relevance}")
    else:
        #print(ontology_dict)
        
        display_ratio = 0.001

        # Step 1: Sort categories by total counts (positive + negative)
        sorted_categories = sorted(
            ontology_dict.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )
        
        # Step 2: Calculate the number of categories to display
        num_categories = max(1, int(len(sorted_categories) * display_ratio))
        
        # Step 3: Select the top categories based on the ratio
        top_categories = sorted_categories[:num_categories]

        categories = [item[0] for item in top_categories]
        positive_counts = [item[1]['positive'] for item in top_categories]
        negative_counts = [item[1]['negative'] for item in top_categories]

        width = 0.5

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.barh(categories, positive_counts, width, label='Positive', color='skyblue')
        ax.barh(categories, negative_counts, width, left=positive_counts, label='Negative', color='coral')

        ax.set_xlabel('Counts')
        ax.set_ylabel('Categories')
        ax.set_title('Positive and Negative Counts per Category')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig("interpretation.png")
        # plt.show()
