# Copyright 2024 Thales Group

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, GCNConv


class GATEncoder(nn.Module):
    """GAT-based model layer with attention value extraction"""

    def __init__(self, in_dim, out_dim, heads=1):
        super().__init__()
        self.conv1 = GATConv(in_dim, out_dim)

    def forward(self, x, edge_index):
        # Returns node features and attention weights
        x_out, (edge_index, attn_weights) = self.conv1(
            x, edge_index, return_attention_weights=True,
        )
        return (
            F.elu(x_out),
            edge_index,
            attn_weights.mean(dim=1),
        )  # Average multi-head attention


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
    """PyTorch graph neural network model using tabular data and an ontology graph.
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
        task="classification",
        dropout_rate=0.3,
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
        self.fc2 = nn.Linear(self.n_nodes * self.GNN_output_dim, self.out_channels)

    def forward(self, data, mask=None):
        """Runs the forward pass of the module."""
        # processing the graphs with OntologyEncoder and CommunityDetection models
        all_x_enc = []
        all_comm_assn = []
        # for i, data in enumerate(loader):

        x_enc, edge_index, attn_weights = self.OntologyEncoder(data.x, data.edge_index)

        comm_assn = self.CommunityDetection(x_enc, edge_index, edge_weight=attn_weights)
        comm_assn = F.softmax(comm_assn, dim=-1)
        # print(comm_assn.shape)

        # # Get the number of nodes for each graph in the batch
        batch_size = data.batch_size  # len(data_list)  # or whatever your batch size is
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
        if self.task == "classification":
            if self.out_channels == 1:
                x = torch.sigmoid(x)
                return x, all_comm_assn
            x = F.softmax(x, dim=1)
            return x, all_comm_assn

        if self.task == "regression":
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

                if self.task == "classification":
                    if self.out_channels == 1:
                        pred = (output > 0.5).long()
                    else:
                        pred, _ = output  # unpack if returning (x, all_comm_assn)
                        pred = torch.argmax(pred, dim=1)
                    train_preds.append(pred.cpu())
                    train_labels.append(data.y.cpu())

                elif self.task == "regression":
                    output = output.squeeze()
                    train_preds.append(output.cpu())
                    train_labels.append(data.y.cpu())

            # Evaluation on test data
            for data in test_loader:
                data = data.to(device)
                output = self(data)

                if self.task == "classification":
                    if self.out_channels == 1:
                        pred = (output > 0.5).long()
                    else:
                        pred, _ = output
                        pred = torch.argmax(pred, dim=1)
                    test_preds.append(pred.cpu())
                    test_labels.append(data.y.cpu())

                elif self.task == "regression":
                    output = output.squeeze()
                    test_preds.append(output.cpu())
                    test_labels.append(data.y.cpu())

        # Compute metrics
        if self.task == "classification":
            train_preds = torch.cat(train_preds).numpy()
            train_labels = torch.cat(train_labels).numpy()
            test_preds = torch.cat(test_preds).numpy()
            test_labels = torch.cat(test_labels).numpy()

            train_accuracy = accuracy_score(train_labels, train_preds)
            test_accuracy = accuracy_score(test_labels, test_preds)

        elif self.task == "regression":
            train_preds = torch.cat(train_preds)
            train_labels = torch.cat(train_labels)
            test_preds = torch.cat(test_preds)
            test_labels = torch.cat(test_labels)

            train_accuracy = F.mse_loss(train_preds, train_labels).item()
            test_accuracy = F.mse_loss(test_preds, test_labels).item()

        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

        return {"Training accuracy": train_accuracy, "Test accuracy": test_accuracy}

    def get_trained_communities(
        self, data_loader, device, print_stats=True, filter_label=None,
    ):
        self.eval()
        all_preds, all_communities, all_labels = [], [], []

        with torch.no_grad():
            sample_counter = 1
            for i, data in enumerate(data_loader):
                data = data.to(device)
                output = self(data)

                # If model returns both prediction and community assignment
                predictions, comm_assign_batch = (
                    output if isinstance(output, tuple) else (output, None)
                )

                # Get predicted labels based on task type
                if self.task == "classification" and self.out_channels > 1:
                    pred_labels = torch.argmax(predictions, dim=1)
                elif self.task == "classification":
                    pred_labels = (predictions > 0.5).long().squeeze()
                else:
                    pred_labels = predictions.squeeze()

                all_preds.append(pred_labels.cpu())
                all_labels.append(data.y.cpu())

                if comm_assign_batch is not None:
                    batch_size = data.num_graphs

                    # Identify start indices of each graph in batch
                    batch_starts = torch.cat(
                        [
                            torch.tensor([0], device=device),
                            torch.bincount(data.batch).cumsum(dim=0)[:-1],
                        ],
                    )

                    for graph_idx in range(batch_size):
                        # Apply filter if specified
                        if (
                            filter_label is not None
                            and pred_labels[graph_idx].item() != filter_label
                        ):
                            continue

                        comm = comm_assign_batch[graph_idx]
                        comm_ids = torch.argmax(comm, dim=1)
                        all_communities.append(comm_ids.cpu())

                        if print_stats:
                            print("sample:", sample_counter)
                            print(
                                f"  Predicted Label = {pred_labels[graph_idx].item()}, Actual Label = {data.y[graph_idx].item()}",
                            )
                            print(f"  Node Community assignments = {comm_ids.tolist()}")

                            # Count how many nodes in each community
                            comm_counts = torch.bincount(
                                comm_ids, minlength=self.num_communities,
                            )
                            for cid in range(self.num_communities):
                                print(
                                    f"    Community {cid}: {comm_counts[cid].item()} nodes",
                                )

                            # Extract edges belonging to current graph
                            node_mask = data.batch == graph_idx
                            graph_edge_mask = (
                                node_mask[data.edge_index[0]]
                                & node_mask[data.edge_index[1]]
                            )
                            graph_edge_index = data.edge_index[:, graph_edge_mask]
                            graph_edge_index_local = (
                                graph_edge_index - batch_starts[graph_idx]
                            )

                            # Count intra- and inter-community edges
                            intra = torch.zeros(
                                self.num_communities, dtype=torch.long, device=device,
                            )
                            inter = torch.zeros(
                                (self.num_communities, self.num_communities),
                                dtype=torch.long,
                                device=device,
                            )

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
                                        print(
                                            f"    Community {src_c} <-> {dst_c}: {inter[src_c, dst_c].item()} edges",
                                        )

                        sample_counter += 1

        return all_communities

    # function to compute importance of communities by masking each community and checking change in prediction loss per node
    def compute_community_importance(
        self,
        data_loader,
        criterion,
        device,
        num_communities,
        print_stats=True,
        filter_label=None,
    ):
        self.eval()
        community_importance = torch.zeros(num_communities, device=device)
        counts = torch.zeros(num_communities, device=device)

        with torch.no_grad():
            for i, data in enumerate(data_loader):
                data = data.to(device)
                output_batch, comm_assign_batch = self(data)
                targets = data.y

                # Skip batch if no matching predicted label
                if filter_label is not None:
                    preds = (
                        torch.argmax(output_batch, dim=1)
                        if self.out_channels > 1
                        else (output_batch > 0.5).long().squeeze()
                    )
                    if not any(preds == filter_label):
                        continue

                # Original loss without masking any community
                original_loss = criterion(output_batch.squeeze(1), targets)
                comm_ids = comm_assign_batch.argmax(dim=-1)

                # Loop over all communities to compute importance
                for comm_idx in range(num_communities):
                    comm_mask = comm_ids == comm_idx
                    if comm_mask.sum() == 0:
                        continue

                    # Clone data and mask nodes in the selected community
                    masked_data = data.clone()
                    batch = data.batch
                    masked_nodes = torch.zeros(
                        data.x.size(0), dtype=torch.bool, device=device,
                    )

                    for graph_idx in range(data.num_graphs):
                        node_mask_graph = batch == graph_idx
                        node_indices = node_mask_graph.nonzero(as_tuple=False).squeeze()
                        comm_mask_graph = comm_mask[graph_idx]
                        if comm_mask_graph.sum() == 0:
                            continue
                        nodes_to_mask = node_indices[comm_mask_graph]
                        masked_nodes[nodes_to_mask] = True

                    # Zero out node features
                    masked_data.x = masked_data.x.clone()
                    masked_data.x[masked_nodes] = 0
                    masked_output, _ = self(masked_data)

                    # Compute new loss and delta
                    masked_loss = criterion(masked_output.squeeze(1), targets)
                    delta_loss = (masked_loss - original_loss) / comm_mask.sum()

                    # Accumulate per-community statistics
                    community_importance[comm_idx] += delta_loss
                    counts[comm_idx] += 1

                    if print_stats:
                        print(f"\nSample {i + 1}")
                        print(
                            f"  Community {comm_idx}: Masked nodes = {comm_mask.sum().item()}, ΔLoss = {delta_loss.item():.6f}",
                        )

        return (community_importance / (counts + 1e-8)).cpu()

    # function to get number of occurances of all nodes within important communities (for all samples or for samples of a target group)
    def important_community_nodes(
        self, data_loader, device, criterion, print_stats=True, filter_label=None,
    ):
        self.eval()
        node_frequency = torch.zeros(self.n_nodes, dtype=torch.long, device=device)

        # Identify most important community based on loss impact
        comm_importance = self.compute_community_importance(
            data_loader,
            criterion,
            device,
            self.num_communities,
            print_stats=print_stats,
            filter_label=filter_label,
        )
        most_imp_comm = torch.argmax(comm_importance).item()

        with torch.no_grad():
            for data in data_loader:
                data = data.to(device)
                preds, communities = self(data)

                # Predicted labels per graph
                pred_labels = (
                    torch.argmax(preds, dim=1)
                    if self.out_channels > 1
                    else (preds > 0.5).long().squeeze()
                )
                batch_mask = (
                    pred_labels == filter_label
                    if filter_label is not None
                    else torch.ones(data.num_graphs, dtype=torch.bool, device=device)
                )

                # Get community assignment per node
                comm_ids_batch = communities.argmax(dim=-1)

                for i, comm_ids in enumerate(comm_ids_batch):
                    if batch_mask[i]:
                        node_frequency += (comm_ids == most_imp_comm).long()

        # Return dictionary mapping node names to counts
        return {
            self.node_list[i]: node_frequency[i].item() for i in range(self.n_nodes)
        }

    # function to get number of occurances of all edges within important communities (for all samples or for samples of a target group)
    def important_community_edges(
        self,
        data_loader,
        edge_index,
        device,
        criterion,
        print_stats=True,
        filter_label=None,
    ):
        self.eval()
        edge_counts = {}
        comm_importance = self.compute_community_importance(
            data_loader,
            criterion,
            device,
            self.num_communities,
            print_stats=print_stats,
            filter_label=filter_label,
        )
        most_imp_comm = torch.argmax(comm_importance).item()

        with torch.no_grad():
            for data in data_loader:
                data = data.to(device)
                preds, communities = self(data)
                pred_labels = (
                    torch.argmax(preds, dim=1)
                    if self.out_channels > 1
                    else (preds > 0.5).long().squeeze()
                )
                if filter_label is not None:
                    batch_mask = pred_labels == filter_label
                else:
                    batch_mask = torch.ones(
                        data.num_graphs, dtype=torch.bool, device=device,
                    )

                comm_ids_batch = communities.argmax(dim=-1)
                for i, comm_ids in enumerate(comm_ids_batch):
                    if not batch_mask[i]:
                        continue
                    for j in range(edge_index.size(1)):
                        src = edge_index[0, j].item()
                        dst = edge_index[1, j].item()
                        if (
                            comm_ids[src] == most_imp_comm
                            and comm_ids[dst] == most_imp_comm
                        ):
                            edge = tuple(sorted((src, dst)))
                            edge_counts[edge] = edge_counts.get(edge, 0) + 1

        return edge_counts

    def print_gradients(self):
        """Prints the gradients of all parameters in the model.
        """
        for name, param in self.named_parameters():
            if param.grad is not None:
                print(f"{name}: {param.grad.abs().mean()}")
            else:
                print(f"{name}: No gradient")
