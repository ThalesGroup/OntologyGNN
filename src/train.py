import pandas as pd
import numpy as np
import argparse
import json
import pickle
import os
import torch
from src.GNNmodel import OntologyCommunityDetection
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch_geometric.data import Data, Batch
import datetime
import matplotlib
import warnings
matplotlib.use('Agg')
warnings.filterwarnings('ignore')


# function to compute modularity loss (for detected communities)
def compute_modularity_loss_fast(all_comm_assign, edge_index, num_nodes):
    # Precompute reusable terms
    row, col = edge_index
    m = edge_index.size(1) // 2  # Undirected edges
    norm = 1 / (2 * m)

    # Compute degrees (vectorized)
    deg = torch.zeros(num_nodes, device=edge_index.device)
    deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float32))

    # Precompute deg[i] * deg[j] / (2m) for all edges
    deg_prod = deg[row] * deg[col] / (2 * m)

    # Process all community assignments in parallel
    all_mod = []
    for comm_assign in all_comm_assign:
        comm_assign = F.softmax(comm_assign, dim=-1)  # Soft assignments

        # Vectorized computation of (comm_assign[i] * comm_assign[j]).sum() for all edges
        pairwise_prods = (comm_assign[row] * comm_assign[col]).sum(dim=1)

        # Compute Q in one go (vectorized)
        Q = norm * ((1 - deg_prod) * pairwise_prods).sum()

        # Maximize modularity = minimize -Q
        all_mod.append(-Q)

    return torch.stack(all_mod)


# training function main
def train_model(train_loader, test_loader, device, config, node_list=None, print_stats=True, plot=True, model=None):

    model_specs = config['model']
    params = config['hyperparams']
    training_specs = config['training']
    
    n_nodes = train_loader.dataset[0].x.shape[0]
    n_features = train_loader.dataset[0].x.shape[1]

    y_values = []
    for data in train_loader:
        y_values.extend(data.y.tolist())

    unique_y_count = len(set(y_values))

    if model is None:
        # Create the model
        model = OntologyCommunityDetection(
            n_features=n_features,
            n_nodes=n_nodes,
            node_embedding_dim=train_loader.dataset[0].x.shape[1],
            GNN_output_dim=train_loader.dataset[0].x.shape[1],
            node_list=node_list,
            num_communities=params['num_communities'],
            out_channels=unique_y_count,
            task='classification',
            dropout_rate=0.3
        )
        model = model.to(device)

    else:
        print("Using existing model")
        model = model.to(device)

    # model.apply(init_weights)

    # Use multiple GPUs if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # assign different learning rates for different model layers
    learning_rates = {
      "CommunityDetection.conv1.bias": 1e-06,
      "CommunityDetection.conv1.lin.weight": 1e-06,
      "CommunityDetection.norm.weight": 1e-04,
      "CommunityDetection.norm.bias": 1e-04
    }

    all_params = []
    for name, param in model.named_parameters():
        if name in learning_rates:
            all_params.append({'params': [param], 'lr': learning_rates[name]})
        else:
            all_params.append({'params': [param], 'lr': training_specs['lr']}) # Default learning rate

    optimizer = optim.Adam(all_params, weight_decay=1e-4)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    criterion = nn.CrossEntropyLoss().to(device)

    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5, verbose=True)

    train_losses = []
    test_losses = []
    test_mod = []
    modularities = []

    epochs = training_specs['epochs']

    for epoch in range(epochs):
        model.train()
        batch_train_loss = 0

        mod_loss = []
        all_communities = []
        for data in train_loader:
            data = data.to(device, non_blocking=True)
        # target_batch = target_batch.to(device, non_blocking=True)

            optimizer.zero_grad()

            output_batch, communities = model(data)
            # print(output_batch.squeeze(1), torch.tensor(data.y))

            modularity = torch.mean(compute_modularity_loss_fast(communities, train_loader.dataset[0].edge_index.to(device), n_nodes))
            loss = criterion(output_batch.squeeze(1), data.y) + params['lambda']*modularity

            loss.backward()
            optimizer.step()

            mod_loss.append(modularity.cpu().detach().numpy())
            batch_train_loss += loss.item()

        train_losses.append(batch_train_loss / len(train_loader))


        # Evaluate on test set
        model.eval()
        batch_test_loss = 0

        with torch.no_grad():
            for data in test_loader:
                data = data.to(device, non_blocking=True)#, target_batch.to(device, non_blocking=True)

                output_batch, communities = model(data)

                loss = criterion(output_batch.squeeze(1), data.y)# + mod_loss
                batch_test_loss += loss.item()

        test_losses.append(batch_test_loss / len(test_loader))
        #scheduler.step(batch_test_loss / len(test_loader))

        if print_stats:
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}', 'Modularity: ', -np.mean(mod_loss))
        modularities.append(-np.mean(mod_loss))

    if plot:
        # Plot losses and save figure
        plt.figure()
        ax1 = plt.gca()
        ax1.plot(train_losses, label="Train Loss", color='tab:blue')
        ax1.plot(test_losses, label="Test Loss", color='tab:orange')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color='tab:blue')
        ax1.legend()
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Create second y-axis for modularity
        ax2 = ax1.twinx()
        ax2.plot(modularities, label="Modularity", color='tab:green')
        ax2.set_ylabel("Modularity", color='tab:green')
        ax2.legend()
        ax2.tick_params(axis='y', labelcolor='tab:green')

        # Title and legend
        plt.title("Train/Test Loss and Modularity Over Epochs")
        fig = plt.gcf()
        fig.tight_layout()
        plt.savefig("loss_modularity_plot.png")
        plt.show()

    return model, test_losses, modularities