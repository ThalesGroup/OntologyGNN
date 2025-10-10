   # Copyright 2021 Victoria Bourgeais
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

import pandas as pd
import numpy as np
import argparse
import json
import pickle
import os
import logging
import torch
from src.ontology_dataloader import OntologyDataLoader
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
from datetime import datetime
import yaml

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train a GNN model using ontology-based modeling with detection of important communities withing the ontology")
    parser.add_argument("--dataset", type=str, required=False, help="Path to the dataset directory")
    parser.add_argument("--n_communities", type=int, default=3, help="number of communities")
    parser.add_argument("--lambda_param", type=float, default=1.0, help="loss weighting between prediction and community metric (modularity)")
    parser.add_argument("--ontology_file", type=str, help="Path to the ontology file (ttl, rdf, owl)")
    parser.add_argument("--feature_class_map", type=str, default='mapping_titanic.json', help="Path feature-to-class map file (json")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--n_samples", type=int, required=False, help="Number of data samples for training")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--save", type=bool, default=False,
                        help="save the model checkpoint")

    return parser.parse_args()
    

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
        

def override_config(config, args):
    # Map CLI args to config paths
    if args.dataset is not None:
        config["data"]["data_directory"] = args.dataset
    if args.n_samples is not None:
        config["data"]["n_samples"] = args.n_samples
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.ontology_file is not None:
        config["data"]["ontology_file"] = args.ontology_file
    if args.feature_class_map is not None:
        config["data"]["feature_class_map"] = args.feature_class_map
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["lr"] = args.lr
    if args.n_communities is not None:
        config["hyperparams"]["num_communities"] = args.n_communities
    if args.lambda_param is not None:
        config["hyperparams"]["lambda"] = args.lambda_param
    if args.save is not None:
        config["experiment"]["save"] = args.save
 
    return config


def setup_logger(log_dir, experiment_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_path = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(experiment_path, exist_ok=True)

    log_file = os.path.join(experiment_path, "experiment.log")

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Logging to {log_file}")
    return experiment_path  # So you can save models here too

def load_data(dataset_path, device, data_config):

    if not dataset_path:
        raise ValueError("Error: dataset directory path is required.")
    
    batch_size=data_config['batch_size'],
    n_samples=data_config['n_samples']
    # features = None
    # target = None

    try:
        if "tcga" in dataset_path.lower():
            loaded_data = np.load(os.path.join(dataset_path, 'tcga.npz'))

            # Extract training and testing data
            X_train, X_test, y_train, y_test = loaded_data['X_train'], loaded_data['X_test'], loaded_data['y_train'], loaded_data['y_test']

            # If ontology graph is provided, load and process it
            #if ontology_graph:
            graph_path = os.path.join(dataset_path, 'tcga_graph.pickle')

            # Check if the graph file exists
            try:
                with open(graph_path, 'rb') as f:
                    graph = pickle.load(f)

                G = graph
                if G.is_directed():
                    edge_index = [(u, v) for u, v, _ in G.edges(data=True)]
                    edge_index = torch.tensor(edge_index, dtype=torch.long).T.to(device)
                else:
                    print("Graph is undirected.")
            except FileNotFoundError:
                raise FileNotFoundError(f"Error: The graph file '{graph_path}' was not found.")
            # else:
            #     raise FileNotFoundError(f"Error: The graph file '{graph_path}' was not found.")

            try:
                matrix_connections = pd.read_csv(os.path.join(dataset_path, 'matrix_connection_truncated.csv'))
                feature_map = matrix_connections.iloc[:, 1:].values
                ontology_node_list = matrix_connections.columns[1:].values

            except FileNotFoundError:
                raise FileNotFoundError(f"Error: The feature->node connection matrix file was not found.")

            def create_dataset(X, y, feature_map, edge_index, device=None):
                # Pre-transform the feature map (once, since it's constant)
                feature_map_t = torch.tensor(feature_map, dtype=torch.float)  # Transpose once

                X_tensor = torch.tensor(X, dtype=torch.float)  # Shape: [N, F]
                y_tensor = torch.tensor(y, dtype=torch.float if len(y.shape) == 2 else torch.long)

                # matmul of features to ontology class mappingg
                X_transformed = torch.matmul(X_tensor, feature_map_t)  # Shape: [N, F]

                data_list = []
                for i in range(X_transformed.shape[0]):
                    data = Data(x=X_transformed[i].unsqueeze(1), edge_index=edge_index.clone(), y=y_tensor[i])
                    if device:
                        data = data.to(device)
                    data_list.append(data)

                return data_list

            train_dataset = create_dataset(X_train, y_train, feature_map, edge_index, device=device)
            test_dataset = create_dataset(X_test, y_test, feature_map, edge_index, device=device)

            if n_samples:
                train_dataset = train_dataset[:n_samples]
                test_dataset = test_dataset[:int(0.2*n_samples)]

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            print('Loaded TCGA dataset')
            return train_loader, test_loader, edge_index.to(device), ontology_node_list


        elif "titanic" in dataset_path.lower():
            data = pd.read_csv(os.path.join(dataset_path, "titanic.csv"))
            # Select relevant features
            features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'CabinClass']
            data.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)
            data.dropna(inplace=True)

            targets = data.pop('Survived')

            try:
                # Process ontology
                ontology_path = os.path.join(dataset_path, data_config['ontology_file'])
                file = OntologyDataLoader(ontology_path)
                G = file.load_ontology(ontology_path)
                edges = G.edges(data=True)

                relation_matrix = [
                    (src, dst, data['label'])
                    for src, dst, data in edges
                    # if data.get('label') == 'instance'
                ]

                class_mapping_path = os.path.join(dataset_path, data_config['feature_class_map'])
                mapping = json.load(open(class_mapping_path))

                # Extract all second elements (targets of relationships)
                relation_classes = {r[1] for r in relation_matrix}

                # Combine with previously computed classes from mapping
                all_classes = sorted(set(
                    cls for cls_list in mapping.values() for cls in cls_list
                ).union(relation_classes))

                # Number of classes
                n_classes = len(all_classes)
                n_features= len(features)

                # Map class name to index
                class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}

                # Build class relationships from relation triples
                class_relationships = []

                for source, target, relation_type in relation_matrix:
                    if source in class_to_idx and target in class_to_idx:
                        class_relationships.append((class_to_idx[source], class_to_idx[target]))

                df = data.copy()
                feature_to_class = mapping

                data_list = []

                for i in range(len(df)):  # assuming `df` is your sample-wise data
                    row = df.iloc[i]
                    target = torch.tensor(targets.iloc[i])

                    # Initialize node feature vector: shape (n_classes, n_features)
                    x = torch.zeros(n_classes, n_features)

                    # Fill x based on sample's feature values and their class mappings
                    for j, feature_name in enumerate(df.columns):
                        value = row[feature_name]

                        # Convert to list of values (handle comma-separated strings)
                        if isinstance(value, str):
                            values = [v.strip() for v in value.split(',')]
                        else:
                            values = [value]

                        for v in values:
                            # Case 1: feature name maps directly to class
                            if feature_name in feature_to_class:
                                for cls in feature_to_class[feature_name]:
                                    idx = class_to_idx.get(cls)
                                    if idx is not None:
                                        x[idx, j] = float(v) if isinstance(v, (int, float, str)) and str(v).replace('.', '', 1).isdigit() else 1.0

                            # Case 2: value maps to class
                            elif str(v) in feature_to_class:
                                for cls in feature_to_class[str(v)]:
                                    idx = class_to_idx.get(cls)
                                    if idx is not None:
                                        x[idx, j] = 1.0
                    # Convert class relationships to edge_index tensor
                    edge_index = torch.tensor(class_relationships, dtype=torch.long).t().contiguous()

                    # Create PyG Data object
                    data = Data(x=x, edge_index=edge_index, y=target)

                    data_list.append(data)

                train_split = 0.7
                train_loader = DataLoader(data_list[:int(train_split*len(data_list))], batch_size=16, shuffle=True)
                test_loader = DataLoader(data_list[int(train_split*len(data_list)):], batch_size=16, shuffle=False)
                ontology_node_list = all_classes

            except FileNotFoundError:
                raise FileNotFoundError(f"Error: The file was not found or error reading it.")

            print('loaded titanic dataset')

            return train_loader, test_loader, edge_index, ontology_node_list

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The dataset file '{dataset_path}' was not found.")

    except KeyError as e:
        raise KeyError(f"Error: The expected key '{e}' was not found in the dataset.")

    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")