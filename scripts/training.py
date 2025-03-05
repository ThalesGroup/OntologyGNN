import pandas as pd
import numpy as np
import argparse
import json
import pickle
import os
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
from ontologyNN import OntologyNN, interpret
import datetime
import matplotlib
import warnings
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model using ontology-based modeling")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset directory")
    #parser.add_argument("--ontology_graph", type=str, required=False, help="Path to the ontology graph (pickle)")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--propagation", type=str, required=True, choices=["GCNPropagation", "GATPropagation", "DAGProp"], 
                        help="Propagation method: GCNPropagation, GATPropagation or DAGProp")
    parser.add_argument("--interpret", nargs="?", const=True, type=int, 
                        help="Interpret the model predictions. If a number is provided, it is used as the sample index.")
    parser.add_argument("--label", type=int, default=1, 
                        help="Prediction label to interpret (default: 1)")
    parser.add_argument("-ontology_nodes", nargs="?", const=True, type=list, 
                        help="list of ontology nodes for interpretation")
    return parser.parse_args()

def load_data(dataset_path, ontology_graph=None):
    if not dataset_path:
        raise ValueError("Error: dataset directory path is required.")
    
    features = None
    target = None
    
    try:
        if "tcga" in dataset_path.lower():
            loaded_data = np.load(os.path.join(dataset_path, 'tcga.npz'))

            # Extract training and testing data
            X_train, X_test, y_train, y_test = loaded_data['X_train'], loaded_data['X_test'], loaded_data['y_train'], loaded_data['y_test']
            
            # If ontology graph is provided, load and process it
            #if ontology_graph:
            graph_path = os.path.join(dataset_path, 'tcga_graph.pickle')#ontology_graph#'/beegfs/home/u1257/ontologyNN/TCGA_data/TGCA_graph.pickle'
            
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
                    
                
            print('loaded tcga data')
            return X_train, X_test, y_train, y_test, feature_map, edge_index, ontology_node_list
        
    
        elif "titanic" in dataset_path.lower():
            data = pd.read_csv(os.path.join(dataset_path, "titanic.csv"))
            # Select relevant features
            features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'CabinClass']
            target = 'Survived'
            
            # Handle missing values
            data.dropna(inplace=True)
            
            # Encode categorical variables
            le = LabelEncoder()
            data['Sex'] = le.fit_transform(data['Sex'])
            data['Embarked'] = le.fit_transform(data['Embarked'].astype(str))
            
            # Scale numerical values
            scaler = StandardScaler()
            data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])
            
            # Extract cabin class
            def extract_cabin_class(cabin):
                if pd.isna(cabin):
                    return 'Unknown'
                else:
                    return cabin[0]
            
            data['CabinClass'] = data['Cabin'].apply(extract_cabin_class)
            data['CabinClass'] = le.fit_transform(data['CabinClass'])
            
            np.random.seed(42)
            X_train, X_test, y_train, y_test = train_test_split(
             data[features], data['Survived'], test_size=0.2, random_state=42)
             
            try:
                
                ontology_path = os.path.join(dataset_path, "titanic_ontology.json")
                # Process ontology
                with open(ontology_path, "r") as file:
                    ont = json.load(file)
                
                ontology_mapping = ont["ontology_mapping"]
                ontology_relationships = ont["ontology_relationships"]
                
                n_features = len(features)
                n_nodes = len(ontology_mapping)  # Number of ontological nodes
                adj_mat_fc1 = np.zeros((n_features, n_nodes))
                
                # Assign connections based on the provided mapping
                feature_to_node_index = {feature: i for i, feature in enumerate(features)}
                node_to_index = {node: i for i, node in enumerate(ontology_mapping)}
                
                for feature, node_list in ontology_mapping.items():
                    for feature_name in node_list:
                        if feature_name in feature_to_node_index:
                            adj_mat_fc1[feature_to_node_index[feature_name], node_to_index[feature]] = 1
                
                feature_map = adj_mat_fc1
                ontology_node_list = ontology_mapping.keys()
                # Create edge_index tensor and move to device
                edge_index = torch.tensor(ontology_relationships, dtype=torch.long).t().contiguous().to(device)
            
            except FileNotFoundError:
                raise FileNotFoundError(f"Error: The feature->node connection matrix file was not found.")
            
            print('loaded titanic dataset')
             
            return X_train.values, X_test.values, y_train.values, y_test.values, feature_map, edge_index, ontology_node_list
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The dataset file '{dataset_path}' was not found.")
    
    except KeyError as e:
        raise KeyError(f"Error: The expected key '{e}' was not found in the dataset.")

    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")
        

def train_model(X_train, X_test, y_train, y_test, feature_map, edge_index, args, device):
    
    # Convert to tensors (will be moved to device during training)
    feature_data_train = torch.Tensor(X_train)
    feature_data_test = torch.Tensor(X_test)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Batch size
    batch_size = len(X_test)
    
    # Create datasets and data loaders for batching
    train_dataset = TensorDataset(feature_data_train, y_train_tensor)
    test_dataset = TensorDataset(feature_data_test, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    n_nodes=feature_map.shape[1]
    n_nodes_emb = feature_map.shape[1]#len(ontology_mapping)
    n_nodes_annot = feature_map.shape[1]#len(ontology_mapping)
    
    # Create the model and move it to device
    model = OntologyNN(
        n_features=feature_data_train.shape[1],
        n_nodes=feature_map.shape[1],
        n_nodes_annot=feature_map.shape[1],
        n_nodes_emb=feature_map.shape[1],
        n_prop1=1,
        adj_mat_fc1=feature_map,
        propagation=args.propagation,  # GCNPropagation or DAGProp
        selection='top',
        ratio=1.0,
        out_channels=len(set(y_train)),
        out_activation=None,
        task='classification',
        dropout_rate=0.3
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5, verbose=True)
    
    train_losses = []
    test_losses = []
    
    epochs = args.epochs
    for epoch in range(epochs):
        model.train()
        batch_train_loss = 0
        for feature_batch, target_batch in train_loader:
            # Move batch data to device
            feature_batch = feature_batch.to(device)
            target_batch = target_batch.to(device)
            
            optimizer.zero_grad()
            
            # Create batched graph data on device
            graph_data_batch = Data(
                x=torch.ones(n_nodes, n_nodes_emb, device=device),
                edge_index=edge_index,
                batch=torch.zeros(n_nodes, dtype=torch.int64, device=device)
            )
            
            # Forward pass
            output_batch = model(feature_batch, graph_data_batch)
            loss = criterion(output_batch, target_batch)
            loss.backward()
            optimizer.step()
            
            batch_train_loss += loss.item()
        
        train_losses.append(batch_train_loss / len(train_loader))
        
        # Evaluate on test set
        model.eval()
        batch_test_loss = 0
        with torch.no_grad():
            for feature_batch, target_batch in test_loader:
                feature_batch = feature_batch.to(device)
                target_batch = target_batch.to(device)
                graph_data_batch = Data(
                    x=torch.ones(n_nodes, n_nodes_emb, device=device),
                    edge_index=edge_index,
                    batch=torch.zeros(n_nodes, dtype=torch.int64, device=device)
                )
                output_batch = model(feature_batch, graph_data_batch)
                loss = criterion(output_batch, target_batch)
                batch_test_loss += loss.item()
        
        test_losses.append(batch_test_loss / len(test_loader))
        scheduler.step(batch_test_loss / len(test_loader))
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')
    
    # save the model checkpoint
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Define filename with timestamp
    filename = f"checkpoint_{timestamp}.pth"
    
    torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
    }, filename)
    
    # Plot losses and save figure (since we use Agg backend)
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.legend()
    plt.savefig("loss_plot.png")
    
    # For evaluation, move feature tensors to device
    feature_data_train = feature_data_train.to(device)
    feature_data_test = feature_data_test.to(device)
    
    # Prepare a graph_data_batch for evaluation (structure remains constant)
    graph_data_batch = Data(
        x=torch.ones(n_nodes, n_nodes_emb, device=device),
        edge_index=edge_index,
        batch=torch.zeros(n_nodes, dtype=torch.int64, device=device)
    )
    
    return model, feature_data_train, y_train_tensor, feature_data_test, y_test_tensor, graph_data_batch#, list(ontology_mapping.keys())

def evaluate_model(model, X_train, y_train, X_test, y_test, graph_data_batch, device):
    model.eval()
    with torch.no_grad():
        train_predictions = model(X_train, graph_data_batch)
        train_predicted_labels = torch.argmax(train_predictions, dim=1)
        train_accuracy = accuracy_score(y_train.cpu(), train_predicted_labels.cpu())
        
        test_predictions = model(X_test, graph_data_batch)
        test_predicted_labels = torch.argmax(test_predictions, dim=1)
        test_accuracy = accuracy_score(y_test.cpu(), test_predicted_labels.cpu())
    
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    args = parse_args()
    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    X_train, X_test, y_train, y_test, feature_map, edge_index, ontology_node_list = load_data(args.dataset)
    model, X_train, y_train, X_test, y_test, graph_data_batch = train_model(
        X_train, X_test, y_train, y_test, feature_map, edge_index, args, device)
    
    evaluate_model(model, X_train, y_train, X_test, y_test, graph_data_batch, device)
    
    #print(torch.sigmoid(model.selection.ratio))
    
    if args.interpret is not None:
        sample_index = None if args.interpret is True else args.interpret
        interpret(model, graph_data_batch.edge_index, X_test, ontology_node_list, pred_label=args.label, sample_index=sample_index)
