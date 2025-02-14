# example run script
# python ontology_dataloader titanic_ontology.obo --connect_features 'Cabin' 'Age' --save ontology ontology_data_connections.json  (allows the user to connect specified features to classes and then save the connections)


import torch
import obonet
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from owlready2 import *
import json
import argparse

class OntologyDataLoader:
    def __init__(self, filepath):
        """Initialize the OntologyProcessor class and load the ontology."""
        self.ontology = self.load_ontology(filepath)

    @staticmethod
    def load_ontology(filepath):
        """Loads an ontology file using either owlready2 or obonet based on the file extension.

        Args:
            filepath: The path to the ontology file.

        Returns:
            The loaded ontology object, or None if the file extension is not recognized.
        """
        if filepath.endswith((".owl", ".rdf", ".xml")):
            try:
                onto = get_ontology(filepath).load()
                G = nx.DiGraph()
                for cls in onto.classes():
                    G.add_node(cls.name, shape="ellipse", style="filled", fillcolor="lightblue")
                    for sub_cls in cls.subclasses():
                        G.add_node(sub_cls.name, shape="ellipse", style="filled", fillcolor="lightgreen")
                        G.add_edge(sub_cls.name, cls.name, label="is-a", color="blue")
                print('loaded', filepath, 'successfully')
                return G
            except Exception as e:
                print(f"Error loading ontology with owlready2: {e}")
                return None
        elif filepath.endswith(".obo"):
            try:
                graph = obonet.read_obo(filepath)
                print('loaded', filepath, 'successfully')
                return graph
            except Exception as e:
                print(f"Error loading ontology with obonet: {e}")
                return None
        else:
            print(f"Unsupported file extension for ontology file: {filepath}")
            return None

    def add_edge(self, parent_node, child_node, label=None):
        if self.ontology.has_edge(parent_node, child_node):
            print(f"Edge between {parent_node} and {child_node} already exists.")
        else:
            self.ontology.add_edge(parent_node, child_node, label=label)
            print(f"Added edge between {parent_node} and {child_node} with label '{label}'")

    def remove_edge(self, parent_node, child_node):
        self.ontology.remove_edge(parent_node, child_node)
        print('Removed edge between', parent_node, 'and', child_node)

    def get_adjacency_matrix(self):
        if self.ontology is None:
            return None
        adjacency_matrix = nx.adjacency_matrix(self.ontology).todense()
        node_names = list(self.ontology.nodes())
        df = pd.DataFrame(adjacency_matrix, index=node_names, columns=node_names)
        return df

    def visualize_ontology(self):
        if self.ontology is None:
            print("No ontology loaded to visualize.")
            return

        G = self.ontology
        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(15, 15))
        nx.draw(G, pos, with_labels=True, node_size=300, node_color='lightblue', font_size=10, font_weight="bold", arrows=True)
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.show()

    def get_feature_node_map(self, features):

        """Returns the (n_features, n_nodes) dimensional matrix denoting connections of given features to other nodes in the ontology"""

        if features is None:
            return None

        adj_matrix = self.get_adjacency_matrix()

        feature_node_map = adj_matrix.loc[features]
        feature_node_map = feature_node_map.drop(columns=features)

        return feature_node_map

    def get_node_map(self, features):

        """Returns the (n_nodes, n_nodes) dimensional matrix denoting connections of nodes (classes) to other nodes in the ontology"""
        """It doesn't include the connections to features (if any specified by the user)"""

        if features is None:
            return self.get_adjacency_matrix()

        adj_matrix = self.get_adjacency_matrix()

        node_map = adj_matrix.drop(columns=features, index=features)

        # Convert the adjacency matrix to a NumPy array
        adj_array = node_map.to_numpy()

        # Find non-zero elements (edges)
        rows, cols = np.where(adj_array)

        # Create the edge index
        edge_index = np.stack([rows, cols], axis=0)

        # Convert to a PyTorch tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        return node_map, edge_index

    def connect_features_to_nodes(self, features):
        """Allows the user to connect features to nodes in the ontology.

        Args:
            features: A list of feature names to be connected.
        """
        if self.ontology is None:
            print("No ontology loaded.")
            return

        nodes = list(self.ontology.nodes())
        print("Available nodes in the ontology:")
        for i, node in enumerate(nodes):
            print(f"{i}: {node}")

        for feature in features:
            print(f"\nConnecting: {feature}")
            selected_indices = input("Enter the indices of nodes to connect, separated by commas: ")
            selected_indices = [int(idx.strip()) for idx in selected_indices.split(',')]
            for idx in selected_indices:
                if 0 <= idx < len(nodes):
                    self.add_edge(feature, nodes[idx], label="feature-to-node")
                else:
                    print(f"Invalid index: {idx}")

    def save_connections_with_categories(self, targets, features, filename="connections_with_categories.json"):
        connections = {"targets": {}, "features to classes connections": {}, 
        "classes to classes connections": {}}

        for node in self.ontology.nodes():
            connected_nodes = list(self.ontology.successors(node))
            if node in targets:
                connections["targets"][node] = connected_nodes
            elif node in features:
                connections["features to classes connections"][node] = connected_nodes
            else:
                connections["classes to classes connections"][node] = connected_nodes

        with open(filename, "w") as f:
            json.dump(connections, f, indent=4)
        print(f"Connections with categories saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Ontology Data Loader and Processor")
    parser.add_argument("filepath", type=str, help="Path to the ontology file (.owl, .rdf, .xml, .obo)")
    parser.add_argument("--visualize", action="store_true", help="Visualize the ontology graph")
    parser.add_argument("--save", type=str, help="Save the connections with categories to a JSON file")
    parser.add_argument("--connect_features", nargs='+', help="Specify feature names to connect to nodes interactively")
    parser.add_argument("--connect_classes", nargs='+', help="Specify classes names to connect to nodes interactively")
    parser.add_argument("--add_target", nargs='+', help="Specify target feature")
    args = parser.parse_args()

    loader = OntologyDataLoader(args.filepath)

    if loader.ontology is None:
        print("Failed to load ontology.")
        return

    if args.visualize:
        loader.visualize_ontology()

    if args.connect_features:
        print('connecting features to classes')
        loader.connect_features_to_nodes(args.connect_features)

    if args.connect_classes:
        print('connecting classes to other classes')
        loader.connect_features_to_nodes(args.connect_classes)

    if args.save:
        if args.add_target:
            targets = args.add_target
        else:
            targets = []
        if args.connect_features:
            features = args.connect_features 
        else:
            features = []
        loader.save_connections_with_categories(targets, features, args.save)


if __name__ == "__main__":
    main()
