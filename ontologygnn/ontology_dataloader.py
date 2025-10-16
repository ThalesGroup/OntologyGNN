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

# example run script
# python ontology_dataloader titanic_ontology.obo --connect_features 'Cabin' 'Age' --save ontology ontology_data_connections.json  (allows the user to connect specified features to classes and then save the connections)


import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import obonet
import pandas as pd
import torch
from owlready2 import *

DEBUG = True


class OntologyDataLoader:
    def __init__(self, filepath):
        self.ontology = self.load_ontology(filepath)

    @staticmethod
    def shorten_uri(uri):
        if isinstance(uri, str):
            return uri.split("#")[-1] if "#" in uri else uri.split("/")[-1]
        return str(uri).split("#")[-1] if "#" in str(uri) else str(uri).split("/")[-1]

    @staticmethod
    def load_rdf(owl_filepath, complete):
        onto = get_ontology(owl_filepath).load()
        G = nx.DiGraph()
        for cls in onto.classes():
            G.add_node(cls.name)
            for sub_cls in cls.subclasses():
                G.add_node(sub_cls.name)
                G.add_edge(sub_cls.name, cls.name, label="is-a")
            if complete:
                for i in cls.instances():
                    G.add_node(i.name)
                    G.add_edge(i.name, cls.name, label="instance")
                    for prop in i.get_properties():
                        for value in prop[i]:
                            G.add_node(str(value))
                            G.add_edge(str(value), i.name, label=prop.python_name)
        print("Loaded", filepath, "with owlready2")
        if DEBUG:
            print(G)
        return G

    @staticmethod
    def load_ontology(filepath, complete=True):
        if filepath.endswith(".obo"):
            try:
                graph = obonet.read_obo(filepath)
                print("Loaded", filepath, "with obonet")
                return graph
            except Exception as e:
                print(f"Error loading .obo file: {e}")
                return None

        elif filepath.endswith((".ttl", ".owl", ".rdf", ".xml")):
            owl_filepath = ""
            if filepath.endswith(".ttl"):
                try:
                    from rdflib import Graph

                    g = Graph()
                    g.parse(filepath, format="turtle")
                    owl_filepath = "".join([filepath[0:-4], ".rdf"])
                    # print("TTL ontology : "+filepath)
                    # print("RDF ontology : "+owl_filepath)
                    g.serialize(destination=owl_filepath, format="xml")

                except Exception as e:
                    print(f"Error loading .ttl file with rdflib: {e}")
                    return None
            else:
                owl_filepath = filepath

            try:
                onto = get_ontology(owl_filepath).load()
                G = nx.DiGraph()
                for cls in onto.classes():
                    G.add_node(cls.name)
                    for sub_cls in cls.subclasses():
                        G.add_node(sub_cls.name)
                        G.add_edge(sub_cls.name, cls.name, label="is-a")
                    if complete:
                        for i in cls.instances():
                            G.add_node(i.name)
                            G.add_edge(i.name, cls.name, label="instance")
                            for prop in i.get_properties():
                                for value in prop[i]:
                                    G.add_node(str(value))
                                    G.add_edge(
                                        str(value), i.name, label=prop.python_name,
                                    )
                print("Loaded", filepath, "with owlready2")
                if DEBUG:
                    print(G)
                return G
            except Exception as e:
                print(f"Error loading OWL file: {e}")
                return None

        else:
            print("Unsupported file format.")
            return None

    def add_edge(self, parent_node, child_node, label=None):
        if self.ontology.has_edge(parent_node, child_node):
            print(f"Edge between {parent_node} and {child_node} already exists.")
        else:
            self.ontology.add_edge(parent_node, child_node, label=label)
            print(
                f"Added edge between {parent_node} and {child_node} with label '{label}'",
            )

    def remove_edge(self, parent_node, child_node):
        self.ontology.remove_edge(parent_node, child_node)
        print("Removed edge between", parent_node, "and", child_node)

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
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=300,
            node_color="lightblue",
            font_size=10,
            font_weight="bold",
            arrows=True,
        )
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
            selected_indices = input(
                "Enter the indices of nodes to connect, separated by commas: ",
            )
            selected_indices = [int(idx.strip()) for idx in selected_indices.split(",")]
            for idx in selected_indices:
                if 0 <= idx < len(nodes):
                    self.add_edge(feature, nodes[idx], label="feature-to-node")
                else:
                    print(f"Invalid index: {idx}")

    def save_connections_with_categories(
        self, targets, features, filename="connections_with_categories.json",
    ):
        connections = {
            "targets": {},
            "features to classes connections": {},
            "classes to classes connections": {},
        }

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

    def connect_data_with_ontology(self, features_dict):
        # G = self.load_ontology()
        edges = self.ontology.edges(data=True)

        relation_matrix = [
            (src, dst, data["label"])
            for src, dst, data in edges
            # if data.get('label') == 'instance'
        ]

        # Step 1: Map instance -> set of classes
        instance_to_classes = defaultdict(set)
        for instance, cls, label in relation_matrix:
            if label == "instance":
                instance_to_classes[instance.lower()].add(cls)

        # Step 2: Extract all unique classes from the relation matrix (subject/object in non-'instance' triples)
        all_classes = set()
        for subj, obj, label in relation_matrix:
            if label != "instance":
                all_classes.update([subj, obj])

        class_features_mapping = defaultdict(set)

        # Step 3: Match features to classes
        for feature, values in features_dict.items():
            safe_values = [v.lower() for v in values if isinstance(v, str)]
            all_candidates = [feature.lower()] + safe_values

            matched_classes = set()
            for candidate in all_candidates:
                matched_classes.update(instance_to_classes.get(candidate, []))

            for cls in matched_classes:
                class_features_mapping[cls].add(feature)

        # Step 4: Ensure all classes are present, even if empty
        final_mapping = {}
        for cls in all_classes:
            features = class_features_mapping.get(cls, set())
            final_mapping[cls] = sorted(features)

        class_feature_map = {"class_features_mapping": final_mapping}

        # Get class relationships among them
        # Step 1: Get list of classes in the order of class_features_mapping keys
        class_list = list(final_mapping.keys())
        class_to_index = {cls: idx for idx, cls in enumerate(class_list)}

        # Step 2: Extract class-to-class relationships from non-'instance' entries
        alledges = set()
        for subj, obj, label in relation_matrix:
            if label != "instance" and subj in class_to_index and obj in class_to_index:
                i, j = class_to_index[subj], class_to_index[obj]
                alledges.add((i, j))

        # Step 3: Return sorted list of unique edges
        class_relations = {"class_relationships": sorted(list(alledges))}

        return class_feature_map, class_relations


def main():
    parser = argparse.ArgumentParser(description="Ontology Data Loader and Processor")
    parser.add_argument(
        "filepath", type=str, help="Path to the ontology file (.owl, .rdf, .xml, .obo)",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize the ontology graph",
    )
    parser.add_argument(
        "--save", type=str, help="Save the connections with categories to a JSON file",
    )
    parser.add_argument(
        "--connect_features",
        nargs="+",
        help="Specify feature names to connect to nodes interactively",
    )
    parser.add_argument(
        "--connect_classes",
        nargs="+",
        help="Specify classes names to connect to nodes interactively",
    )
    parser.add_argument("--add_target", nargs="+", help="Specify target feature")
    parser.add_argument(
        "--data", required=False, help="Specify the ontology based data file path",
    )
    args = parser.parse_args()

    loader = OntologyDataLoader(args.filepath)

    if loader.ontology is None:
        print("Failed to load ontology.")
        return

    if args.data:
        data = pd.read(args.data)
        # Get data features and unique values of each feature
        feature_unique_items = {}
        for col in data.columns:
            feature_unique_items[col] = data[col].unique().tolist()

        all_relations = loader.connect_data_with_ontology(feature_unique_items)

    if args.visualize:
        loader.visualize_ontology()

    if args.connect_features:
        print("connecting features to classes")
        loader.connect_features_to_nodes(args.connect_features)

    if args.connect_classes:
        print("connecting classes to other classes")
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
