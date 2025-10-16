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

import os

import matplotlib
import torch
from torch import nn

matplotlib.use("Agg")  # Use Agg backend for non-interactive plotting
import logging
from collections import Counter, defaultdict

from .train import train_model
from .utils import load_config, load_data, parse_args, setup_logger

# Main execution block
if __name__ == "__main__":
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config("config.yaml")
    args = parse_args()
    #    config = override_config(config, args)

    data_specs = config["data"]
    data_directory = data_specs["data_directory"]

    train_loader, test_loader, edge_index, ontology_node_list = load_data(
        data_directory, device=device, data_config=data_specs,
    )

    experiment_path = setup_logger(
        log_dir=config["experiment"]["log_dir"],
        experiment_name=config["experiment"]["name"],
    )
    logging.info("Experiment started.")
    logging.info(f"Using device: {device}")
    logging.info(f"Hyperparameters: {config['hyperparams']}")
    logging.info(f"Model config: {config['model']}")
    logging.info(f"Data: {config['data']}")

    repeat_analysis = config["experiment"].get("repeat_analysis", 1)
    cumulative_node_freq = Counter()
    cumulative_edge_freq = defaultdict(int)

    for run_idx in range(repeat_analysis):
        logging.info(
            f"\n==================== Run {run_idx + 1} / {repeat_analysis} ====================",
        )

        # Load the data.
        # Reloading the data at each run enable to randomly select the training/test data
        # and avoid non pertinent impotant nodes selection for "small" test datasets
        if data_specs["shuffle_datasets"]:
            print("Shuffle the datasets")
            train_loader, test_loader, edge_index, ontology_node_list = load_data(
                data_directory, device=device, data_config=data_specs,
            )

        # Train the model
        logging.info("Starting training...")
        model, test_losses, modularities = train_model(
            train_loader,
            test_loader,
            device=device,
            node_list=ontology_node_list,
            config=config,
            print_stats=True,
            plot=True,  # Set to False if you don't want plots saved
        )
        logging.info("Training finished.")

        # Evaluate the model
        logging.info("Evaluating model performance...")
        metrics = model.evaluate_model(train_loader, test_loader, device=device)
        for k, v in metrics.items():
            logging.info(f"{k}: {v:.4f}")

        # Community-based interpretation
        if config["experiment"]["interpretation_label"] is not None:
            logging.info(
                f"Analyzing communities for label: {config['experiment']['interpretation_label']}",
            )
        else:
            logging.info("Analyzing communities for all predicted labels")

        criterion = nn.CrossEntropyLoss().to(device)

        logging.info("Calculating community importance...")
        comm_importance = model.compute_community_importance(
            test_loader,
            criterion,
            device=device,
            num_communities=model.num_communities,
            print_stats=False,
            filter_label=config["experiment"]["interpretation_label"],
        )
        most_imp_comm_idx = torch.argmax(comm_importance).item()
        logging.info(f"Most important community index: {most_imp_comm_idx}")

        logging.info("Identifying important nodes in the most important community...")
        important_nodes_freq = model.important_community_nodes(
            test_loader,
            device=device,
            criterion=criterion,
            print_stats=False,
            filter_label=config["experiment"]["interpretation_label"],
        )
        cumulative_node_freq.update(important_nodes_freq)

        logging.info("Identifying important edges in the most important community...")
        imp_edges = model.important_community_edges(
            test_loader,
            edge_index=edge_index,
            device=device,
            criterion=criterion,
            print_stats=False,
            filter_label=config["experiment"]["interpretation_label"],
        )
        for edge, count in imp_edges.items():
            cumulative_edge_freq[edge] += count

    # Final result logging after all runs
    sorted_nodes = sorted(
        cumulative_node_freq.items(), key=lambda item: item[1], reverse=True,
    )
    logging.info(f"\n=== Top nodes across {repeat_analysis} runs ===")
    logging.info(f"   {len(sorted_nodes)} nodes in the most important community")
    for node_name, freq in sorted_nodes[:20]:
        logging.info(f"  {node_name}: appears {freq} times")

    sorted_edges = sorted(
        cumulative_edge_freq.items(), key=lambda item: item[1], reverse=True,
    )
    logging.info(f"\n=== Top edges across {repeat_analysis} runs ===")
    logging.info(f"   {len(sorted_edges)} edges in the most important community")
    for (i, j), count in sorted_edges[:20]:
        name_i = ontology_node_list[i]
        name_j = ontology_node_list[j]
        logging.info(f"  ({name_i}, {name_j}): appears in {count} samples")

    # Save final model and results from last run
    save = config["experiment"]["save"]
    if save:
        save_dir = config["experiment"]["save_dir"]
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(
            config["experiment"]["save_dir"], config["experiment"]["name"] + ".pt",
        )

        torch.save(model.state_dict(), save_path)
        logging.info(f"Model saved to {save_path}")
