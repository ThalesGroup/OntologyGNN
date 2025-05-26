import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import warnings
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
from matplotlib import pyplot as plt
from src.train import train_model
from src.utils import load_data, load_config, override_config, parse_args
import yaml
import logging    
    
# Main execution block
if __name__ == "__main__":

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    config = load_config("config.yaml")
    args = parse_args()
    config = override_config(config, args)    

    data_specs = config['data']
    data_directory = data_specs['data_directory']
    
    train_loader, test_loader, edge_index, ontology_node_list = load_data(
            data_directory,
            device=device,
            data_config=data_specs
        )

    experiment_path = setup_logger(
        log_dir=config["experiment"]["log_dir"],
        experiment_name=config["experiment"]["name"]
    )

    logging.info("Experiment started.")
    logging.info(f"Hyperparameters: {config['hyperparams']}")
    logging.info(f"Model config: {config['model']}")

    # Train the model
    logging.info("Starting training...")
    model, test_losses, modularities = train_model(
        train_loader,
        test_loader,
        device=device,
        node_list=ontology_node_list,
        config=config,
        print_stats=True,
        plot=True # Set to False if you don't want plots saved
    )
    logging.info("Training finished.")

    # Evaluate the model
    logging.info("\nEvaluating model performance...")
    model.evaluate_model(train_loader, test_loader, device=device)
    
    # Analyze communities
    logging.info("\nAnalyzing detected communities...")
    # Ensure criterion is defined for importance calculation
    criterion = nn.CrossEntropyLoss().to(device) # Define criterion again if not global or passed
    
    logging.info("\nCalculating community importance...")
    comm_importance = model.compute_community_importance(
        test_loader,
        criterion,
        device=device,
        num_communities=model.num_communities,
        print_stats=False # Set to True for detailed output per sample
    )
    logging.info(f"Community Importance (per community): {comm_importance.tolist()}")
    most_imp_comm_idx = torch.argmax(comm_importance).item()
    logging.info(f"Most important community index: {most_imp_comm_idx}")
    
    logging.info("\nIdentifying important nodes in the most important community...")
    important_nodes_freq = model.important_community_nodes(
        test_loader,
        device=device,
        criterion=criterion,
        print_stats=False # Set to True for detailed output per sample
    )
    
    # Sort and print top nodes
    sorted_nodes = sorted(important_nodes_freq.items(), key=lambda item: item[1], reverse=True)
    logging.info("Top nodes in the most important community:")
    for node_name, freq in sorted_nodes[:20]: # Print top 20
        logging.info(f"  {node_name}: appears in {freq} samples")
        

    save = config['experiment']['save']
    
    if save:
        save_dir = config['experiment']['save_dir']
        # Create the save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
    
        save_path = os.path.join(config['experiment']['save_dir'], config['experiment']['name']+'.pt')
        # Need to import os
        torch.save(model.state_dict(), save_path)
        logging.info(f"Model saved to {save_path}")
    