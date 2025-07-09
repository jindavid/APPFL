"""
Weighted Gradient Federated Learning Serial Simulation.

This script demonstrates the weighted gradient federated learning algorithm
where gradients are weighted by label distribution ratios: p(x) / p_i(x).

The algorithm computes label distribution weights before training starts and
applies them during client training to handle non-IID data distributions.
Uses standard FedAvg aggregator with weighted gradient training.
"""

import argparse
from omegaconf import OmegaConf
from appfl.agent import ClientAgent, ServerAgent
from appfl.algorithm.trainer.weighted_gradient_trainer import compute_global_label_distribution
from torch.utils.data import DataLoader
import torch
import numpy as np

import wandb

def compute_model_differences(local_model, global_model):
    """
    Compute max and mean differences between local and global model weights.
    
    Args:
        local_model: Local model state dict
        global_model: Global model state dict
        
    Returns:
        float: diff
    """
    all_diffs = []
    
    for param_name in local_model.keys():
        if param_name in global_model:
            local_param = local_model[param_name]
            global_param = global_model[param_name]
            
            # Compute absolute differences
            diff = (local_param - global_param) ** 2
            all_diffs.append(diff.flatten())
    
    # Concatenate all parameter differences
    all_diffs = torch.cat(all_diffs)
    
    diff = torch.sum(all_diffs).item()
    
    return diff

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--server_config", type=str, default="./resources/configs/cifar10/server_fedavg_weighted_gradient.yaml"
)
argparser.add_argument(
    "--client_config", type=str, default="./resources/configs/cifar10/client_1.yaml"
)
argparser.add_argument("--num_clients", type=int, default=10)
args = argparser.parse_args()

# Load server agent configurations and set the number of clients
server_agent_config = OmegaConf.load(args.server_config)
server_agent_config.server_configs.num_clients = args.num_clients

# Create server agent
server_agent = ServerAgent(server_agent_config=server_agent_config)

# Load base client configurations and set corresponding fields for different clients
client_agent_configs = [
    OmegaConf.load(args.client_config) for _ in range(args.num_clients)
]
for i in range(args.num_clients):
    client_agent_configs[i].client_id = f"Client{i + 1}"
    client_agent_configs[i].data_configs.dataset_kwargs.num_clients = args.num_clients
    client_agent_configs[i].data_configs.dataset_kwargs.client_id = i
    client_agent_configs[i].data_configs.dataset_kwargs.visualization = (
        True if i == 0 else False
    )
    
    # only enable wandb for the first client is sufficient for logging all clients in serial run
    if hasattr(client_agent_configs[i], "wandb_configs") and client_agent_configs[
        i
    ].wandb_configs.get("enable_wandb", False):
        if i == 0:
            client_agent_configs[i].wandb_configs.enable_wandb = True
        else:
            client_agent_configs[i].wandb_configs.enable_wandb = False

# Load client agents
client_agents = [
    ClientAgent(client_agent_config=client_agent_configs[i])
    for i in range(args.num_clients)
]

# Save both the server and client config files to wandb
wandb.save(args.server_config)
wandb.save(args.client_config)

# Get additional client configurations from the server
client_config_from_server = server_agent.get_client_configs()
for client_agent in client_agents:
    client_agent.load_config(client_config_from_server)

# Load initial global model from the server
init_global_model = server_agent.get_parameters(serial_run=True)
for client_agent in client_agents:
    client_agent.load_parameters(init_global_model)

# [Optional] Set number of local data to the server
for i in range(args.num_clients):
    sample_size = client_agents[i].get_sample_size()
    server_agent.set_sample_size(
        client_id=client_agents[i].get_id(), sample_size=sample_size
    )

# NEW: Compute global label distribution before training
print("Computing global label distribution for weighted gradient FL...")

# Get client data loaders for global distribution computation
client_data_loaders = {}
for i, client_agent in enumerate(client_agents):
    # Access the client's training dataset
    train_dataset = client_agent.trainer.train_dataset
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,  # Use a reasonable batch size for analysis
        shuffle=False,  # Don't shuffle for consistent analysis
        num_workers=0
    )
    client_data_loaders[client_agent.get_id()] = train_dataloader

# Compute global label distribution across all clients
global_label_distribution = compute_global_label_distribution(client_data_loaders)
print(f"Global label distribution: {global_label_distribution}")

# Set global distribution in each client's trainer to compute local weights
for client_agent in client_agents:
    if hasattr(client_agent.trainer, 'set_global_label_distribution'):
        client_agent.trainer.set_global_label_distribution(global_label_distribution)
        print(f"Set global distribution for {client_agent.get_id()}")

print("Label distribution weights computed successfully!")

while not server_agent.training_finished():
    new_global_models = []
    local_models_for_comparison = []
    
    for i, client_agent in enumerate(client_agents):
        print(f"Training client {client_agent.get_id()}...")
        
        # Client local training (weights are already computed in trainer)
        client_agent.train()
        local_model = client_agent.get_parameters()
        if isinstance(local_model, tuple):
            local_model, metadata = local_model[0], local_model[1]
        else:
            metadata = {}
        
        # Store local model for comparison later
        local_models_for_comparison.append((client_agent.get_id(), local_model.copy()))
        
        # "Send" local model to server and get a Future object for the new global model
        # The Future object will be resolved when the server receives local models from all clients
        new_global_model_future = server_agent.global_update(
            client_id=client_agent.get_id(),
            local_model=local_model,
            blocking=False,
            **metadata,
        )
        new_global_models.append(new_global_model_future)
    
    # Get the aggregated global model
    aggregated_global_model = new_global_models[0].result()  # All futures return the same global model
    
    # Compute differences between local models and aggregated global model
    diffs = []
    
    print("\nComputing model differences:")
    for client_id, local_model in local_models_for_comparison:
        diff = compute_model_differences(local_model, aggregated_global_model)
        diffs.append(diff)
    
    # Compute statistics across all clients
    overall_max_diff = max(diffs)
    overall_mean_diff = np.mean(diffs)
    
    # Load the new global model from the server (standard FedAvg aggregation)
    for client_agent, new_global_model_future in zip(client_agents, new_global_models):
        client_agent.load_parameters(new_global_model_future.result())
    
    val_loss, val_accuracy = server_agent.server_validate()    
    
    # Log all metrics to wandb
    wandb.log({
        "val_loss": val_loss, 
        "val_accuracy": val_accuracy,
        "max_diff": overall_max_diff,
        "mean_diff": overall_mean_diff,
    })
    
    # # Log individual client differences
    # for i, (client_id, _) in enumerate(local_models_for_comparison):
    #     wandb.log({
    #         f"{client_id}_max_diff": max_diffs[i],
    #         f"{client_id}_mean_diff": mean_diffs[i],
    #         "round": round_num
    #     })


print("Weighted gradient federated learning completed!") 