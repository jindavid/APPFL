"""
Serial simulation of SCAFFOLD Federated learning.
It should be noted that only synchronous FL can be simulated in this way.
"""

import argparse
from omegaconf import OmegaConf
from appfl.agent import ClientAgent, ServerAgent
import torch
import numpy as np

import wandb

def compute_model_differences(local_model, global_model):
    """
    Compute sum of squared differences between local and global model weights.
    
    Args:
        local_model: Structured dict with 'model_state' section
        global_model: Structured dict with 'model_state' section
        
    Returns:
        float: Sum of squared differences
    """
    # Extract model states from structured format
    local_model_state = local_model["model_state"]
    global_model_state = global_model["model_state"]
    
    all_diffs = []
    
    for param_name in local_model_state.keys():
        if param_name in global_model_state:
            local_param = local_model_state[param_name]
            global_param = global_model_state[param_name]
            
            # Compute squared differences
            diff = (local_param - global_param) ** 2
            all_diffs.append(diff.flatten())
    
    # Concatenate all parameter differences
    all_diffs = torch.cat(all_diffs)
    
    diff = torch.sum(all_diffs).item()
    
    return diff

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--server_config", type=str, default="./resources/configs/mnist/server_scaffold.yaml"
)
argparser.add_argument(
    "--client_config", type=str, default="./resources/configs/mnist/client_1.yaml"
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

while not server_agent.training_finished():
    local_models_for_comparison = []
    
    for i, client_agent in enumerate(client_agents):
        # Client local training
        client_agent.train()
        local_model = client_agent.get_parameters()
        if isinstance(local_model, tuple):
            local_model, metadata = local_model[0], local_model[1]
        else:
            metadata = {}
        
        # Store local model for comparison later
        local_models_for_comparison.append((client_agent.get_id(), local_model.copy()))
        
        # "Send" local model to server 
        # For SCAFFOLD, only block on the last client to ensure proper aggregation
        is_last_client = (i == len(client_agents) - 1)
        server_agent.global_update(
            client_id=client_agent.get_id(),
            local_model=local_model,
            blocking=is_last_client,
            **metadata,
        )
    
    # Get the aggregated global model with control variates
    aggregated_global_model = server_agent.get_parameters(serial_run=True)
    
    # Compute differences between local models and aggregated global model
    diffs = []
    
    print("\nComputing model differences:")
    for client_id, local_model in local_models_for_comparison:
        diff = compute_model_differences(local_model, aggregated_global_model)
        diffs.append(diff)
    
    # Compute statistics across all clients
    overall_max_diff = max(diffs)
    overall_mean_diff = np.mean(diffs)
    
    # Load the new global model to all clients
    for client_agent in client_agents:
        client_agent.load_parameters(aggregated_global_model)
    
    val_loss, val_accuracy = server_agent.server_validate()
    
    # Log all metrics to wandb
    wandb.log({
        "val_loss": val_loss, 
        "val_accuracy": val_accuracy,
        "max_diff": overall_max_diff,
        "mean_diff": overall_mean_diff,
    }) 