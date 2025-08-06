"""
Unified Federated Learning Serial Simulation.

This script provides a unified interface for running federated learning experiments
across multiple datasets (CIFAR10, AG_NEWS, MNIST, etc.) and algorithms (FedAvg, SCAFFOLD).
It automatically detects the aggregator type and dataset from configuration files
and applies the appropriate communication patterns and feature handling.

Features:
- Multi-dataset support (CIFAR10, AG_NEWS, MNIST, etc.)
- Multi-algorithm support (FedAvg, SCAFFOLD, etc.)
- Automatic aggregator detection
- Gradient weighting for non-IID data
- Flexible data partitioning strategies
- Comprehensive logging and metrics
"""

import argparse
import os
from omegaconf import OmegaConf
from appfl.agent import ClientAgent, ServerAgent
from appfl.algorithm.trainer.weighted_gradient_trainer import compute_global_label_distribution
from torch.utils.data import DataLoader
import torch
import numpy as np
import wandb

def compute_model_differences(local_model, global_model):
    """
    Compute sum of squared differences between local and global model weights.
    Handles both flat parameter dicts (FedAvg) and structured dicts (SCAFFOLD).
    
    Args:
        local_model: Local model parameters (dict or structured dict)
        global_model: Global model parameters (dict or structured dict)
        
    Returns:
        float: Sum of squared differences
    """
    # Extract model state dicts from potentially structured formats
    def extract_model_state(model_params):
        if isinstance(model_params, dict) and "model_state" in model_params:
            return model_params["model_state"]
        else:
            return model_params
    
    local_state = extract_model_state(local_model)
    global_state = extract_model_state(global_model)
    
    all_diffs = []
    
    for param_name in local_state.keys():
        if param_name in global_state:
            local_param = local_state[param_name]
            global_param = global_state[param_name]
            
            # Compute squared differences
            diff = (local_param - global_param) ** 2
            all_diffs.append(diff.flatten())
    
    # Concatenate all parameter differences
    all_diffs = torch.cat(all_diffs)
    
    diff = torch.sum(all_diffs).item()
    
    return diff

def compute_global_label_distribution_legacy(client_agents):
    """
    Legacy method for computing global label distribution from client agents directly.
    Used for datasets that don't support the weighted gradient trainer's method.
    
    Args:
        client_agents: List of client agents
        
    Returns:
        dict: Global label distribution {label: probability}
    """
    global_label_counts = {}
    total_samples = 0
    
    for client_agent in client_agents:
        # Get local label distribution
        train_dataset = client_agent.train_dataset
        if train_dataset is None:
            continue
            
        for _, label in train_dataset:
            label_idx = label.item() if torch.is_tensor(label) else int(label)
            global_label_counts[label_idx] = global_label_counts.get(label_idx, 0) + 1
            total_samples += 1
    
    # Convert to probabilities
    global_distribution = {}
    for label, count in global_label_counts.items():
        global_distribution[label] = count / total_samples
    
    print(f"Global label distribution: {global_distribution}")
    return global_distribution

def detect_dataset_type(config_path):
    """
    Detect dataset type from configuration file path.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        str: Dataset type (cifar10, agnews, mnist, etc.)
    """
    path_lower = config_path.lower()
    if 'cifar10' in path_lower:
        return 'cifar10'
    elif 'agnews' in path_lower:
        return 'agnews'
    elif 'mnist' in path_lower:
        return 'mnist'
    elif 'flamby' in path_lower:
        return 'flamby'
    elif 'adult' in path_lower:
        return 'adult'
    else:
        return 'unknown'

def setup_dataset_specific_configs(client_agent_configs, args, dataset_type):
    """
    Apply dataset-specific configuration adjustments.
    
    Args:
        client_agent_configs: List of client configurations
        args: Command line arguments
        dataset_type: Detected dataset type
    """
    for i in range(len(client_agent_configs)):
        client_agent_configs[i].client_id = f"Client{i + 1}"
        client_agent_configs[i].data_configs.dataset_kwargs.num_clients = args.num_clients
        client_agent_configs[i].data_configs.dataset_kwargs.client_id = i
        client_agent_configs[i].data_configs.dataset_kwargs.visualization = (
            True if i == 0 else False
        )
        
        # Apply dataset-specific partitioning configurations
        if dataset_type in ['agnews', 'cifar10'] and hasattr(client_agent_configs[i].data_configs.dataset_kwargs, 'partition_strategy'):
            if args.partition_strategy:
                client_agent_configs[i].data_configs.dataset_kwargs.partition_strategy = args.partition_strategy
            if args.alpha is not None:
                client_agent_configs[i].data_configs.dataset_kwargs.alpha2 = args.alpha

def setup_wandb_configs(client_agent_configs, exp_name):
    """
    Configure Weights & Biases logging for clients.
    
    Args:
        client_agent_configs: List of client configurations
        exp_name: Experiment name for logging
    """
    for i in range(len(client_agent_configs)):
        if hasattr(client_agent_configs[i], "wandb_configs") and client_agent_configs[i].wandb_configs.get("enable_wandb", False):
            client_agent_configs[i].wandb_configs.exp_name = exp_name
            if i == 0:
                client_agent_configs[i].wandb_configs.enable_wandb = True
            else:
                client_agent_configs[i].wandb_configs.enable_wandb = False

def setup_gradient_weighting(client_agents, dataset_type):
    """
    Set up gradient weighting for datasets that support it.
    
    Args:
        client_agents: List of client agents
        dataset_type: Detected dataset type
        
    Returns:
        bool: Whether gradient weighting was successfully set up
    """
    # Check if any client supports weighted gradient training
    supports_weighting = any(
        hasattr(client_agent.trainer, 'set_global_label_distribution') 
        for client_agent in client_agents
    )
    
    if not supports_weighting:
        print("Gradient weighting not supported by current trainer configuration")
        return False
    
    print("Computing global label distribution for gradient weighting...")
    
    try:
        # Try the modern weighted gradient trainer method first
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
        
    except Exception as e:
        print(f"Modern gradient weighting failed: {e}")
        print("Falling back to legacy method...")
        # Fall back to legacy method
        global_label_distribution = compute_global_label_distribution_legacy(client_agents)
    
    print(f"Global label distribution: {global_label_distribution}")
    
    # Set global distribution in each client's trainer
    for client_agent in client_agents:
        if hasattr(client_agent.trainer, 'set_global_label_distribution'):
            client_agent.trainer.set_global_label_distribution(global_label_distribution)
            print(f"Set global distribution for {client_agent.get_id()}")
    
    return True

def run_training_loop(server_agent, client_agents, is_scaffold, algorithm_name):
    """
    Execute the main federated learning training loop.
    
    Args:
        server_agent: Server agent instance
        client_agents: List of client agents
        is_scaffold: Whether using SCAFFOLD algorithm
        algorithm_name: Name of the algorithm for logging
    """
    print(f"\nStarting {algorithm_name} federated training...")
    round_num = 0

    while not server_agent.training_finished():
        round_num += 1
        print(f"\n=== Round {round_num} ===")
        
        local_models_for_comparison = []
        new_global_models = []
        
        for i, client_agent in enumerate(client_agents):
            print(f"Training {client_agent.get_id()}...")
            
            # Client local training
            try:
                client_agent.train(round=round_num-1)  # Pass round number for logging
            except TypeError:
                # Some trainers don't accept round parameter
                client_agent.train()
                
            local_model = client_agent.get_parameters()
            if isinstance(local_model, tuple):
                local_model, metadata = local_model[0], local_model[1]
            else:
                metadata = {}
            
            # Store local model for comparison later
            local_models_for_comparison.append((client_agent.get_id(), local_model.copy()))
            
            if is_scaffold:
                # SCAFFOLD communication pattern: block on last client
                is_last_client = (i == len(client_agents) - 1)
                server_agent.global_update(
                    client_id=client_agent.get_id(),
                    local_model=local_model,
                    blocking=is_last_client,
                    **metadata,
                )
            else:
                # FedAvg communication pattern: use futures
                new_global_model_future = server_agent.global_update(
                    client_id=client_agent.get_id(),
                    local_model=local_model,
                    blocking=False,
                    **metadata,
                )
                new_global_models.append(new_global_model_future)
        
        # Get the aggregated global model
        if is_scaffold:
            # For SCAFFOLD: get model with control variates from server
            aggregated_global_model = server_agent.get_parameters(serial_run=True)
        else:
            # For FedAvg: get model from futures
            aggregated_global_model = new_global_models[0].result()
        
        # Compute differences between local models and aggregated global model
        diffs = []
        
        print("Computing model differences:")
        for client_id, local_model in local_models_for_comparison:
            diff = compute_model_differences(local_model, aggregated_global_model)
            diffs.append(diff)
            print(f"  {client_id}: {diff:.6f}")
        
        # Compute statistics across all clients
        overall_max_diff = max(diffs)
        overall_mean_diff = np.mean(diffs)
        
        print(f"  Max difference: {overall_max_diff:.6f}")
        print(f"  Mean difference: {overall_mean_diff:.6f}")
        
        # Load the new global model to all clients
        if is_scaffold:
            # For SCAFFOLD: load the model with control variates
            for client_agent in client_agents:
                client_agent.load_parameters(aggregated_global_model)
        else:
            # For FedAvg: load from futures
            for client_agent, new_global_model_future in zip(client_agents, new_global_models):
                client_agent.load_parameters(new_global_model_future.result())
        
        # Server validation
        val_loss, val_accuracy = server_agent.server_validate()
        print(f"Global validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        
        # Log all metrics to wandb
        wandb.log({
            "round": round_num,
            "val_loss": val_loss, 
            "val_accuracy": val_accuracy,
            "max_diff": overall_max_diff,
            "mean_diff": overall_mean_diff,
        })

    print(f"\n{algorithm_name} training completed after {round_num} rounds!")

def main():
    # Command line argument parsing
    argparser = argparse.ArgumentParser(description="Unified Federated Learning Script")
    argparser.add_argument(
        "--server_config", type=str, required=True,
        help="Path to server configuration file"
    )
    argparser.add_argument(
        "--client_config", type=str, required=True,
        help="Path to client configuration file"
    )
    argparser.add_argument("--num_clients", type=int, default=10,
                          help="Number of clients")
    argparser.add_argument("--partition_strategy", type=str, default=None,
                          help="Data partitioning strategy: iid, class_noniid, dirichlet_noniid")
    argparser.add_argument("--alpha", type=float, default=None,
                          help="Dirichlet alpha parameter for non-IID partitioning")
    argparser.add_argument("--enable_gradient_weighting", action="store_true",
                          help="Enable gradient weighting (if supported by trainer)")
    args = argparser.parse_args()

    # Detect dataset type from configuration paths
    dataset_type = detect_dataset_type(args.server_config)
    print(f"Detected dataset type: {dataset_type}")
    
    if args.partition_strategy:
        print(f"Partition strategy: {args.partition_strategy}")
    if args.alpha is not None:
        print(f"Alpha parameter: {args.alpha}")

    # Generate experiment name from server config filename
    server_config_filename = os.path.basename(args.server_config)
    exp_name = os.path.splitext(server_config_filename)[0]

    # Load server agent configurations and set the number of clients
    server_agent_config = OmegaConf.load(args.server_config)
    server_agent_config.server_configs.num_clients = args.num_clients

    # Detect if we're using SCAFFOLD based on aggregator type
    is_scaffold = "SCAFFOLD" in server_agent_config.server_configs.aggregator
    algorithm_name = "SCAFFOLD" if is_scaffold else "FedAvg"
    print(f"Detected aggregator: {algorithm_name}")

    # Create server agent
    server_agent = ServerAgent(server_agent_config=server_agent_config)

    # Load base client configurations and set corresponding fields for different clients
    client_agent_configs = [
        OmegaConf.load(args.client_config) for _ in range(args.num_clients)
    ]

    # Apply dataset-specific configurations
    setup_dataset_specific_configs(client_agent_configs, args, dataset_type)
    
    # Configure wandb logging
    setup_wandb_configs(client_agent_configs, exp_name)

    # Load client agents
    print("Loading client agents and datasets...")
    client_agents = [
        ClientAgent(client_agent_config=client_agent_configs[i])
        for i in range(args.num_clients)
    ]

    # Set up gradient weighting if requested and supported
    if args.enable_gradient_weighting:
        setup_gradient_weighting(client_agents, dataset_type)

    # Save configuration files to wandb if enabled
    try:
        wandb.save(args.server_config)
        wandb.save(args.client_config)
    except:
        pass  # Wandb might not be initialized

    # Get additional client configurations from the server
    client_config_from_server = server_agent.get_client_configs()
    for client_agent in client_agents:
        client_agent.load_config(client_config_from_server)

    # Load initial global model from the server
    init_global_model = server_agent.get_parameters(serial_run=True)
    for client_agent in client_agents:
        client_agent.load_parameters(init_global_model)

    # [Optional] Set number of local data to the server
    print("Client dataset sizes:")
    for i in range(args.num_clients):
        sample_size = client_agents[i].get_sample_size()
        print(f"  {client_agents[i].get_id()}: {sample_size} samples")
        server_agent.set_sample_size(
            client_id=client_agents[i].get_id(), sample_size=sample_size
        )

    # Run the main training loop
    run_training_loop(server_agent, client_agents, is_scaffold, algorithm_name)

if __name__ == "__main__":
    main() 