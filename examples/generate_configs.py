import yaml
import argparse
import os
from copy import deepcopy
import itertools

# --- Helper functions ---
def get_nested_key(data, keys):
    """Access a nested key in a dictionary."""
    for key in keys:
        data = data[key]
    return data

def set_nested_key(data, keys, value):
    """Set a value for a nested key in a dictionary."""
    for key in keys[:-1]:
        data = data.setdefault(key, {})
    data[keys[-1]] = value

# --- Main script ---
def generate_configs(base_config_path, output_dir):
    """
    Generates server configuration files by varying parameters in a base YAML file.
    """
    # --- Configuration for parameter sweep ---
    # Define the parameters you want to vary here.
    # The key is a tuple of nested keys to access the parameter within the server config.
    # The value is a list of values to try for that parameter.
    param_grid = {
        ('client_configs', 'train_configs', 'power_lambda'): [0.0, 0.25, 0.5, 0.75, 1.0],
        ('client_configs', 'train_configs', 'optim_args', 'lr'): [0.01, 0.005, 0.001],
        ('client_configs', 'train_configs', 'train_batch_size'): [128, 256, 1024],
        ('client_configs', 'train_configs', 'clip_grad'): [True, False]
    }

    # Load the base configuration file
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all combinations of parameter values
    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())
    value_combinations = list(itertools.product(*param_values))

    print(f"Generating {len(value_combinations)} configuration files...")

    for i, combo in enumerate(value_combinations):
        new_config = deepcopy(base_config)
        filename_parts = []

        for keys, value in zip(param_keys, combo):
            set_nested_key(new_config, list(keys), value)
            # Use the last part of the key for the filename
            key_name = keys[-1]
            filename_parts.append(f"{key_name}_{value}")

        # Generate a descriptive filename
        output_filename = f"server_config_{'_'.join(filename_parts)}.yaml"
        output_path = os.path.join(output_dir, output_filename)

        # Write the new configuration to a file
        with open(output_path, 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"Generated {output_path}")

    print("\nDone.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate experiment configuration files from a base YAML.")
    parser.add_argument(
        "--base_config",
        type=str,
        required=True,
        help="Path to the base YAML server configuration file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the generated YAML files."
    )

    args = parser.parse_args()

    # Inform user about dependencies
    try:
        import yaml
    except ImportError:
        print("PyYAML is not installed. Please install it using: pip install PyYAML")
        exit(1)

    generate_configs(args.base_config, args.output_dir) 