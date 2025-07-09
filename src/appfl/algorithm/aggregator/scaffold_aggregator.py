import copy
import torch
from omegaconf import DictConfig
from appfl.algorithm.aggregator import BaseAggregator
from typing import Union, Dict, OrderedDict, Any, Optional


class SCAFFOLDAggregator(BaseAggregator):
    """
    SCAFFOLD Aggregator: https://arxiv.org/abs/1910.06378
    
    Server-side aggregator for the SCAFFOLD algorithm.
    Maintains both the global model parameters and the server control variates.
    
    The SCAFFOLD algorithm corrects for client drift using control variates.
    This aggregator expects to work with a SCAFFOLDTrainer that implements
    the client-side SCAFFOLD logic.
    
    :param `model`: An optional instance of the model to be trained in the federated learning setup.
        This can be useful for aggregating parameters that does requires gradient, such as the batch
        normalization layers. If not provided, the aggregator will only aggregate the parameters
        sent by the clients.
    :param `aggregator_configs`: Configuration for the aggregator. It should be specified in the YAML
        configuration file under `aggregator_kwargs`.
    :param `logger`: An optional instance of the logger to be used for logging.
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        aggregator_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None,
    ):
        self.model = model
        self.logger = logger
        self.aggregator_configs = aggregator_configs
        self.client_weights_mode = aggregator_configs.get(
            "client_weights_mode", "equal"
        )

        if self.model is not None:
            self.named_parameters = set()
            for name, _ in self.model.named_parameters():
                self.named_parameters.add(name)
        else:
            self.named_parameters = None

        self.global_state = None  # Global model parameters (x in the paper)
        self.client_control_variates = {}  # Client control variates (c_i in the paper)
        self.step = {}
        
        # Initialize server control variates if model is provided
        if self.model is not None:
            self.server_control_variates = {
                name: torch.zeros_like(param)
                for name, param in self.model.named_parameters()
                if param.requires_grad
            }
            
            # Count total scalar parameters for better debugging
            total_scalar_params = sum(
                param.numel() for name, param in self.model.named_parameters()
                if param.requires_grad
            )
            print(f"SCAFFOLD INFO: Server control variates initialized - {len(self.server_control_variates)} parameter tensors covering {total_scalar_params:,} total scalar parameters")
        else:
            self.server_control_variates = {}


    def get_parameters(self, **kwargs) -> Dict:
        """
        Returns the global model parameters along with server control variates.
        Returns a structured dictionary with 'model_state' and 'server_control_variates' sections.
        """
        if self.global_state is None:
            if self.model is not None:
                model_state = copy.deepcopy(self.model.state_dict())
                total_scalar_params = sum(param.numel() for param in self.server_control_variates.values())
                print(f"SCAFFOLD INFO: Sending {len(self.server_control_variates)} server control variate tensors covering {total_scalar_params:,} total scalar parameters to clients")
                return {
                    "model_state": model_state,
                    "server_control_variates": {k: v.clone() for k, v in self.server_control_variates.items()}
                }
            else:
                raise ValueError("Model is not provided to the aggregator.")
        
        # Return structured dictionary with model parameters and server control variates
        return {
            "model_state": {k: v.clone() for k, v in self.global_state.items()},
            "server_control_variates": {k: v.clone() for k, v in self.server_control_variates.items()}
        }

    def aggregate(
        self, local_models: Dict[Union[str, int], Union[Dict, OrderedDict]], **kwargs
    ) -> Dict:
        """
        SCAFFOLD aggregation algorithm.
        
        Expects local_models to contain both model parameters and client control variates
        (sent by SCAFFOLDTrainer with special prefixes).
        
        Args:
            local_models: Dictionary mapping client_id -> model_state_dict_with_control_variates
            **kwargs: Additional arguments
            
        Returns:
            Dict: Updated global model parameters
        """
        if self.global_state is None:
            # Extract first model to initialize global state
            first_model_data = list(local_models.values())[0]
            first_model_state = first_model_data["model_state"]
            
            if self.model is not None:
                try:
                    # Initialize from model, but only include parameters that exist in client data
                    self.global_state = {
                        name: self.model.state_dict()[name]
                        for name in first_model_state
                    }
                except:  # noqa E722
                    self.global_state = {
                        name: tensor.detach().clone()
                        for name, tensor in first_model_state.items()
                    }
            else:
                self.global_state = {
                    name: tensor.detach().clone()
                    for name, tensor in first_model_state.items()
                }

        # Separate model parameters from SCAFFOLD control variates
        clean_local_models = {}
        client_control_updates = {}
        
        for client_id, model_data in local_models.items():
            # Structured format
            clean_local_models[client_id] = model_data["model_state"]
            if "client_control_variates" in model_data:
                client_control_updates[client_id] = model_data["client_control_variates"]

        # 1. Aggregate model parameters (same as FedAvg)
        self.compute_steps(clean_local_models)

        for name in self.global_state:
            if name in self.step:
                self.global_state[name] = self.global_state[name] + self.step[name]
            else:
                param_sum = torch.zeros_like(self.global_state[name])
                for _, model in clean_local_models.items():
                    param_sum += model[name]
                self.global_state[name] = torch.div(param_sum, len(clean_local_models)).type(
                    param_sum.dtype
                )

        # 2. Update server control variates (SCAFFOLD-specific)
        if client_control_updates:
            print(f"SCAFFOLD INFO: Received client control updates from {len(client_control_updates)} clients")
            self._update_server_control_variates(client_control_updates)
        else:
            print("SCAFFOLD INFO: No client control updates received")

        if self.model is not None:
            self.model.load_state_dict(self.global_state, strict=False)
        
        return {k: v.clone() for k, v in self.global_state.items()}

    def compute_steps(
        self, local_models: Dict[Union[str, int], Union[Dict, OrderedDict]]
    ):
        """
        Compute the changes to the global model after the aggregation.
        This follows the same pattern as FedAvgAggregator.
        """
        for name in self.global_state:
            # Skip integer parameters by averaging them later in the `aggregate` method
            if (
                self.named_parameters is not None and name not in self.named_parameters
            ) or (
                self.global_state[name].dtype == torch.int64
                or self.global_state[name].dtype == torch.int32
            ):
                continue
            self.step[name] = torch.zeros_like(self.global_state[name])

        for client_id, model in local_models.items():
            if (
                self.client_weights_mode == "sample_size"
                and hasattr(self, "client_sample_size")
                and client_id in self.client_sample_size
            ):
                weight = self.client_sample_size[client_id] / sum(
                    self.client_sample_size.values()
                )
            else:
                weight = 1.0 / len(local_models)

            for name in model:
                if name in self.step:
                    self.step[name] += weight * (model[name] - self.global_state[name])

    def _update_server_control_variates(self, client_control_updates: Dict[Union[str, int], Dict[str, torch.Tensor]]):
        """
        Update server control variates based on client control variate updates.
        
        SCAFFOLD algorithm: c ← c + (1/N) * Σ(c_i^+ - c_i)
        
        Args:
            client_control_updates: Dictionary mapping client_id -> new_client_control_variates
        """
        if not client_control_updates:
            return
            
        # Calculate control variate updates: Σ(c_i^+ - c_i)
        control_variate_delta = {
            name: torch.zeros_like(param)
            for name, param in self.server_control_variates.items()
        }
        
        num_participating_clients = len(client_control_updates)
        
        for client_id, new_client_cv in client_control_updates.items():
            # Get previous client control variate (or zero if first time)
            old_client_cv = self.client_control_variates.get(
                client_id,
                {name: torch.zeros_like(param) for name, param in new_client_cv.items()}
            )
            
            # Accumulate the difference: c_i^+ - c_i
            for name in control_variate_delta:
                if name in new_client_cv and name in old_client_cv:
                    control_variate_delta[name] += (new_client_cv[name] - old_client_cv[name])
            
            # Store the new client control variate
            self.client_control_variates[client_id] = {
                name: param.clone() for name, param in new_client_cv.items()
            }

        # Update server control variates: c ← c + (1/N) * Σ(c_i^+ - c_i)
        for name in self.server_control_variates:
            if name in control_variate_delta:
                self.server_control_variates[name] += control_variate_delta[name] / num_participating_clients

    def get_server_control_variates(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get a copy of the current server control variates.
        
        Returns:
            Dictionary of server control variates or None if not initialized
        """
        if not self.server_control_variates:
            return None
        return {k: v.clone() for k, v in self.server_control_variates.items()}
