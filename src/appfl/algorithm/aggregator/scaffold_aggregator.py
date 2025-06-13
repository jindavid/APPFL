import copy
import torch
from omegaconf import DictConfig
from . import BaseAggregator
from typing import Union, Dict, OrderedDict, Any, Optional

class SCAFFOLDAggregator(BaseAggregator):
    """
    SCAFFOLD Aggregator: https://arxiv.org/abs/1910.06378
    :param `model`: An instance of the model to be trained in the federated learning setup.
    :param `aggregator_configs`: Configuration for the aggregator.
    :param `logger`: An optional instance of the logger to be used for logging.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        aggregator_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None,
    ):
        self.model = model
        self.logger = logger
        self.aggregator_configs = aggregator_configs

        self.global_state = {
            k: v.cpu().clone() for k, v in self.model.state_dict().items()
        }
        self.server_control_variates = {
            name: torch.zeros_like(param)
            for name, param in self.global_state.items()
            if param.requires_grad
        }

    def get_parameters(self, **kwargs) -> Dict:
        """
        Get the global model parameters and server control variates to be sent to the clients.
        """
        return {
            "global_state": {k: v.clone() for k, v in self.global_state.items()},
            "server_control_variates": {
                k: v.clone() for k, v in self.server_control_variates.items()
            },
        }

    def aggregate(
        self, local_updates: Dict[Union[str, int], Dict[str, Union[Dict, OrderedDict]]], **kwargs
    ) -> Dict:
        """
        Aggregate model updates and control variate updates from clients.
        The `local_updates` from clients should be a dictionary containing the updated local model
        and the update for the client's control variate.
        For example:
        `local_updates = {
            'client_1': {
                'model': y_1_new,
                'control_variate_update': delta_c_1
            },
            ...
        }`
        """
        # Aggregate model updates
        model_update_sum = {
            name: torch.zeros_like(param) for name, param in self.global_state.items()
        }
        for client_id in local_updates:
            local_model = local_updates[client_id]["model"]
            for name in model_update_sum:
                model_update_sum[name] += local_model[name] - self.global_state[name]

        for name in self.global_state:
            if name in model_update_sum:
                self.global_state[name] += model_update_sum[name] / len(local_updates)

        # Aggregate control variate updates
        control_variate_update_sum = {
            name: torch.zeros_like(param)
            for name, param in self.server_control_variates.items()
        }
        for client_id in local_updates:
            control_variate_update = local_updates[client_id]["control_variate_update"]
            for name in control_variate_update_sum:
                control_variate_update_sum[name] += control_variate_update[name]

        for name in self.server_control_variates:
            self.server_control_variates[name] += control_variate_update_sum[
                name
            ] / len(local_updates)

        self.model.load_state_dict(self.global_state, strict=False)

        return {k: v.clone() for k, v in self.global_state.items()} 