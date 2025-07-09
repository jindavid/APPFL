import copy
import time
import torch
import wandb
import importlib
import numpy as np
from torch.nn import Module
from omegaconf import DictConfig
from typing import Tuple, Dict, Optional, Any
from torch.utils.data import Dataset, DataLoader
from appfl.privacy import laplace_mechanism_output_perturb
from appfl.algorithm.trainer.base_trainer import BaseTrainer
from appfl.misc.utils import parse_device_str, apply_model_device


class SCAFFOLDTrainer(BaseTrainer):
    """
    SCAFFOLD Trainer: https://arxiv.org/abs/1910.06378
    
    Client-side trainer for the SCAFFOLD algorithm, which corrects for client drift
    using control variates. The trainer maintains client control variates and uses
    corrected gradients during local training.
    
    The SCAFFOLD algorithm uses the update rule:
    y_i ← y_i - η_l(g_i(y_i) - c_i + c)
    
    Where:
    - y_i: local model parameters
    - η_l: local learning rate
    - g_i(y_i): local gradient
    - c_i: client control variate
    - c: server control variate
    """

    def __init__(
        self,
        model: Optional[Module] = None,
        loss_fn: Optional[Module] = None,
        metric: Optional[Any] = None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        train_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            metric=metric,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_configs=train_configs,
            logger=logger,
            **kwargs,
        )
        if not hasattr(self.train_configs, "device"):
            self.train_configs.device = "cpu"
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.train_configs.get("train_batch_size", 32),
            shuffle=self.train_configs.get("train_data_shuffle", True),
            num_workers=self.train_configs.get("num_workers", 0),
        )
        self.val_dataloader = (
            DataLoader(
                self.val_dataset,
                batch_size=self.train_configs.get("val_batch_size", 32),
                shuffle=self.train_configs.get("val_data_shuffle", False),
                num_workers=self.train_configs.get("num_workers", 0),
            )
            if self.val_dataset is not None
            else None
        )
        
        if (
            hasattr(self.train_configs, "enable_wandb")
            and self.train_configs.enable_wandb
        ):
            self.enabled_wandb = True
            self.wandb_logging_id = self.train_configs.wandb_logging_id
        else:
            self.enabled_wandb = False
        
        self._sanity_check()

        # Extract train device, and configurations for possible DataParallel
        self.device_config, self.device = parse_device_str(self.train_configs.device)
        
        # SCAFFOLD-specific attributes
        self.server_control_variates = None  # c in the paper (received from server)
        self.initial_model_state = None  # y_i at the beginning of the round
        
        # Initialize client control variates (c_i) as zeros
        self.client_control_variates = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        print(f"SCAFFOLD INFO: Client control variates initialized in __init__ with {len(self.client_control_variates)} parameters")

    def load_parameters(self, params: Dict, **kwargs):
        """
        Override BaseTrainer.load_parameters to handle SCAFFOLD control variates.
        Expects structured format with 'model_state' and 'server_control_variates' sections.
        """
        
        # Structured format
        clean_model_params = params["model_state"]
        server_control_variates = params.get("server_control_variates", {})
        
        # Set model parameters
        self.model.load_state_dict(clean_model_params, strict=False)
        
        # Store server control variates (keep on same device as model for efficiency)
        self.server_control_variates = server_control_variates
        
        print(f"SCAFFOLD INFO: Client received server control variates with {len(server_control_variates)} parameters")

    def train(self, **kwargs):
        """
        Train the model using the SCAFFOLD algorithm for a certain number of local epochs or steps.
        """
        if "round" in kwargs:
            self.round = kwargs["round"]
        self.val_results = {"round": self.round + 1}

        # Store the initial model state for SCAFFOLD control variate update
        self.initial_model_state = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Configure model for possible DataParallel
        self.model = apply_model_device(self.model, self.device_config, self.device)

        do_validation = (
            self.train_configs.get("do_validation", False)
            and self.val_dataloader is not None
        )
        do_pre_validation = (
            self.train_configs.get("do_pre_validation", False)
            and self.val_dataloader is not None
        )

        # Set up logging title
        title = (
            ["Round", "Time", "Train Loss", "Train Accuracy"]
            if (not do_validation) and (not do_pre_validation)
            else (
                [
                    "Round",
                    "Pre Val?",
                    "Time",
                    "Train Loss",
                    "Train Accuracy",
                    "Val Loss",
                    "Val Accuracy",
                ]
                if do_pre_validation
                else [
                    "Round",
                    "Time",
                    "Train Loss",
                    "Train Accuracy",
                    "Val Loss",
                    "Val Accuracy",
                ]
            )
        )
        if self.train_configs.mode == "epoch":
            title.insert(1, "Epoch")

        if self.round == 0:
            self.logger.log_title(title)
        self.logger.set_title(title)

        if do_pre_validation:
            val_loss, val_accuracy = self._validate()
            self.val_results["pre_val_loss"] = val_loss
            self.val_results["pre_val_accuracy"] = val_accuracy
            content = [self.round, "Y", " ", " ", " ", val_loss, val_accuracy]
            if self.train_configs.mode == "epoch":
                content.insert(1, 0)
            self.logger.log_content(content)
            if self.enabled_wandb:
                wandb.log(
                    {
                        f"{self.wandb_logging_id}/val-loss (before train)": val_loss,
                        f"{self.wandb_logging_id}/val-accuracy (before train)": val_accuracy,
                    }
                )

        # Start SCAFFOLD training
        optim_module = importlib.import_module("torch.optim")
        assert hasattr(optim_module, self.train_configs.optim), (
            f"Optimizer {self.train_configs.optim} not found in torch.optim"
        )
        optimizer = getattr(optim_module, self.train_configs.optim)(
            self.model.parameters(), **self.train_configs.optim_args
        )
        
        if self.train_configs.mode == "epoch":
            for epoch in range(self.train_configs.num_local_epochs):
                start_time = time.time()
                train_loss, target_true, target_pred = 0, [], []
                for data, target in self.train_dataloader:
                    loss, pred, label = self._train_batch_scaffold(optimizer, data, target)
                    train_loss += loss
                    target_true.append(label)
                    target_pred.append(pred)
                train_loss /= len(self.train_dataloader)
                target_true, target_pred = (
                    np.concatenate(target_true),
                    np.concatenate(target_pred),
                )
                train_accuracy = float(self.metric(target_true, target_pred))
                if do_validation:
                    val_loss, val_accuracy = self._validate()
                    if "val_loss" not in self.val_results:
                        self.val_results["val_loss"] = []
                        self.val_results["val_accuracy"] = []
                    self.val_results["val_loss"].append(val_loss)
                    self.val_results["val_accuracy"].append(val_accuracy)
                per_epoch_time = time.time() - start_time
                if self.enabled_wandb:
                    if do_validation:
                        wandb.log(
                            {
                                f"{self.wandb_logging_id}/train-loss (during train)": train_loss,
                                f"{self.wandb_logging_id}/train-accuracy (during train)": train_accuracy,
                                f"{self.wandb_logging_id}/val-loss (during train)": val_loss,
                                f"{self.wandb_logging_id}/val-accuracy (during train)": val_accuracy,
                            }
                        )
                    else:
                        wandb.log(
                            {
                                f"{self.wandb_logging_id}/train-loss (during train)": train_loss,
                                f"{self.wandb_logging_id}/train-accuracy (during train)": train_accuracy,
                            }
                        )
                self.logger.log_content(
                    [self.round, epoch, per_epoch_time, train_loss, train_accuracy]
                    if (not do_validation) and (not do_pre_validation)
                    else (
                        [
                            self.round,
                            epoch,
                            per_epoch_time,
                            train_loss,
                            train_accuracy,
                            val_loss,
                            val_accuracy,
                        ]
                    )
                )
        else:
            start_time = time.time()
            data_iter = iter(self.train_dataloader)
            train_loss, target_true, target_pred = 0, [], []
            for _ in range(self.train_configs.num_local_steps):
                try:
                    data, target = next(data_iter)
                except:  # noqa E722
                    data_iter = iter(self.train_dataloader)
                    data, target = next(data_iter)
                loss, pred, label = self._train_batch_scaffold(optimizer, data, target)
                train_loss += loss
                target_true.append(label)
                target_pred.append(pred)
            train_loss /= len(self.train_dataloader)
            target_true, target_pred = (
                np.concatenate(target_true),
                np.concatenate(target_pred),
            )
            train_accuracy = float(self.metric(target_true, target_pred))
            if do_validation:
                val_loss, val_accuracy = self._validate()
                self.val_results["val_loss"] = val_loss
                self.val_results["val_accuracy"] = val_accuracy
            per_step_time = time.time() - start_time
            if self.enabled_wandb:
                wandb.log(
                    {
                        f"{self.wandb_logging_id}/train-loss (during train)": train_loss,
                        f"{self.wandb_logging_id}/train-accuracy (during train)": train_accuracy,
                        f"{self.wandb_logging_id}/val-loss (during train)": val_loss,
                        f"{self.wandb_logging_id}/val-accuracy (during train)": val_accuracy,
                    }
                )
            self.logger.log_content(
                [self.round, per_step_time, train_loss, train_accuracy]
                if (not do_validation) and (not do_pre_validation)
                else (
                    [
                        self.round,
                        per_step_time,
                        train_loss,
                        train_accuracy,
                        val_loss,
                        val_accuracy,
                    ]
                    if not do_pre_validation
                    else [
                        self.round,
                        "N",
                        per_step_time,
                        train_loss,
                        train_accuracy,
                        val_loss,
                        val_accuracy,
                    ]
                )
            )

        # If model was wrapped in DataParallel, unload it
        if self.device_config["device_type"] == "gpu-multi":
            self.model = self.model.module.to(self.device)

        # Update client control variates using SCAFFOLD formula
        self._update_client_control_variates()

        self.round += 1

        # Differential privacy
        if self.train_configs.get("use_dp", False):
            assert hasattr(self.train_configs, "clip_value"), (
                "Gradient clipping value must be specified"
            )
            assert hasattr(self.train_configs, "epsilon"), (
                "Privacy budget (epsilon) must be specified"
            )
            sensitivity = (
                2.0 * self.train_configs.clip_value * self.train_configs.optim_args.lr
            )
            self.model_state = laplace_mechanism_output_perturb(
                self.model,
                sensitivity,
                self.train_configs.epsilon,
            )
        else:
            self.model_state = copy.deepcopy(self.model.state_dict())

        # Move to CPU for communication
        if "cuda" in self.train_configs.device:
            for k in self.model_state:
                self.model_state[k] = self.model_state[k].cpu()

    def get_parameters(self) -> Dict:
        """
        Return model parameters along with client control variates.
        Returns a structured dictionary with 'model_state' and 'client_control_variates' sections.
        """
        if not hasattr(self, "model_state"):
            self.model_state = copy.deepcopy(self.model.state_dict())
        
        # Prepare client control variates on CPU
        client_control_variates_cpu = {
            name: param.cpu() if param.is_cuda else param
            for name, param in self.client_control_variates.items()
        }
        
        # Prepare structured result
        result = {
            "model_state": copy.deepcopy(self.model_state),
            "client_control_variates": client_control_variates_cpu
        }
        
        return (
            (result, self.val_results)
            if hasattr(self, "val_results")
            else result
        )

    def _sanity_check(self):
        """
        Check if the configurations are valid.
        """
        assert hasattr(self.train_configs, "mode"), "Training mode must be specified"
        assert self.train_configs.mode in [
            "epoch",
            "step",
        ], "Training mode must be either 'epoch' or 'step'"
        if self.train_configs.mode == "epoch":
            assert hasattr(self.train_configs, "num_local_epochs"), (
                "Number of local epochs must be specified"
            )
        else:
            assert hasattr(self.train_configs, "num_local_steps"), (
                "Number of local steps must be specified"
            )

    def _validate(self) -> Tuple[float, float]:
        """
        Validate the model
        :return: loss, accuracy
        """
        device = self.device
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            target_pred, target_true = [], []
            for data, target in self.val_dataloader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                val_loss += self.loss_fn(output, target).item()
                target_true.append(target.detach().cpu().numpy())
                target_pred.append(output.detach().cpu().numpy())
        val_loss /= len(self.val_dataloader)
        val_accuracy = float(
            self.metric(np.concatenate(target_true), np.concatenate(target_pred))
        )
        self.model.train()
        return val_loss, val_accuracy

    def _train_batch_scaffold(
        self, optimizer: torch.optim.Optimizer, data, target
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Train the model for one batch of data using SCAFFOLD correction.
        
        SCAFFOLD uses corrected gradients: g_i(y_i) - c_i + c
        
        :param optimizer: torch optimizer
        :param data: input data
        :param target: target label
        :return: loss, prediction, label
        """
        device = self.device
        data = data.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = self.model(data)
        loss = self.loss_fn(output, target)
        loss.backward()
        
        # Apply SCAFFOLD correction to gradients
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                    # Apply SCAFFOLD correction: g_i(y_i) - c_i + c
                client_cv = self.client_control_variates[name].to(device)
                server_cv = self.server_control_variates[name].to(device)
                param.grad.data = param.grad.data - client_cv + server_cv
        
        if getattr(self.train_configs, "clip_grad", False) or getattr(
            self.train_configs, "use_dp", False
        ):
            assert hasattr(self.train_configs, "clip_value"), (
                "Gradient clipping value must be specified"
            )
            assert hasattr(self.train_configs, "clip_norm"), (
                "Gradient clipping norm must be specified"
            )
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.train_configs.clip_value,
                norm_type=self.train_configs.clip_norm,
            )
        
        optimizer.step()
        return loss.item(), output.detach().cpu().numpy(), target.detach().cpu().numpy()

    def _update_client_control_variates(self):
        """
        Update client control variates using SCAFFOLD formula.
        
        SCAFFOLD client control variate update:
        c_i^+ = c_i - c + (1/(K*η_l)) * (x - y_i)
        
        Where:
        - c_i^+: new client control variate
        - c_i: old client control variate  
        - c: server control variate
        - K: number of local steps/epochs
        - η_l: local learning rate
        - x: initial model parameters (at start of round)
        - y_i: final model parameters (after local training)
        """

        
        # Get local learning rate and number of local steps
        local_lr = self.train_configs.optim_args.lr
        if self.train_configs.mode == "epoch":
            # Approximate number of steps per epoch
            steps_per_epoch = len(self.train_dataloader)
            num_local_steps = self.train_configs.num_local_epochs * steps_per_epoch
        else:
            num_local_steps = self.train_configs.num_local_steps
        
        # Update client control variates
        device = self.device
        for name, param in self.model.named_parameters():
            if name in self.client_control_variates:
                    # c_i^+ = c_i - c + (1/(K*η_l)) * (x - y_i)
                    # Ensure all tensors are on the same device as the model for efficient computation
                    old_client_cv = self.client_control_variates[name].to(device)
                    server_cv = self.server_control_variates[name].to(device)
                    initial_param = self.initial_model_state[name].to(device)
                    final_param = param.data  # Already on model device
                    
                    correction_term = (initial_param - final_param) / (num_local_steps * local_lr)
                    
                    # Store result on CPU for consistency and communication
                    self.client_control_variates[name] = (
                        old_client_cv - server_cv + correction_term
                    ).cpu() 