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

        # Gradient weighting attributes from WeightedGradientTrainer
        self.label_weights = {}
        self.use_gradient_weighting = self.train_configs.get("use_gradient_weighting", False)
        self.use_uniform_weights = self.train_configs.get("use_uniform_weights", False)
        self.use_interpolated_weights = self.train_configs.get("use_interpolated_weights", False)
        self.lambda_interp = self.train_configs.get("lambda_interp", 0.5)
        self.use_power = self.train_configs.get("use_power", False)
        self.power_lambda = self.train_configs.get("power_lambda", 0.5)
        self.global_label_distribution = {}
        self.local_label_distribution = {}

        if self.use_gradient_weighting and self.logger:
            self.logger.info("SCAFFOLD with gradient weighting enabled.")
            self.logger.info(f"Using uniform weights: {self.use_uniform_weights}")
            self.logger.info(f"Using interpolated weights: {self.use_interpolated_weights}")
            self.logger.info(f"Using power transformation: {self.use_power}")
            if self.use_power:
                self.logger.info(f"Power transformation parameter: {self.power_lambda}")
            if self.use_interpolated_weights:
                self.logger.info(f"Lambda interpolation parameter: {self.lambda_interp}")
        
        self._sanity_check()

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
        
        # Debug: Check if server control variates are non-zero
        if server_control_variates:
            total_server_cv_norm = sum(torch.norm(cv).item() for cv in server_control_variates.values())
            print(f"SCAFFOLD DEBUG: Server CV total norm = {total_server_cv_norm:.8f}")

    def train(self, **kwargs):
        """
        Train the model using the SCAFFOLD algorithm for a certain number of local epochs or steps.
        """
        print("SCAFFOLD DEBUG: Starting SCAFFOLD training!")
        print(f"SCAFFOLD DEBUG: Server control variates available: {self.server_control_variates is not None}")
        if self.server_control_variates:
            total_server_norm = sum(torch.norm(cv).item() for cv in self.server_control_variates.values())
            print(f"SCAFFOLD DEBUG: Server control variates total norm: {total_server_norm:.8f}")
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
        
        # Store the batch count at the start of the round
        self._initial_batch_count = getattr(self, '_batch_count', 0)

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
            train_loss /= self.train_configs.num_local_steps
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

        # Print the actual number of batches trained
        actual_batches_trained = self._batch_count - self._initial_batch_count
        print(f"SCAFFOLD DEBUG: Actual batches trained this round: {actual_batches_trained}")

        # Get local learning rate from optimizer configs
        local_lr = self.train_configs.optim_args.get("lr")
        if local_lr is None:
            raise ValueError("Learning rate 'lr' must be specified in 'optim_args' for SCAFFOLD.")

        # Update client control variates using SCAFFOLD formula
        self._update_client_control_variates(
            num_local_steps=actual_batches_trained, 
            local_lr=local_lr
        )

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

    def set_global_label_distribution(self, global_distribution: Dict[int, float]):
        """
        Set the global label distribution and compute weights. This method is from WeightedGradientTrainer.
        
        Args:
            global_distribution: Dictionary mapping label to global probability
        """
        self.global_label_distribution = global_distribution
        self._compute_label_weights()

    def _compute_local_label_distribution(self):
        """
        Compute the local label distribution for this client's training dataset. From WeightedGradientTrainer.
        """
        if self.train_dataset is None:
            if self.logger:
                self.logger.warning("No training dataset available for label distribution computation")
            return
            
        label_counts = {}
        total_samples = 0
        
        temp_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_configs.get("train_batch_size", 32),
            shuffle=False,
            num_workers=0
        )
        
        for batch in temp_loader:
            _, targets = batch
            if torch.is_tensor(targets):
                targets = targets.numpy()
            
            unique_labels, counts = np.unique(targets, return_counts=True)
            for label, count in zip(unique_labels, counts):
                label = int(label)
                label_counts[label] = label_counts.get(label, 0) + count
                total_samples += count
        
        self.local_label_distribution = {}
        for label, count in label_counts.items():
            self.local_label_distribution[label] = count / total_samples
            
        if self.logger:
            self.logger.info(f"Local label distribution: {self.local_label_distribution}")

    def _compute_label_weights(self):
        """
        Compute label weights based on global vs local distribution. From WeightedGradientTrainer.
        """
        if not self.global_label_distribution:
            if self.logger:
                self.logger.warning("Global label distribution not set. Cannot compute weights.")
            return
            
        if not self.local_label_distribution:
            self._compute_local_label_distribution()
        
        if self.use_uniform_weights:
            self.label_weights = {label: 1.0 for label in self.global_label_distribution}
            if self.logger:
                self.logger.info(f"Using uniform weights (all weights = 1.0): {self.label_weights}")
            return
        
        self.label_weights = {}
        for label in self.global_label_distribution:
            global_prob = self.global_label_distribution[label]
            local_prob = self.local_label_distribution.get(label, 0.0)
            
            if local_prob > 0:
                if self.use_interpolated_weights:
                    denominator = (1 - self.lambda_interp) * local_prob + self.lambda_interp * global_prob
                    weight = global_prob / denominator if denominator > 0 else 0.0
                else:
                    weight = global_prob / local_prob
            else:
                weight = 0.0
            
            if weight > 0 and self.use_power:
                weight = np.power(weight, self.power_lambda)
                
            self.label_weights[label] = weight
        
        if self.logger:
            method_info = f" using interpolated IW (λ={self.lambda_interp})" if self.use_interpolated_weights else " using standard IW"
            transformation_info = f" with power transformation (λ={self.power_lambda})" if self.use_power else ""
            self.logger.info(f"Computed label weights{method_info}{transformation_info}: {self.label_weights}")

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
        if not hasattr(self, '_batch_count'):
            self._batch_count = 0
        self._batch_count += 1
        
        device = self.device
        data = data.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = self.model(data)

        if self.use_gradient_weighting and self.label_weights:
            batch_size = data.size(0)
            if batch_size == 0:
                return 0.0, np.array([]), np.array([])
            
            total_weighted_loss = 0.0
            for i in range(batch_size):
                sample_label = target[i].item()
                weight = self.label_weights.get(sample_label, 1.0)
                
                sample_output = output[i:i+1]
                sample_target = target[i:i+1]
                sample_loss = self.loss_fn(sample_output, sample_target)
                
                weighted_loss = weight * sample_loss
                total_weighted_loss += weighted_loss
            
            loss = total_weighted_loss / batch_size
        else:
            loss = self.loss_fn(output, target)

        loss.backward()
        
        # Apply SCAFFOLD correction to gradients
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Apply SCAFFOLD correction: g_i(y_i) - c_i + c
                client_cv = self.client_control_variates[name].to(device)
                server_cv = self.server_control_variates[name].to(device)
                param.grad.data.add_(-client_cv).add_(server_cv) # In-place is more efficient
        
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

    def _update_client_control_variates(self, num_local_steps: int, local_lr: float):
        """
        Update client control variates based on the change in model parameters.
        c_i^+ = c_i - c + (1/(K*η_l)) * (x - y_i)
        """
        print("SCAFFOLD DEBUG: Updating client control variates...")
        if num_local_steps == 0 or local_lr == 0:
            print("SCAFFOLD WARNING: num_local_steps or local_lr is zero, skipping control variate update.")
            return

        device = self.device
        divisor = num_local_steps * local_lr
        current_model_state = self.model.state_dict()

        for name in self.client_control_variates:
            # Move all tensors to the same device for computation
            initial_param = self.initial_model_state[name].to(device)
            final_param = current_model_state[name].to(device) # Ensure this is also on the correct device
            old_client_cv = self.client_control_variates[name].to(device)
            server_cv = self.server_control_variates[name].to(device)

            # c_i^+ = c_i - c + (x - y_i) / (K * η_l)
            param_change = initial_param - final_param
            correction_term = param_change / divisor
            
            new_client_cv = old_client_cv - server_cv + correction_term
            # Store the updated control variate back on the CPU
            self.client_control_variates[name] = new_client_cv.cpu()

        # Debugging norms (with device correction)
        param_change_norm = torch.norm(
            torch.cat([
                (self.initial_model_state[n].to(device) - current_model_state[n].to(device)).view(-1)
                for n in self.initial_model_state
            ])
        ).item()
        
        # client_control_variates are on CPU now, so no need to move for norm calculation
        client_cv_norm = torch.norm(
            torch.cat([cv.view(-1) for cv in self.client_control_variates.values()])
        ).item()
        
        print(f"SCAFFOLD DEBUG: Update divisor (K * η_l) = {divisor:.6f}")
        print(f"SCAFFOLD DEBUG: Total model parameter change norm (x - y): {param_change_norm:.8f}")
        print(f"SCAFFOLD DEBUG: Updated client CV norm (c_i^+): {client_cv_norm:.8f}")