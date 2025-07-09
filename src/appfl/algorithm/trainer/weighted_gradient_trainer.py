import copy
import time
import torch
import wandb
import importlib
import numpy as np
from torch.nn import Module
from omegaconf import DictConfig
from typing import Tuple, Dict, Optional, Any, List
from torch.utils.data import Dataset, DataLoader
from appfl.algorithm.trainer.vanilla_trainer import VanillaTrainer


class WeightedGradientTrainer(VanillaTrainer):
    """
    Trainer that applies label-based gradient weighting during training.
    
    This trainer computes label weights based on global vs local label distribution
    ratios: w_c(y) = p(y) / p_c(y), where y is the sample's label.
    
    The weights are computed during initialization by analyzing the training dataset.
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
        
        # Label weights for this client
        self.label_weights = {}
        
        # Whether gradient weighting is enabled
        self.use_gradient_weighting = train_configs.get("use_gradient_weighting", True)
        
        # Importance weighting options
        self.use_uniform_weights = train_configs.get("use_uniform_weights", False)
        self.use_interpolated_weights = train_configs.get("use_interpolated_weights", False)
        self.lambda_interp = train_configs.get("lambda_interp", 0.5)  # Interpolation parameter
        
        # Unified power transformation: (weight)^power_lambda
        self.use_power = train_configs.get("use_power", False)
        self.power_lambda = train_configs.get("power_lambda", 0.5)  # Power parameter (0.5=sqrt, 1.0=no transform)
        
        # Global label distribution (will be set externally)
        self.global_label_distribution = {}
        
        # Local label distribution (computed from training data)
        self.local_label_distribution = {}
        
        if self.logger:
            self.logger.info(f"WeightedGradientTrainer initialized with gradient weighting: {self.use_gradient_weighting}")
            self.logger.info(f"Using uniform weights: {self.use_uniform_weights}")
            self.logger.info(f"Using interpolated weights: {self.use_interpolated_weights}")
            self.logger.info(f"Using power transformation: {self.use_power}")
            if self.use_power:
                self.logger.info(f"Power transformation parameter: {self.power_lambda}")
            if self.use_interpolated_weights:
                self.logger.info(f"Lambda interpolation parameter: {self.lambda_interp}")

    def set_global_label_distribution(self, global_distribution: Dict[int, float]):
        """
        Set the global label distribution and compute weights.
        
        Args:
            global_distribution: Dictionary mapping label to global probability
        """
        self.global_label_distribution = global_distribution
        self._compute_label_weights()

    def _compute_local_label_distribution(self):
        """
        Compute the local label distribution for this client's training dataset.
        """
        if self.train_dataset is None:
            self.logger.warning("No training dataset available for label distribution computation")
            return
            
        label_counts = {}
        total_samples = 0
        
        # Create a temporary dataloader to analyze the dataset
        temp_loader = DataLoader(
            self.train_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0
        )
        
        for batch in temp_loader:
            _, targets = batch
            if torch.is_tensor(targets):
                targets = targets.numpy()
            
            # Count labels
            unique_labels, counts = np.unique(targets, return_counts=True)
            for label, count in zip(unique_labels, counts):
                label = int(label)
                label_counts[label] = label_counts.get(label, 0) + count
                total_samples += count
        
        # Convert counts to probabilities
        self.local_label_distribution = {}
        for label, count in label_counts.items():
            self.local_label_distribution[label] = count / total_samples
            
        if self.logger:
            self.logger.info(f"Local label distribution: {self.local_label_distribution}")

    def _compute_label_weights(self):
        """
        Compute label weights based on global vs local distribution.
        Applies importance weighting transformations based on configuration.
        """
        if not self.global_label_distribution:
            self.logger.warning("Global label distribution not set. Cannot compute weights.")
            return
            
        # Compute local distribution if not already done
        if not self.local_label_distribution:
            self._compute_local_label_distribution()
        
        # Check for uniform weights option first
        if self.use_uniform_weights:
            # Set all weights to 1.0 (uniform weighting)
            self.label_weights = {}
            for label in self.global_label_distribution:
                self.label_weights[label] = 1.0
            
            if self.logger:
                self.logger.info("Using uniform weights (all weights = 1.0)")
                self.logger.info(f"Label weights: {self.label_weights}")
            return
        
        # Compute weights based on selected method
        self.label_weights = {}
        for label in self.global_label_distribution:
            global_prob = self.global_label_distribution[label]
            local_prob = self.local_label_distribution.get(label, 0.0)
            
            if local_prob > 0:
                if self.use_interpolated_weights:
                    # Interpolated importance weighting: p_global / {(1-λ)*p_local + λ*p_global}
                    denominator = (1 - self.lambda_interp) * local_prob + self.lambda_interp * global_prob
                    if denominator > 0:
                        weight = global_prob / denominator
                    else:
                        weight = 0.0
                else:
                    # Standard importance weighting: w(x) = p_global(x) / p_local(x)
                    weight = global_prob / local_prob
            else:
                # If client doesn't have this label, set weight to 0
                weight = 0.0
            
            # Apply power transformation: weight^power_lambda
            if weight > 0 and self.use_power:
                weight = np.power(weight, self.power_lambda)
                
            self.label_weights[label] = weight
        
        if self.logger:
            method_info = ""
            if self.use_interpolated_weights:
                method_info = f" using interpolated IW (λ={self.lambda_interp})"
            else:
                method_info = " using standard IW"
            
            transformation_info = ""
            if self.use_power:
                if self.power_lambda == 0.5:
                    transformation_info = " with square root transformation"
                else:
                    transformation_info = f" with power transformation (λ={self.power_lambda})"
            
            self.logger.info(f"Computed label weights{method_info}{transformation_info}: {self.label_weights}")

    def _train_batch(
        self, optimizer: torch.optim.Optimizer, data, target
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Train the model for one batch of data with gradient weighting.
        
        This method applies label-based weights to the gradients during backpropagation.
        For each sample in the batch, the gradient is scaled by the weight corresponding
        to that sample's label.
        """
        device = self.device
        data = data.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = self.model(data)
        
        if self.use_gradient_weighting and self.label_weights:
            # Apply weighted loss for each sample in the batch
            batch_size = data.size(0)
            total_weighted_loss = 0
            
            for i in range(batch_size):
                # Get the label for this sample
                sample_label = target[i].item()
                
                # Get the weight for this label
                weight = self.label_weights.get(sample_label, 1.0)
                
                # Compute loss for this sample
                sample_output = output[i:i+1]  # Keep batch dimension
                sample_target = target[i:i+1]
                sample_loss = self.loss_fn(sample_output, sample_target)
                
                # Apply weight to the loss
                weighted_loss = weight * sample_loss
                total_weighted_loss += weighted_loss
            
            # Average the weighted losses
            loss = total_weighted_loss / batch_size
        else:
            # Standard unweighted loss
            loss = self.loss_fn(output, target)
        
        loss.backward()
        
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

    def train(self, **kwargs):
        """
        Override train method to ensure label weights are applied if available.
        """
        if "label_weights" in kwargs:
            self.set_label_weights(kwargs["label_weights"])
        
        # Call parent train method
        super().train(**kwargs) 


def compute_global_label_distribution(client_data_loaders: Dict[str, DataLoader]) -> Dict[int, float]:
    """
    Utility function to compute global label distribution across all clients.
    
    Args:
        client_data_loaders: Dictionary mapping client_id to their training DataLoader
        
    Returns:
        Dictionary mapping label to global probability
    """
    global_label_counts = {}
    total_samples = 0
    
    for client_id, data_loader in client_data_loaders.items():
        for batch in data_loader:
            _, targets = batch
            if torch.is_tensor(targets):
                targets = targets.numpy()
            
            # Count labels
            unique_labels, counts = np.unique(targets, return_counts=True)
            for label, count in zip(unique_labels, counts):
                label = int(label)
                global_label_counts[label] = global_label_counts.get(label, 0) + count
                total_samples += count
    
    # Convert to probabilities
    global_distribution = {}
    for label, count in global_label_counts.items():
        global_distribution[label] = count / total_samples
    
    return global_distribution 