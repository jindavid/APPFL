# Weighted Gradient Federated Learning Algorithm

This implementation provides a novel federated learning algorithm that extends FedAvg with gradient weighting based on label distribution ratios to handle non-IID data distributions. **The algorithm uses the standard FedAvg aggregator with weighted gradient training.**

## Algorithm Overview

### Theoretical Foundation

The weighted gradient federated learning algorithm addresses the challenges of non-IID data distributions in federated learning by weighting gradients based on label frequency imbalances between local client datasets and the global dataset.

**Key Formula:** For each client `c` and label `x`, the weight is computed as:
```
w_c(x) = p(x) / p_c(x)
```

Where:
- `p(x)` is the probability density of label `x` in the global dataset
- `p_c(x)` is the probability density of label `x` in client `c`'s local dataset

### Algorithm Steps

1. **Pre-training Analysis**: Before federated training begins, analyze all client datasets to compute:
   - Global label distribution across all clients
   - Local label distribution for each client
   - Weight ratios for each label per client

2. **Weighted Training**: During each training round:
   - Clients compute their label weights based on global vs local distributions
   - Local training applies weights to loss gradients based on sample labels
   - Weighted gradients are aggregated using **standard FedAvg**

3. **Aggregation**: Server aggregates weighted local models using **standard FedAvg methodology**

## Implementation Components

### 1. WeightedGradientTrainer

**File**: `APPFL/src/appfl/algorithm/trainer/weighted_gradient_trainer.py`

The trainer extends VanillaTrainer with:
- Label distribution analysis for local datasets
- Weight computation based on global vs local distributions
- Per-sample gradient weighting during backpropagation

**Key Methods**:
- `set_global_label_distribution()`: Receives global distribution and computes local weights
- `_compute_local_label_distribution()`: Analyzes local training data
- `_compute_label_weights()`: Computes weight ratios
- `_train_batch()`: Applies weights during training

### 2. Standard FedAvg Aggregator

**Aggregator**: `FedAvgAggregator` (standard APPFL implementation)

The algorithm uses the standard FedAvg aggregator without modifications. The gradient weighting happens during local training, and the standard aggregation naturally handles the weighted gradients.

### 3. Utility Functions

**Function**: `compute_global_label_distribution()`

Utility function to compute global label distribution across all client datasets before training begins.

## Usage Example

### Configuration

Use the provided configuration file with **standard FedAvg aggregator**:
```yaml
# server_fedavg_weighted_gradient.yaml
server_configs:
  aggregator: "FedAvgAggregator"  # Standard FedAvg
  
client_configs:
  train_configs:
    trainer: "WeightedGradientTrainer"
    use_gradient_weighting: True
```

### Running the Algorithm

```bash
cd APPFL/examples/serial
python run_weighted_gradient_fl.py \
    --server_config ./resources/configs/cifar10/server_fedavg_weighted_gradient.yaml \
    --client_config ./resources/configs/cifar10/client_1.yaml \
    --num_clients 10
```

### Example for CIFAR-10

For CIFAR-10 with 10 classes, if:
- Global dataset has uniform distribution: each class = 10%
- Client 1 has 90% class 0, 10% others
- Client 2 has 90% class 1, 10% others

Then weights would be:
```python
# Client 1 weights
{0: 0.10/0.90 ≈ 0.11, 1: 0.10/0.01 = 10.0, ...}

# Client 2 weights  
{0: 0.10/0.01 = 10.0, 1: 0.10/0.90 ≈ 0.11, ...}
```

This ensures gradients from underrepresented classes are amplified while overrepresented classes are diminished.

## Benefits

1. **Non-IID Robustness**: Handles severe label distribution skews across clients
2. **Standard FedAvg Compatible**: Uses unmodified FedAvg aggregation
3. **Theoretical Grounding**: Based on principled reweighting of gradients
4. **Easy Integration**: Minimal changes to existing FedAvg infrastructure
5. **Automatic Weight Computation**: No manual parameter tuning required

## Performance Considerations

- **Memory**: Requires storing label distributions and weights for each client
- **Computation**: Additional overhead during pre-training analysis phase
- **Communication**: Standard FedAvg communication costs (no additional overhead)

## Comparison with Standard FedAvg

| Aspect | Standard FedAvg | Weighted Gradient FL |
|--------|----------------|---------------------|
| Aggregator | FedAvgAggregator | FedAvgAggregator (same) |
| Training | Standard | Weighted gradients |
| Non-IID handling | Limited | Strong |
| Setup complexity | Simple | Moderate |
| Pre-training overhead | None | Label analysis |
| Training overhead | Minimal | Per-sample weighting |

## Advanced Configuration

### Custom Weight Smoothing

```python
# In trainer weight computation
if local_prob > 0:
    weight = global_prob / local_prob
else:
    weight = 0.0  # or some smoothing factor
```

### Disable Gradient Weighting

```yaml
client_configs:
  train_configs:
    use_gradient_weighting: False  # Falls back to standard training
```

## Algorithm Flow

1. **Initialize**: Create standard FedAvg server and weighted gradient trainers
2. **Analyze**: Compute global label distribution across all clients
3. **Setup**: Each trainer computes local weights based on global distribution
4. **Train**: Clients apply weights during local training
5. **Aggregate**: Server uses standard FedAvg to combine models
6. **Repeat**: Continue for specified rounds

## Troubleshooting

### Common Issues

1. **No weights computed**: Ensure global distribution is set before training
2. **Memory issues**: Reduce batch size during weight computation phase
3. **NaN weights**: Check for clients with zero samples for certain labels

### Debug Mode

Enable detailed logging to monitor weight computation:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Testing

Run the test suite to validate the implementation:
```bash
cd APPFL/examples/serial
python test_weighted_gradient_fl.py
```

## Citation

If you use this weighted gradient federated learning algorithm in your research, please cite:

```bibtex
@article{weighted_gradient_fl,
  title={Weighted Gradient Federated Learning for Non-IID Data},
  author={[Author Name]},
  journal={[Journal]},
  year={2024}
}
```

## License

This implementation is part of the APPFL framework and follows the same license terms. 