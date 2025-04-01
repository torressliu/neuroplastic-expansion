from functools import partial
import math
from typing import Dict, List, Tuple, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class BaseReDo:
    """Base class for ReDo (Reinitialization of Dormant Neurons) implementations.
    
    Args:
        tau (float): Threshold for determining dormant neurons based on activation/gradient values
        use_lecun_init (bool): Whether to use LeCun initialization instead of Kaiming initialization
        frequency (int): Number of steps between neuron reinitialization checks
        optimizer (optim.Adam): Optimizer instance for resetting momentum states
        reset_steps (List[int], optional): Steps range to perform resets [start_step, end_step]. 
                                         If None, resets will continue indefinitely.
    """
    def __init__(self, model: nn.Module, tau: float = 0, use_lecun_init: bool = False, 
                 frequency: int = 1000, optimizer: optim.Adam = None, reset_steps: List[int] = None):
        assert tau >= 0, "tau must be non-negative"
        self.model = model
        self.tau = tau
        self.use_lecun_init = use_lecun_init
        self.current_step = 0
        self.frequency = frequency
        self.optimizer = optimizer
        self.reset_steps = reset_steps
        
        # Validate reset_steps format
        if self.reset_steps is not None:
            assert len(self.reset_steps) == 2, "reset_steps must be a list with [start_step, end_step]"
            assert self.reset_steps[0] <= self.reset_steps[1], "start_step must be <= end_step"

    @staticmethod
    def _kaiming_uniform_reinit(layer: Union[nn.Linear, nn.Conv2d], mask: torch.Tensor) -> None:
        """Partially re-initializes a layer using Kaiming uniform."""
        fan_in = nn.init._calculate_correct_fan(tensor=layer.weight, mode="fan_in")
        gain = nn.init.calculate_gain(nonlinearity="relu", param=math.sqrt(5))
        std = gain / math.sqrt(fan_in)
        bound = math.sqrt(3.0) * std
        
        # Reset weights
        with torch.no_grad():
            layer.weight.data[mask, ...] = torch.empty_like(
                layer.weight.data[mask, ...]
            ).uniform_(-bound, bound)

            # Reset bias if it exists
            if layer.bias is not None:
                if isinstance(layer, nn.Conv2d):
                    if fan_in != 0:
                        bound = 1 / math.sqrt(fan_in)
                        layer.bias.data[mask, ...] = torch.empty_like(
                            layer.bias.data[mask, ...]
                        ).uniform_(-bound, bound)
                else:
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    layer.bias.data[mask, ...] = torch.empty_like(
                        layer.bias.data[mask, ...]
                    ).uniform_(-bound, bound)

    @staticmethod
    def _lecun_normal_reinit(layer: Union[nn.Linear, nn.Conv2d], mask: torch.Tensor) -> None:
        """Partially re-initializes a layer using Lecun normal."""
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
        variance = 1.0 / fan_in
        stddev = math.sqrt(variance) / 0.87962566103423978
        
        # Reset weights
        with torch.no_grad():
            layer.weight[mask] = nn.init._no_grad_trunc_normal_(
                layer.weight[mask], mean=0.0, std=1.0, a=-2.0, b=2.0
            )
            layer.weight[mask] *= stddev
            
            # Reset bias if it exists
            if layer.bias is not None:
                layer.bias.data[mask] = 0.0

    def _reset_adam_moments(self, reset_masks) -> None:
        """Resets the momentum states of the Adam optimizer for dormant neurons.
        
        Args:
            reset_masks: List of boolean masks indicating which neurons to reset
        """
        assert isinstance(self.optimizer, optim.Adam), "Moment resetting currently only supported for Adam optimizer"
        
        # Get all parameters from the first parameter group
        params = self.optimizer.param_groups[0]['params']
        
        for layer_idx, mask in enumerate(reset_masks):
            try:
                # Get indices for current layer's weight and bias parameters
                weight_idx = 2 * layer_idx
                bias_idx = 2 * layer_idx + 1
                next_weight_idx = 2 * (layer_idx + 1)

                # Reset momentum states for current layer's weights
                weight_param = params[weight_idx]
                weight_state = self.optimizer.state[weight_param]
                
                # Reset first and second moment estimates
                weight_state["exp_avg"][mask, ...] = 0.0
                weight_state["exp_avg_sq"][mask, ...] = 0.0
                weight_state["step"] = 0

                # Reset momentum states for current layer's bias if it exists
                if bias_idx < len(params):
                    bias_param = params[bias_idx]
                    if bias_param in self.optimizer.state:
                        bias_state = self.optimizer.state[bias_param]
                        bias_state["exp_avg"][mask] = 0.0
                        bias_state["exp_avg_sq"][mask] = 0.0
                        bias_state["step"] = 0

                # Reset momentum states for next layer's weights (output connections)
                if next_weight_idx < len(params):
                    next_weight_param = params[next_weight_idx]
                    next_weight_state = self.optimizer.state[next_weight_param]
                    
                    # Handle transition from convolutional to linear layer
                    if len(weight_state["exp_avg"].shape) == 4 and len(next_weight_state["exp_avg"].shape) == 2:
                        num_repetition = next_weight_state["exp_avg"].shape[1] // mask.shape[0]
                        linear_mask = torch.repeat_interleave(mask, num_repetition)
                        next_weight_state["exp_avg"][:, linear_mask] = 0.0
                        next_weight_state["exp_avg_sq"][:, linear_mask] = 0.0
                    else:
                        # Standard case: same layer type connections
                        next_weight_state["exp_avg"][:, mask, ...] = 0.0
                        next_weight_state["exp_avg_sq"][:, mask, ...] = 0.0
                    next_weight_state["step"] = 0

            except (IndexError, KeyError) as e:
                print(f"Warning: Layer {layer_idx} parameter not found in optimizer state")
                continue

    def _reset_dormant_neurons(self, model: nn.Module, redo_masks: List[torch.Tensor]) -> nn.Module:
        """Re-initializes the weights of dormant neurons."""
        # Only get Conv2d and Linear layers
        layers = [(name, layer) for name, layer in model.named_modules() 
                 if isinstance(layer, (nn.Conv2d, nn.Linear))]
        
        
        assert len(redo_masks) == len(layers) - 1, (
            f"Number of masks ({len(redo_masks)}) must match number of layers-1 ({len(layers)-1})"
        )

        # Reset ingoing weights
        with torch.no_grad():
            for i in range(len(layers)-1):
                mask = redo_masks[i]
                layer = layers[i][1]
                next_layer = layers[i + 1][1]

                # Skip if no dead neurons
                if torch.all(~mask):
                    continue

                # Reset weights using specified initialization
                if self.use_lecun_init:
                    self._lecun_normal_reinit(layer, mask)
                else:
                    self._kaiming_uniform_reinit(layer, mask)


                # Reset outgoing weights to 0
                if isinstance(layer, nn.Conv2d) and isinstance(next_layer, nn.Linear):
                    # Handle conv to linear transition
                    num_repeat = next_layer.weight.data.shape[1] // mask.shape[0]
                    linear_mask = torch.repeat_interleave(mask, num_repeat)
                    next_layer.weight.data[:, linear_mask] = 0.0
                    # Reset gradients for outgoing weights
                    if next_layer.weight.grad is not None:
                        next_layer.weight.grad[:, linear_mask] = 0.0
                else:
                    # Standard case: both conv or both linear
                    next_layer.weight.data[:, mask, ...] = 0.0
                    # Reset gradients for outgoing weights
                    if next_layer.weight.grad is not None:
                        next_layer.weight.grad[:, mask, ...] = 0.0

        return model

    def _get_redo_masks(self, *args, **kwargs) -> List[torch.Tensor]:
        """Abstract method to compute masks for dormant neurons."""
        raise NotImplementedError("Subclasses must implement _get_redo_masks")

    def step(self, *args, **kwargs) -> Dict[str, any]:
        """Abstract method to perform ReDo step."""
        raise NotImplementedError("Subclasses must implement step")


class ReDo(BaseReDo):
    """Activation-based ReDo implementation."""
    
    def _get_activation(self, name: str, activations: Dict[str, torch.Tensor]) -> Callable:
        def hook(layer: Union[nn.Linear, nn.Conv2d], 
                input: Tuple[torch.Tensor], 
                output: torch.Tensor) -> None:
            activations[name] = F.relu(output)
        return hook

    def _get_redo_masks(self, activations: Dict[str, torch.Tensor], tau: float) -> List[torch.Tensor]:
        """Computes masks based on activation values."""
        masks = []
        
        # Last activation (q-values) are never reset
        for name, activation in list(activations.items())[:-1]:
            if activation.ndim == 4:  # Conv layer
                score = activation.abs().mean(dim=(0, 2, 3))
            else:  # Linear layer
                score = activation.abs().mean(dim=0)
                
            # Normalize scores by mean activation
            normalized_score = score / (score.mean() + 1e-9)
            
            # Create mask (True for dormant neurons)
            layer_mask = torch.zeros_like(normalized_score, dtype=torch.bool)
            if tau > 0.0:
                layer_mask[normalized_score <= tau] = 1
            else:
                layer_mask[torch.isclose(normalized_score, torch.zeros_like(normalized_score))] = 1
            masks.append(layer_mask)
            
        return masks

    @torch.inference_mode()
    def step(self, obs: torch.Tensor) -> Dict[str, any]:
        """Performs ReDo step using activation values."""
        self.current_step += 1
        
        # Modify condition to support step range
        in_reset_range = True
        if self.reset_steps is not None:
            in_reset_range = self.current_step >= self.reset_steps[0] and self.current_step <= self.reset_steps[1]
            
        if self.current_step % self.frequency == 0 and in_reset_range:
            activations = {}
            
            # Register hooks
            handles = []
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    handles.append(
                        module.register_forward_hook(
                            self._get_activation(name, activations)
                        )
                    )

            # Get activations
            if isinstance(obs, tuple):
                _ = self.model(*obs)
            else:
                _ = self.model(obs)

            # Get masks for logging (tau=0) and resetting
            zero_masks = self._get_redo_masks(activations, 0.0)
            total_neurons = sum(torch.numel(mask) for mask in zero_masks)
            zero_count = sum(torch.sum(mask) for mask in zero_masks)
            zero_fraction = (zero_count / total_neurons) * 100

            masks = self._get_redo_masks(activations, self.tau)
            dormant_count = sum(torch.sum(mask) for mask in masks)
            dormant_fraction = (dormant_count / total_neurons) * 100

            # Reset dormant neurons if requested
            print(f"Re-initializing dormant neurons")
            print(f"Total neurons: {total_neurons} | "
                    f"Dormant neurons: {dormant_count} | "
                    f"Dormant fraction: {dormant_fraction:.2f}%")
            self.model = self._reset_dormant_neurons(self.model, masks)
            if self.optimizer is not None:
                self._reset_adam_moments(masks)

            # Clean up hooks
            for handle in handles:
                handle.remove()

            return {
                "zero_fraction": zero_fraction,
                "zero_count": zero_count,
                "dormant_fraction": dormant_fraction,
                "dormant_count": dormant_count,
            }
        else:
            return {}


class GradientReDo(BaseReDo):
    """Gradient-based ReDo implementation with multiple reset modes.
    
    Args:
        mode (str): Reset mode selection:
            'threshold' - Reset neurons below normalized gradient threshold
            'percentage' - Reset neurons with lowest gradient magnitudes up to specified percentage
            'hybrid' - Combine threshold and percentage-based resetting
        tau (float): Threshold for normalized gradient values (used in threshold/hybrid modes)
        percentage (float): Percentage of neurons to reset (0-100) in percentage/hybrid modes
        max_percentage (float): Maximum percentage of neurons to reset (0-100) in hybrid mode
        use_lecun_init (bool): Whether to use LeCun initialization
        frequency (int): Steps between reinitialization checks
        optimizer (optim.Adam): Optimizer for momentum state resetting
        reset_steps (List[int], optional): Steps range for reinitialization [start_step, end_step]
    """
    def __init__(self, model: nn.Module, mode: str = 'threshold', tau: float = 0, 
                 percentage: float = 1, max_percentage: float = 1,
                 use_lecun_init: bool = False, frequency: int = 1000, 
                 optimizer: optim.Adam = None, reset_steps: List[int] = None):
        super().__init__(model, tau, use_lecun_init, frequency, optimizer, reset_steps)
        
        # Validate mode parameter
        assert mode in ['threshold', 'percentage', 'hybrid'], \
            f"Invalid mode: {mode}, must be one of ['threshold', 'percentage', 'hybrid']"
        self.mode = mode
        
        # Validate and convert percentage parameters
        if self.mode in ['percentage', 'hybrid']:
            assert 0 <= percentage <= 100, "percentage must be between 0 and 100"
            self.percentage = percentage / 100  # Convert to decimal
            
        if self.mode == 'hybrid':
            assert 0 <= max_percentage <= 100, "max_percentage must be between 0 and 100"
            self.max_percentage = max_percentage / 100  # Convert to decimal

    def _get_redo_masks(self, tau: float) -> List[torch.Tensor]:
        """Computes masks for neurons to be reinitialized based on gradient magnitudes.
        
        Args:
            tau (float): Threshold for normalized gradient values
            
        Returns:
            List[torch.Tensor]: Boolean masks indicating which neurons to reset
        """
        masks = []
        layers = [(name, layer) for name, layer in self.model.named_modules() 
                 if isinstance(layer, (nn.Conv2d, nn.Linear))]
        
        for name, layer in layers:
            if layer.weight.grad is None:
                # Handle case where gradients are not available
                if isinstance(layer, nn.Conv2d):
                    mask = torch.zeros(layer.out_channels, dtype=torch.bool, device=layer.weight.device)
                else:  # Linear
                    mask = torch.zeros(layer.out_features, dtype=torch.bool, device=layer.weight.device)
            else:
                # Compute mean absolute gradient magnitudes
                if isinstance(layer, nn.Conv2d):
                    grad_magnitude = layer.weight.grad.abs().mean(dim=(1, 2, 3))
                else:  # Linear
                    grad_magnitude = layer.weight.grad.abs().mean(dim=1)
                
                # Apply different reset modes
                if self.mode == 'threshold':
                    normalized_grad = grad_magnitude / (grad_magnitude.mean() + 1e-9)
                    mask = normalized_grad <= tau
                    
                elif self.mode == 'percentage':
                    # Calculate threshold based on percentage of neurons
                    k = int(len(grad_magnitude) * self.percentage)
                    threshold = torch.kthvalue(grad_magnitude, k).values
                    mask = grad_magnitude <= threshold
                    
                elif self.mode == 'hybrid':
                    # First apply threshold filtering
                    normalized_grad = grad_magnitude / (grad_magnitude.mean() + 1e-9)
                    threshold_mask = normalized_grad <= tau
                    
                    # Then apply maximum percentage limit
                    k_max = int(len(grad_magnitude) * self.max_percentage)
                    if threshold_mask.sum() > k_max:
                        # Select k_max neurons with lowest gradients among threshold-satisfying ones
                        combined_grad = grad_magnitude.clone()
                        combined_grad[~threshold_mask] = float('inf')  # Mask non-threshold-satisfying
                        _, indices = torch.topk(combined_grad, k_max, largest=False)
                        mask = torch.zeros_like(grad_magnitude, dtype=torch.bool)
                        mask[indices] = True
                    else:
                        mask = threshold_mask

            masks.append(mask)
            
        return masks
    

    def _reset_dormant_neurons(self, model: nn.Module, redo_masks: List[torch.Tensor]) -> nn.Module:
        """Re-initializes the weights of dormant neurons."""
        # Only get Conv2d and Linear layers
        layers = [(name, layer) for name, layer in model.named_modules() 
                 if isinstance(layer, (nn.Conv2d, nn.Linear))]
        
        
        assert len(redo_masks) == len(layers), (
            f"Number of masks ({len(redo_masks)}) must match number of layers ({len(layers)})"
        )

        # Reset ingoing weights
        with torch.no_grad():
            for i in range(len(layers)):
                mask = redo_masks[i]
                layer = layers[i][1]

                # Skip if no dead neurons
                if torch.all(~mask):
                    continue

                # Reset weights using specified initialization
                if self.use_lecun_init:
                    self._lecun_normal_reinit(layer, mask)
                else:
                    self._kaiming_uniform_reinit(layer, mask)

                # Reset gradients to 0
                if layer.weight.grad is not None:
                    layer.weight.grad[mask] = 0.0
                if layer.bias is not None and layer.bias.grad is not None:
                    layer.bias.grad[mask] = 0.0



        return model

    @torch.no_grad()
    def step(self) -> Dict[str, any]:
        """Performs ReDo step using gradient values."""
        self.current_step += 1
        
        # Modify condition to support step range
        in_reset_range = True
        if self.reset_steps is not None:
            in_reset_range = self.current_step >= self.reset_steps[0] and self.current_step <= self.reset_steps[1]
            
        if self.current_step % self.frequency == 0 and in_reset_range:
            masks = self._get_redo_masks(self.tau)
            dormant_count = sum(torch.sum(mask) for mask in masks)
            total_neurons = sum(torch.numel(mask) for mask in masks)
            dormant_fraction = (dormant_count / total_neurons) * 100

            print(f"Re-initializing dormant neurons based on gradients")
            print(f"Total neurons: {total_neurons} | "
                  f"Dormant neurons: {dormant_count} | "
                  f"Dormant fraction: {dormant_fraction:.2f}%")
            
            self.model = self._reset_dormant_neurons(self.model, masks)
            if self.optimizer is not None:
                self._reset_adam_moments(masks)

            return {
                "dormant_fraction": dormant_fraction,
                "dormant_count": dormant_count,
            }
        else:
            return {}