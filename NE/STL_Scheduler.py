from __future__ import annotations
from typing import Dict, Callable, Union

import numpy as np
import torch
from torch._C import device
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import math

from NE.utils import get_W

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sparse_set(weights:list[torch.Tensor], sparsity:list[float], keep_first_layer_dense:bool=True, used_in_DrQ:bool=False)->list[float]:
    # Compute the sparsity of each layer

    # We adopt the Erdos Renyi strategy to sparsify the networks.
    # This implementation is based on the open-source repository of the paper 
    # "Do We Actually Need Dense Over-Parameterization? In-Time Over-Parameterization in Sparse Training".
    # and "Dormant neuron phenominem in Reinforcement learning".
    # Please refer to https://github.com/Shiweiliuiiiiiii/In-Time-Over-Parameterization/blob/main/sparselearning/core.py .

    # This strategy will try to find the right epsilon which makes the following equations hold:
    #            (1-S)*\sum_{l in L}{I_l*O_l} = \sum_{l in L}{(1-S_l)*I_l*O_l}
    #               1-S_l = epsilon*(I_l+O_l)/(I_l*O_l) , l in L
    # where L denotes the indexes of all the layers.
    # However it is possible that one of the sparsity is less than 0, in which case we hold the 
    # sparsity of this layer remains as 0, and reallocation the sparsities in the rest layers.
    # Denote the indexes of the omitted layers as L', the equations become:
    #            (1-S)*\sum_{l in L}{I_l*O_l} = \sum_{l in L}{(1-S_l)*I_l*O_l}
    #               1-S_l = epsilon*(I_l+O_l)/(I_l*O_l) , l in L/L'
    #                                S_l = 0, l in L'

    ans = []
    is_valid = False
    dense_layers = set()
    # chose the fixed dence layers
    if keep_first_layer_dense:
        dense_layers.add(0)
    if used_in_DrQ:
        #mode 1: actor: skip embedding layer and the last layer
        #dense_layers.add(1)
        #dense_layers.add(3)
        #mode 2: critic: skip embedding layer and the last layer
        dense_layers.add(1)
        dense_layers.add(3)
        dense_layers.add(6)
        dense_layers.add(8)
    while not is_valid:
        divisor = 0
        rhs = 0
        raw_probabilities = {}
        for i, w in enumerate(weights):
            n_param = w.numel()
            n_zeros = n_param * sparsity
            n_ones = n_param * (1 - sparsity)

            if i in dense_layers:
                rhs -= n_zeros
            else:
                rhs += n_ones
                raw_probabilities[i] = np.sum(w.shape) / w.numel()
                                            
                divisor += raw_probabilities[i] * n_param
        if len(dense_layers) == len(weights): raise Exception('Cannot set a proper sparsity')
        epsilon = rhs / divisor
        max_prob = np.max(list(raw_probabilities.values()))
        max_prob_one = max_prob * epsilon
        if max_prob_one > 1:
            is_valid = False
            for weight_i, weight_raw_prob in raw_probabilities.items():
                if weight_raw_prob == max_prob:
                    #print(f"Sparsity of layer {weight_i} has to be set to 0.")
                    dense_layers.add(weight_i)
        else:
            is_valid = True
    for i in range(len(weights)):
        if i in dense_layers:
            ans.append(0)
        else:
            ans.append(1 - raw_probabilities[i] * epsilon)  
    return ans

class IndexMaskHook:
    def __init__(self, layer:torch.Tensor, scheduler:STL_Scheduler):
        self.layer = layer
        self.scheduler = scheduler
        self.dense_grad = None

    def __name__(self):
        return 'IndexMaskHook'

    @torch.no_grad()
    def __call__(self, grad:torch.Tensor):
        mask = self.scheduler.backward_masks[self.layer]

        # only calculate dense_grads when necessary
        if self.scheduler.check_if_backward_hook_should_accumulate_grad():
            if self.dense_grad is None:
                # initialize as all 0s so we can do a rolling average
                self.dense_grad = torch.zeros_like(grad)
            self.dense_grad += grad/self.scheduler.grad_accumulation_n
        else:
            self.dense_grad = None

        return grad * mask

def _create_step_wrapper(scheduler: STL_Scheduler, optimizer: torch.optim.Optimizer):
    _unwrapped_step = optimizer.step
    def _wrapped_step():
        _unwrapped_step()
        scheduler.reset_momentum()
        scheduler.apply_mask_to_weights()
    optimizer.step = _wrapped_step


class STL_Scheduler:
    """
    Small-to-Large (STL) Training Scheduler for managing network sparsity during training.
    This scheduler implements a dynamic pruning and growing mechanism that allows networks
    to adapt their structure during training, starting from a sparse initialization and
    gradually growing based on performance metrics.

    Key features:
    - Dynamic pruning and growing of network connections
    - Support for both static and dynamic topology
    - Momentum reset and gradient masking
    - Distributed training support
    - Small-to-large (STL) training mode
    - Uniform or cosine annealing for growth rate
    - ReDo-style activation-based pruning

    Args:
        model: PyTorch model to be sparsified
        optimizer: Optimizer used for training
        static_topo (bool): Whether to maintain static topology during training
        sparsity (float): Target sparsity ratio for the network
        T_end (int): Total number of training steps
        delta (int): Interval between topology updates
        zeta (float): Initial growth rate parameter
        random_grow (bool): Whether to use random connection growth
        grad_accumulation_n (int): Number of steps to accumulate gradients
        stl (bool): Whether to use small-to-large training mode
        uni (bool): Whether to use uniform annealing instead of cosine
        use_simple_metric (bool): Whether to use simple magnitude-based pruning
        tau (float): Threshold for ReDo-style pruning (only used if use_simple_metric is False)
        init_method (str): Initialization method for newly grown connections ('zero' or 'lecun')
        initial_stl_sparsity (float): initial sparsity for STL mode
    """

    def __init__(
        self, 
        model, 
        optimizer, 
        static_topo=False, 
        sparsity=0, 
        T_end=None, 
        delta:int=100, 
        zeta:float=0.3, 
        random_grow=False, 
        grad_accumulation_n:int=1,
        stl=True,
        uni=True,
        use_simple_metric=True,
        tau:float=0.0,
        init_method:str='zero',
        initial_stl_sparsity:float=0.8,
        complex_prune:bool=False  # New parameter for target STL sparsity
        ):
        # Initialize model and optimizer references
        self.model = model
        self.optimizer:torch.optim.Optimizer = optimizer
        self.complex_prune = complex_prune
        # Training mode flags
        self.stl = stl  # Small-to-large training mode
        self.uni = uni  # Uniform annealing mode
        self.reset = True
        self.random_grow = random_grow
        self.use_simple_metric = use_simple_metric
        self.tau = tau
        self.init_method = init_method  # Store initialization method
        
        # Activation tracking
        self.activations = {}
        self.activation_hooks = []
        
        # Initialize space tracking for STL mode
        self.n_init_space = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
        
        # Extract weights and layer types from model
        self.W, self._layers_type = get_W(model)

        # Modify optimizer to include DST-specific steps
        _create_step_wrapper(self, optimizer)
            
        # Sparsity configuration
        self.sparsity = sparsity
        self.sparsity_stl = initial_stl_sparsity  # Use the provided target sparsity
        self.N = [torch.numel(w) for w in self.W]  # Number of parameters per layer

        # Training configuration
        self.static_topo = static_topo
        self.grad_accumulation_n = grad_accumulation_n
        self.backward_masks:list[torch.Tensor] = None
        
        # Initialize sparsity masks
        if self.stl:
            self.S = sparse_set(self.W, self.sparsity_stl, False)
        else:
            self.S = sparse_set(self.W, sparsity, False)

        # Initial sparsification
        self.weight_sparsify()
        self.flage = True
        
        # Step tracking
        self.step = 0
        self.dst_steps = 0

        # Schedule parameters
        self.delta_T = delta  # Update interval
        self.zeta = zeta     # Growth rate parameter
        self.T_end = T_end   # Total training steps

        # Register backward hooks for gradient masking
        self.backward_hook_objects:list[IndexMaskHook] = []
        for i, w in enumerate(self.W):
            if self.S[i] <= 0:
                self.backward_hook_objects.append(None)
                continue

            if getattr(w, '_has_dst_backward_hook', False):
                raise Exception('This model already has been registered to a DST_Scheduler.')
        
            self.backward_hook_objects.append(IndexMaskHook(i, self))
            w.register_hook(self.backward_hook_objects[-1])
            setattr(w, '_has_dst_backward_hook', True)

        # Register activation hooks if using ReDo-style pruning
        if not self.use_simple_metric:
            self._register_activation_hooks()

        assert self.grad_accumulation_n > 0 and self.grad_accumulation_n < delta

    def _get_activation(self, name: str) -> Callable:
        """Create activation hook for a layer."""
        def hook(module, input, output):
            self.activations[name] = F.relu(output)
        return hook

    def _register_activation_hooks(self):
        """Register activation hooks for all relevant layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hook = self._get_activation(name)
                self.activation_hooks.append(module.register_forward_hook(hook))

    def _remove_activation_hooks(self):
        """Remove all activation hooks."""
        for hook in self.activation_hooks:
            hook.remove()
        self.activation_hooks.clear()

    def _get_redo_score(self, layer_idx: int) -> torch.Tensor:
        """Calculate ReDo-style importance score for a layer."""
        layer = self.W[layer_idx]
        layer_name = f"{layer_idx}"
        
        if layer_name not in self.activations:
            return torch.abs(layer)
            
        activation = self.activations[layer_name]
        
        if isinstance(layer, nn.Conv2d):
            score = activation.abs().mean(dim=(0, 2, 3))
        else:  # Linear layer
            score = activation.abs().mean(dim=0)
            
        # Normalize scores by mean activation
        normalized_score = score / (score.mean() + 1e-9)
        return normalized_score

    def _reset_adam_moments(self, layer_idx: int, mask: torch.Tensor) -> None:
        """Resets the momentum states of the Adam optimizer for dormant neurons.
        
        Args:
            layer_idx: Index of the layer to reset
            mask: Boolean mask indicating which neurons to reset
        """
        assert isinstance(self.optimizer, torch.optim.Adam), "Moment resetting currently only supported for Adam optimizer"
        
        # Get all parameters from the first parameter group
        params = self.optimizer.param_groups[0]['params']
        
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
            # Create a new step tensor for masked parameters
            if "step" in weight_state:
                weight_state["step"] = torch.zeros_like(weight_state["step"])

            # Reset momentum states for current layer's bias if it exists
            if bias_idx < len(params):
                bias_param = params[bias_idx]
                if bias_param in self.optimizer.state:
                    bias_state = self.optimizer.state[bias_param]
                    bias_state["exp_avg"][mask] = 0.0
                    bias_state["exp_avg_sq"][mask] = 0.0
                    if "step" in bias_state:
                        bias_state["step"] = torch.zeros_like(bias_state["step"])

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
                if "step" in next_weight_state:
                    next_weight_state["step"] = torch.zeros_like(next_weight_state["step"])

        except (IndexError, KeyError) as e:
            print(f"")
            return

    @torch.no_grad()
    def weight_sparsify(self):
        """
        Initial sparsification of the network weights using magnitude-based pruning.
        Creates and applies masks to maintain sparsity during training.
        """
        is_dist = dist.is_initialized()
        self.backward_masks = []
        for l, w in enumerate(self.W):
            if self.S[l] < 0:
                self.backward_masks.append(None)
                continue

            n = self.N[l]
            s = int(self.S[l] * n)
            score_drop = torch.abs(w)
            # Create drop mask based on magnitude
            _, sorted_indices = torch.topk(score_drop.view(-1), k=n-s)
            flat_mask = torch.zeros(n, device=w.device)
            flat_mask[sorted_indices] = 1
            mask = torch.reshape(flat_mask, w.shape)

            if is_dist:
                dist.broadcast(mask, 0)

            mask = mask.bool()
            w *= mask
            self.backward_masks.append(mask)

    @torch.no_grad()
    def reset_momentum(self):
        """
        Reset momentum buffers for masked weights to prevent accumulation
        in pruned connections.
        """
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            if s <= 0:
                continue

            param_state = self.optimizer.state[w]
            if 'momentum_buffer' in param_state:
                buf = param_state['momentum_buffer']
                buf *= mask

    @torch.no_grad()
    def apply_mask_to_weights(self):
        """
        Apply sparsity masks to network weights to maintain sparsity.
        """
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            if s <= 0:
                continue
            w *= mask

    @torch.no_grad()
    def apply_mask_to_gradients(self):
        """
        Apply sparsity masks to gradients to prevent updates to pruned connections.
        """
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            # if sparsity is 0%, skip
            if s <= 0:
                continue
            w.grad *= mask
    
    def check_if_backward_hook_should_accumulate_grad(self):
        """
        Determine if gradients should be accumulated for the next DST step.
        Returns True if within grad_accumulation_n steps of the next update.
        """
        if self.step >= self.T_end:
            return False

        steps_til_next_dst_step = self.delta_T - (self.step % self.delta_T)
        return steps_til_next_dst_step <= self.grad_accumulation_n

    def cosine_annealing(self):
        """
        Compute cosine annealing schedule for growth rate.
        """
        return self.zeta / 2 * (1 + np.cos((self.step * np.pi) / self.T_end))

    def uniform_annealing(self):
        """
        Compute uniform annealing schedule for growth rate.
        """
        return 0.008

    def __call__(self, end_grow, state=None, action=None, ac_cr="actor"):
        """
        Main scheduler step function. Called after each training step.
        
        Args:
            end_grow (bool): Whether to stop growing the network
            state (torch.Tensor, optional): Current state for actor/critic networks
            action (torch.Tensor, optional): Current action for critic networks
            ac_cr (str): Whether this is for actor ("actor") or critic ("critic") network
            
        Returns:
            bool: Whether to proceed with optimizer step
        """
        self.step += 1
        if self.static_topo:
            return True
        if (self.step % self.delta_T) == 0 and self.step < self.T_end:
            if end_grow and self.flage:
                self.step = 0
                self.zeta = 0.2
                self.flage = False
            self._dst_step(end_grow, state, action, ac_cr)
            self.dst_steps += 1
            return False
        return True

    @torch.no_grad()
    def _dst_step(self, end_grow, state=None, action=None, ac_cr="actor"):
        """
        Perform dynamic sparse training step.
        Updates network topology by pruning and growing connections based on
        importance scores and current training state.
        
        Args:
            end_grow (bool): Whether to stop growing the network
            state (torch.Tensor, optional): Current state for actor/critic networks
            action (torch.Tensor, optional): Current action for critic networks
            ac_cr (str): Whether this is for actor ("actor") or critic ("critic") network
        """
        if self.stl and (not end_grow):
            # Small-to-large training mode
            total_pruned_num = 0
            total_num = 0
            
            # Select annealing schedule
            if self.uni:
                drop_fraction = self.uniform_annealing()
            else:
                drop_fraction = self.cosine_annealing()

            is_dist = dist.is_initialized()
            world_size = dist.get_world_size() if is_dist else None

            # Get score drops if state/action are provided
            if state is not None:
                if ac_cr == "actor":
                    score_drops, each_nums = self.model.score_drop(state)
                else:
                    score_drops, each_nums = self.model.score_drop(state, action)
            else:
                score_drops = None
                each_nums = None

            for l, w in enumerate(self.W):
                if self.S[l] <= 0:
                    continue

                current_mask = self.backward_masks[l]

                # Calculate importance scores for pruning and growing
                if self.use_simple_metric:
                    score_drop = torch.abs(w)
                else:
                    # Use ReDo-style pruning with dormant neuron detection
                    score_drop = self._get_redo_score(l)
                    # Identify dormant neurons (those with very low activation)
                    dormant_mask = score_drop < self.tau
                    # Reset momentum for dormant neurons
                    self._reset_adam_moments(l, dormant_mask)

                if not self.random_grow:
                    score_grow = torch.abs(self.backward_hook_objects[l].dense_grad)
                else:
                    score_grow = torch.rand(self.backward_hook_objects[l].dense_grad.size()).to(device)

                # Synchronize scores in distributed setting
                if is_dist:
                    dist.all_reduce(score_drop)
                    score_drop /= world_size
                    dist.all_reduce(score_grow)
                    score_grow /= world_size

                # Calculate pruning and growing quantities
                n_total = self.N[l]
                n_ones = torch.sum(current_mask).item()

                # Create pruning mask
                sorted_score, sorted_indices = torch.topk(score_drop.view(-1), k=n_total)
                sorted_score = sorted_score[:n_ones]
                total_num += n_ones

                # Initialize growing space for STL mode
                if self.n_init_space[l]==0.:
                    n_init_space = self.sparsity_stl*n_total - n_ones
                    self.n_init_space[l] = n_init_space

                # Calculate number of connections to grow and prune
                n_grow = int(self.n_init_space[l] * drop_fraction)
                n_prune = int(n_grow * 0.4)
                n_keep = n_ones - n_prune

                # Create pruning mask
                new_values = torch.where(
                    torch.arange(n_total, device=w.device) < n_keep,
                    torch.ones_like(sorted_indices),
                    torch.zeros_like(sorted_indices))
                mask1 = new_values.scatter(0, sorted_indices, new_values)

                total_pruned_num += n_prune

                # Prepare growing scores
                score_grow = score_grow.view(-1)
                score_grow_lifted = torch.where(
                    mask1 == 1,
                    torch.ones_like(mask1) * (torch.min(score_grow) - 1),
                    score_grow)

                # Create growing mask
                _, sorted_indices = torch.topk(score_grow_lifted, k=n_total)
                if self.stl:
                    new_values = torch.where(
                        torch.arange(n_total, device=w.device) < n_grow,
                        torch.ones_like(sorted_indices),
                        torch.zeros_like(sorted_indices))
                else:
                    new_values = torch.where(
                        torch.arange(n_total, device=w.device) < n_prune,
                        torch.ones_like(sorted_indices),
                        torch.zeros_like(sorted_indices))
                mask2 = new_values.scatter(0, sorted_indices, new_values)

                # Update weights and masks
                mask2_reshaped = torch.reshape(mask2, current_mask.shape)
                grow_tensor = torch.zeros_like(w)
                new_connections = ((mask2_reshaped == 1) & (current_mask == 0))
                
                # Initialize new weights based on chosen method
                if self.init_method == 'lecun':
                    self._lecun_init(w, new_connections)
                else:  # default to zero initialization
                    new_weights = torch.where(new_connections.to(w.device), grow_tensor, w)
                    w.data = new_weights
                
                mask_combined = torch.reshape(mask1 + mask2, current_mask.shape).bool()
                current_mask.data = mask_combined

        else:
            # keep dynmaic topology after the growing phase
            total_pruned_num = 0
            total_num = 0

            drop_fraction = self.cosine_annealing()

            is_dist = dist.is_initialized()
            world_size = dist.get_world_size() if is_dist else None

            for l, w in enumerate(self.W):
                # if sparsity is 0%, skip
                if self.S[l] <= 0:
                    continue

                current_mask = self.backward_masks[l]

                # Calculate importance scores for pruning and growing
                score_drop = torch.abs(w)
                if not self.random_grow:
                    score_grow = torch.abs(self.backward_hook_objects[l].dense_grad)
                else:
                    score_grow = torch.rand(self.backward_hook_objects[l].dense_grad.size()).to(device)

                # Synchronize scores in distributed setting
                if is_dist:
                    dist.all_reduce(score_drop)
                    score_drop /= world_size
                    dist.all_reduce(score_grow)
                    score_grow /= world_size

                # Calculate pruning and growing quantities
                n_total = self.N[l]
                n_ones = torch.sum(current_mask).item()

                # Create pruning mask
                sorted_score, sorted_indices = torch.topk(score_drop.view(-1), k=n_total)
                sorted_score = sorted_score[:n_ones]
                n_dead = torch.sum(sorted_score > 0.).item()
                n_prune = int(n_ones * drop_fraction)
                total_num += n_ones
                n_keep = n_ones - n_prune

                # Create pruning mask
                new_values = torch.where(
                    torch.arange(n_total, device=w.device) < n_keep,
                    torch.ones_like(sorted_indices),
                    torch.zeros_like(sorted_indices))
                mask1 = new_values.scatter(0, sorted_indices, new_values)

                total_pruned_num += n_prune

                # Prepare growing scores
                score_grow = score_grow.view(-1)
                score_grow_lifted = torch.where(
                    mask1 == 1,
                    torch.ones_like(mask1) * (torch.min(score_grow) - 1),
                    score_grow)

                # Create growing mask
                _, sorted_indices = torch.topk(score_grow_lifted, k=n_total)
                new_values = torch.where(
                    torch.arange(n_total, device=w.device) < n_prune,
                    torch.ones_like(sorted_indices),
                    torch.zeros_like(sorted_indices))
                mask2 = new_values.scatter(0, sorted_indices, new_values)

                # Update weights and masks
                mask2_reshaped = torch.reshape(mask2, current_mask.shape)
                grow_tensor = torch.zeros_like(w)
                new_connections = ((mask2_reshaped == 1) & (current_mask == 0))
                
                # Initialize new weights based on chosen method
                if self.init_method == 'lecun':
                    self._lecun_init(w, new_connections)
                else:  # default to zero initialization
                    new_weights = torch.where(new_connections.to(w.device), grow_tensor, w)
                    w.data = new_weights
                
                mask_combined = torch.reshape(mask1 + mask2, current_mask.shape).bool()
                current_mask.data = mask_combined

        # Update momentum and apply masks
        if self.complex_prune:
            self.reset_momentum()
        self.apply_mask_to_weights()
        self.apply_mask_to_gradients()

    def __del__(self):
        """Clean up activation hooks when the scheduler is deleted."""
        if not self.use_simple_metric:
            self._remove_activation_hooks()

    @property
    def state_dict(self):
        """
        Get scheduler state for checkpointing.
        """
        return {
            'S': self.S,
            'N': self.N,
            'delta_T': self.delta_T,
            'zeta': self.zeta,
            'T_end': self.T_end,
            'static_topo': self.static_topo,
            'grad_accumulation_n': self.grad_accumulation_n,
            'step': self.step,
            'dst_steps': self.dst_steps,
            'backward_masks': self.backward_masks
        }

    def load_state_dict(self, state_dict:Dict):
        """
        Load scheduler state from checkpoint.
        
        Args:
            state_dict (Dict): State dictionary to load
        """
        for k, v in state_dict.items():
            setattr(self, k, v)

    def _lecun_init(self, layer: Union[nn.Linear, nn.Conv2d], mask: torch.Tensor) -> None:
        """Initialize weights using Lecun initialization for newly grown connections."""
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
        variance = 1.0 / fan_in
        stddev = math.sqrt(variance) / 0.87962566103423978
        
        with torch.no_grad():
            layer.weight[mask] = nn.init._no_grad_trunc_normal_(
                layer.weight[mask], mean=0.0, std=1.0, a=-2.0, b=2.0
            )
            layer.weight[mask] *= stddev
            
            if layer.bias is not None:
                layer.bias.data[mask] = 0.0
