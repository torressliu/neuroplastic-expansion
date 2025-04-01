# Neuroplastic Expansion in Deep Reinforcement Learning

This repository provides an implementation of Neuroplastic Expansion (*NE*) proposed in the ICLR 2025 paper ["Neuroplastic Expansion in Deep Reinforcement Learning"](https://arxiv.org/abs/2410.07994). The implementation demonstrates how NE can be effectively applied to deep reinforcement learning algorithms, featuring dynamic network topology, small-to-large training, and various sparsification techniques to fully realize the potential of NE.


## Method Introduction

The loss of plasticity in learning agents, analogous to the solidification of neural pathways in biological brains, significantly impedes learning and adaptation
in reinforcement learning due to its non-stationary nature. To address this fundamental challenge, we propose a novel approach, Neuroplastic Expansion (NE),
inspired by cortical expansion in cognitive science. NE maintains learnability and
adaptability throughout the entire training process by dynamically growing the
network from a smaller initial size to its full dimension. The growing chedule is designed
with two key components: (1) elastic topology generation based on potential
gradients, (2) dormant neuron pruning to optimize network expressivity.

## Repository Structure

```
.
├── stl/
│   ├── train.py          # Main training script
│   └── TD3.py           # TD3 algorithm implementation
├── NE/
│   ├── STL_Scheduler.py # Small-to-Large Training scheduler
│   └── utils.py         # Utility functions
└── README.md
```

## Key Features

- **Neuroplastic Training**: Implementation of the neuroplastic expansion method from the ICLR 2025 paper
- **Dynamic Sparse Training**: Automatically adjusts network topology during training based on neuroplastic principles
- **Small-to-Large Training**: Gradually grows network from small to large size, mimicking biological neural development
- **Flexible Weight Initialization**: Supports both zero and Lecun initialization for newly grown connections
- **Dynamic Buffer Review**: Adaptive replay samples for stable training
- **Multiple Training Modes**: Supports static/dynamic topology, layer normalization, and elastic weight averaging

## Method Overview

The implementation follows the neuroplastic expansion method described in the paper, adapted specifically for TD3:

1. **Initial Sparse Network**: Start with a small, sparse network
2. **Dynamic Growth**:
   - Monitor neuron activity and connection importance
   - Grow new connections based on activity patterns
   - Prune unused or dormant weights
3. **Adaptive Learning**:
   - Use elastic weight averaging for stability (option)
   - Implement dynamic replay buffer management (option)
   - Apply N-step returns for better credit assignment (option)



## Core Components

### 1. STL_Scheduler (NE/STL_Scheduler.py)

The Small-to-Large (STL) Training scheduler manages network growth and sparsity during training:

- **Dynamic Topology**: Prunes and grows connections based on importance scores
- **Initialization Methods**: 
  - Zero initialization (default)
  - Lecun initialization (better for training stability)
- **Sparsification Strategies**:
  - Magnitude-based pruning
  - Activity-based pruning
  - Random growth option

Key parameters:
```python
STL_Scheduler(
    model,                # Target model to sparsify
    optimizer,           # Optimizer
    sparsity,           # Target sparsity ratio
    T_end,              # Total training steps
    static_topo=False,  # Whether to maintain static topology
    zeta=0.0008,          # Growth rate parameter
    delta=10000,         # Update interval
    init_method='zero' # Weight initialization method
)
```

### 2. Implementation (stl/TD3.py)

Enhanced TD3 algorithm with support for:

- Sparse actor and critic networks
- Dynamic topology updates
- N-step returns (optional)
- Elastic weight averaging (optional)
- Layer normalization option (optional)

Key features:
```python
TD3(
    state_dim,          # State dimension
    action_dim,         # Action dimension
    max_action,         # Maximum action value
    discount=0.99,      # Discount factor
    tau=0.005,         # Target network update rate
    policy_noise=0.2,   # Target policy noise
    noise_clip=0.5,     # Noise clip range
    policy_freq=2       # Policy update frequency
)
```

## Usage

### Small-to-Large Training

```bash
python stl/train.py \
    --env HalfCheetah \
    --seed 0 \
    --static_actor \
    --stl_critic \
    --init_method 'lecun'\
    --actor_sparsity 0.25 \
    --critic_sparsity 0.25\
    --initial_stl_sparsity 0.8\
    --uni\
    --grad_accumulation_n 1  # Optional, defaults to 1 if not specified
```

## Key Parameters

The following hyperparameters can be configured in train.py:

### Core Training Parameters
- `--exp_id`: Experiment name for organizing results
- `--env`: OpenAI Gym environment name 
- `--seed`: Random seed for reproducibility
- `--start_timesteps`: Initial random exploration steps 
- `--eval_freq`: Evaluation frequency in timesteps 
- `--max_timesteps`: Total training timesteps 
- `--batch_size`: Batch size for actor and critic training 
- `--hidden_dim`: Hidden layer dimension 

### Network Architecture Parameters
- `--discount`: Discount factor for future rewards 
- `--tau`: Target network update rate 
- `--policy_noise`: Standard deviation of target policy noise 
- `--noise_clip`: Range to clip target policy noise 
- `--policy_freq`: Frequency of delayed policy updates 

### Sparsity Control Parameters
- `--static_actor`: Fix actor network topology during training
- `--static_critic`: Fix critic network topology during training
- `--actor_sparsity`: Final sparsity target for actor 
- `--critic_sparsity`: Final sparsity target for critic 
- `--initial_stl_sparsity`: Initial sparsity for STL mode 
- `--delta`: Mask update interval 
- `--zeta`: Initial mask update ratio 
- `--init_method`: Weight initialization method 

### STL Training Parameters
- `--stl_actor`: Enable small-to-large training for actor
- `--stl_critic`: Enable small-to-large training for critic
- `--random_grow`: Use random connection growth scheme
- `--uni`: Use uniform growth scheme
- `--grad_accumulation_n`: Gradient accumulation steps 
- `--use_simple_metric`: Use fast dormant weight pruning

### Buffer Management Parameters (Expierence review)
- `--use_dynamic_buffer`: Enable dynamic replay buffer sizing
- `--buffer_max_size`: Maximum buffer capacity 
- `--buffer_min_size`: Minimum buffer capacity 
- `--buffer_threshold`: Policy distance threshold 
- `--buffer_adjustment_interval`: Buffer check interval 

## Results and Monitoring

The training process saves:
- Model checkpoints
- STL scheduler states
- TensorBoard logs including:
  - Reward curves
  - Network sparsity
  - Buffer statistics
  - FAU (Fraction of Active Units)

Results are saved in:
```
results/
└── {exp_id}_{env}/
    └── {seed}/
        ├── model/
        │   ├── actor
        │   ├── critic
        │   ├── actor_pruner
        │   └── critic_pruner
        └── tensorboard/
```

## Requirements

- Python 3.8.12
- PyTorch 1.13.1+cu117
- TensorBoard 2.9.0
- Gym 0.25.2
- NumPy 1.24.2
- SciPy 1.10.1

### Upcoming JAX Version (locate on Vision RL Branch)
- JAX
- Flax
- Optax
- dm_env
- OpenCV
- MediaPy (for video processing)
- Gymnax (JAX-native environments)

We are developing a JAX-based implementation of Neuroplastic Expansion, specifically optimized for Vision-based RL tasks. This implementation will offer:

- **Accelerated Training**: Leveraging JAX's just-in-time compilation and automatic vectorization
- **Efficient Vision Processing**: Optimized handling of high-dimensional visual inputs
- **Enhanced Parallelization**: Better utilization of TPUs/GPUs for vision-heavy workloads
- **Memory Efficiency**: Improved memory management for large visual observation spaces
- **Seamless Integration**: Easy integration with existing JAX-based RL frameworks

Stay tuned for updates!

**Using conda**
conda env create -f env_export/conda_environment.yml

## Citation

If you use this code in your research, please cite related paper:

```bibtex
@article{liu2024neuroplastic,
  title={Neuroplastic Expansion in Deep Reinforcement Learning},
  author={Liu, Jiashun and Obando-Ceron, Johan and Courville, Aaron and Pan, Ling},
  journal={arXiv preprint arXiv:2410.07994},
  year={2024}
}
```

## License

MIT License
