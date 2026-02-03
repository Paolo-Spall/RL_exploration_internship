# RL Exploration Project Structure

## Overview
This is a Reinforcement Learning exploration project focused on frontier-based and grid-based exploration environments with multiple observation types.

## Directory Structure

```
RL_exploration/
├── .git/                                    # Git repository
├── .gitignore                              # Git ignore file
├── classes_RL_exploration.png              # Project class diagram
│
├── 📄 Root Scripts (Training & Testing)
│   ├── script.bash                         # Bash execution script
│   ├── script_train.py                     # Main training script
│   ├── script_train_2.py                   # Alternative training script
│   ├── script_frontier_2D_train.py         # Frontier-based 2D training
│   ├── script_environment_demo.py          # Environment demonstration
│   ├── script_test_render.py               # Rendering test script
│   ├── script_render_test_config.py        # Config rendering test
│   └── sript_check.py                      # Verification script (typo in name)
│
├── 📊 Data Files
│   ├── frontiers_grid.joblib              # Frontier grid data
│   ├── frontiers_grid_1.joblib            # Frontier grid variant 1
│   └── frontiers_grid_2.joblib            # Frontier grid variant 2
│
├── 📁 lib/                                 # Core library modules
│   ├── __init__.py
│   ├── utils.py                            # Utility functions
│   ├── train_utils.py                      # Training utilities
│   ├── __pycache__/                        # Python cache
│   │
│   ├── frontier_exploration/               # Frontier-based exploration module
│   │   ├── __init__.py
│   │   ├── frontiers.py                    # Frontier computation logic
│   │   ├── __pycache__/
│   │   │
│   │   ├── environments/                   # Frontier exploration environments
│   │   │   ├── __init__.py
│   │   │   ├── README.md
│   │   │   ├── dynamics.py                 # Environment dynamics
│   │   │   ├── multiobs_front_base.py      # Base multi-observation frontier env
│   │   │   ├── multiobs_frontier_env.py    # Main frontier exploration env
│   │   │   ├── multiobs_frontier_avoidance_env.py  # Frontier env with obstacle avoidance
│   │   │   ├── __pycache__/
│   │   │   └── old/                        # Legacy environment implementations
│   │   │
│   │   ├── models/                         # Pre-trained models & evaluations
│   │   │   ├── *.zip                       # Trained model archives
│   │   │   ├── evaluation_*.txt            # Evaluation results
│   │   │   └── vec_normalize_*.pkl         # Vectorized normalization files
│   │   │
│   │   └── planning/                       # Path planning algorithms
│   │       ├── __init__.py
│   │       ├── RRT.py                      # Rapidly-exploring Random Tree
│   │       ├── RRT_env.py                  # RRT environment integration
│   │       ├── planning_utils.py           # Planning utility functions
│   │       ├── obst_avoidance.py           # Obstacle avoidance algorithms
│   │       └── __pycache__/
│   │
│   ├── free_RL_exploration/                # Free/unrestricted RL exploration module
│   │   ├── __init__.py
│   │   ├── registration_script.py          # Environment registration
│   │   ├── simple_grid_2d_test.py          # 2D grid testing
│   │   ├── simple_grid_2d_train.py         # 2D grid training
│   │   ├── __pycache__/
│   │   │
│   │   ├── environments/                   # Free exploration environments
│   │   │   ├── __init__.py
│   │   │   ├── simple_2D_grid.py           # Basic 2D grid environment
│   │   │   ├── simple_2D_grid_exploration.py     # 2D grid with exploration
│   │   │   ├── simple_2D_observation_grid.py     # 2D grid with observations
│   │   │   ├── simple_2D_multiobs_grid.py        # 2D grid with multi-observations
│   │   │   ├── environment_simple_2D_grid_EXP.py # 2D grid experiment variant
│   │   │   ├── exp_grid_2d_env_multi_in.py       # Multi-input experiment env
│   │   │   └── __pycache__/
│   │   │
│   │   └── models/                         # Pre-trained models for free exploration
│   │       ├── *.zip                       # Trained model archives
│   │       ├── evaluation_*.txt            # Evaluation results
│   │       ├── vec_normalize_*.pkl         # Vectorized normalization files
│   │       ├── config_ex.yaml              # Example config
│   │       ├── old/                        # Legacy models
│   │       └── __pycache__/
│   │
│   └── grid_env/                           # Grid environment utilities
│       ├── __init__.py
│       ├── obst_grid_env.py                # Basic obstacle grid environment
│       ├── obst_grid_gen.py                # Obstacle grid generation
│       ├── obst_grid_agent_env.py          # Agent-based grid environment
│       ├── obst_grid_agent_env_stepping.py # Agent env with step control
│       ├── obst_grid_agent_expl_env.py     # Agent exploration environment
│       ├── stepper_wrapper.py              # Stepper wrapper utility
│       ├── stepper_wrapper_class.py        # Stepper wrapper class
│       └── __pycache__/
│
├── 📁 configs/                             # Configuration files
│   ├── configfile_maker.py                 # Config file generator
│   ├── configfile_maker_2.py               # Alternative config generator
│   │
│   ├── MultiObsFrontierEnv Configs (24 files)
│   │   ├── config_MultiObsFrontierEnv_{absolute|distance|relative}_{agent_}iGain_DQN_1e5.yaml
│   │
│   ├── MultiObsFrontAvoidanceEnv Configs (12 files)
│   │   ├── config_MultiObsFrontAvoidanceEnv_{absolute|distance|relative}_{agent_}iGain_DQN_1e5.yaml
│   │
│   ├── Manual/Test Configs
│   │   ├── config_new.yaml
│   │   ├── config_try_1.yaml
│   │   ├── config_try_2.yaml
│   │   └── manual_config_MultiObsFrontEnv_absolute_agent_DQN_1e5.yaml
│   │
│   ├── old/                                # Legacy configurations
│   └── batch MultiObsFrontEnv(no reset discovered)/  # Batch results
│
├── 📁 models/                              # Training results & checkpoints
│   ├── MultiObsFrontierEnv Models (12 .zip files)
│   │   └── evaluation_MultiObsFrontierEnv_*.txt
│   │
│   ├── MultiObsFrontAvoidanceEnv Models (8 .zip files)
│   │
│   ├── Vectorized Normalizers
│   │   ├── vec_normalize_MultiObsFrontierEnv_*.pkl
│   │   └── vec_normalize_MultiObsFrontAvoidanceEnv_*.pkl
│   │
│   ├── Batch Results
│   │   ├── batch MultiObsFrontEnv 1e5/
│   │   ├── batch MultiObsFrontAvoidanceEnv 1e4/
│   │   └── batch MultiObsFrontEnv(no reset discovered)/
│   │
│   ├── Evaluation Results
│   │   └── evaluation_results_0.txt
│   │
│   ├── Spreadsheets
│   │   └── results_MultiObsFrontierEnv.xlsx
│   │
│   └── old/                                # Legacy models
│
└── 📁 trash/                               # Deprecated/unused files

```

## Key Modules

### Core Libraries (lib/)

#### **frontier_exploration/**
- Implements frontier-based exploration algorithms
- Environments: MultiObsFrontierEnv, MultiObsFrontierAvoidanceEnv
- Includes RRT path planning and obstacle avoidance
- Multiple observation types: absolute, relative, distance

#### **free_RL_exploration/**
- Implements unrestricted RL exploration environments
- Focuses on simple 2D grids with various observation configurations
- Uses PPO and DQN algorithms
- Supports single and multi-observation agents

#### **grid_env/**
- Low-level grid environment utilities
- Handles obstacle generation and agent movement
- Provides stepping wrappers for environment control

### Configuration System (configs/)
- YAML-based configuration files for different environments and agents
- Naming convention: `config_{EnvironmentName}_{ObsType}_{AgentType}_{Algorithm}_{Steps}.yaml`
- Automatic config generation via `configfile_maker.py`

### Training & Models (models/)
- Pre-trained DQN and PPO models in ZIP format
- Evaluation metrics and normalization parameters
- Batch training results organized by environment
- Results spreadsheet for performance comparison

## Environment Types

### Observation Types
- **absolute**: Absolute position observations
- **distance**: Distance-based observations  
- **relative**: Relative/ego-centric observations

### Agent Types
- **agent**: With agent-specific features
- **iGain**: With integral gain components

### Algorithms
- **DQN**: Deep Q-Network (primary)
- **PPO**: Proximal Policy Optimization (experimental)

## Training Scale
- **1e4**: 10,000 timesteps
- **1e5**: 100,000 timesteps (main training)
- **1e6**: 1,000,000 timesteps (extended training)
- **3e6, 5e6**: Large-scale experiments

## File Naming Conventions

### Training Scripts
- `script_[purpose].py` - Main execution scripts
- `configfile_maker[_variant].py` - Config generators

### Models & Results
- `[EnvironmentName]_[Config]_[Steps].zip` - Model archives
- `evaluation_[Environment]_[Config].txt` - Evaluation metrics
- `vec_normalize_[Environment]_[Config].pkl` - Normalization parameters

### Data Files
- `.joblib` - Serialized frontier data
- `.yaml` - Configuration files
- `.pkl` - Pickle-serialized objects
- `.xlsx` - Spreadsheet results
- `.txt` - Text-based evaluation results
