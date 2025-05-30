# Spring 2025, 535507 Deep Learning
# Lab5: Task 3 - Enhanced DQN with Double DQN, PER, and Multi-Step Return
# Enhanced implementation based on original dqn.py framework

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
import psutil
import GPUtil
from collections import deque, namedtuple
import wandb
import argparse
import time
from typing import Tuple, List, Optional, Dict, Any
import json
from datetime import datetime


# 導入我們的配置系統
from src.config import Config, load_config_from_args

gym.register_envs(ale_py)

# 經驗元組定義
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        # 已在 AtariPreprocessor 中實作
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = gym.wrappers.ResizeObservation(env, (84, 84))
        # env = gym.wrappers.GrayScaleObservation(env)
        # env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk

def init_weights(m):
    """改進的權重初始化"""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))
        # factorized gaussian noise
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)

class DQN(nn.Module):
    """
    Enhanced DQN with Dueling architecture and optional improvements
    """
    def __init__(self, num_actions, n_atoms, v_min, v_max, hidden_dim=512):
        super(DQN, self).__init__()
        self.num_actions = num_actions
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)
        self.register_buffer("support", torch.linspace(self.v_min, self.v_max, self.n_atoms))
        self.hidden_dim = hidden_dim
        
        ########## YOUR CODE HERE ##########
        # Enhanced DQN architecture with Dueling network
        
        # Convolutional layers (following DeepMind DQN architecture)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),  # (84-8)/4+1 = 20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), # (20-4)/2+1 = 9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), # (9-3)/1+1 = 7
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate the size of flattened features
        # For 84x84 input: (84-8)/4+1=20, (20-4)/2+1=9, (9-3)/1+1=7
        # Feature size: 64 * 7 * 7 = 3136 (matches Enhanced-DQN)
        self.feature_size = 64 * 7 * 7
        
        self.value_head = nn.Sequential(
            NoisyLinear(self.feature_size, self.hidden_dim),
            nn.ReLU(),
            NoisyLinear(self.hidden_dim, self.n_atoms)
        )
        
        # Advantage stream  
        self.advantage_head = nn.Sequential(
            NoisyLinear(self.feature_size, self.hidden_dim),
            nn.ReLU(),
            NoisyLinear(self.hidden_dim, self.n_atoms * self.num_actions)
        )
        
        ########## END OF YOUR CODE ##########

    def forward(self, x):
        ########## YOUR CODE HERE ##########
        # Normalize pixel values and forward pass
        h = self.conv_layers(x / 255.0)
        value = self.value_head(h).view(-1, 1, self.n_atoms)
        advantage = self.advantage_head(h).view(-1, self.num_actions, self.n_atoms)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        q_dist = F.softmax(q_atoms, dim=2)
        return q_dist
        ########## END OF YOUR CODE ##########
    
    def reset_noise(self):
        """Reset noise in all NoisyLinear layers"""
        for layer in self.value_head: ## CleanRL
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        for layer in self.advantage_head:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()


class AtariPreprocessor:
    """
    Enhanced preprocessing for Atari environments
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        """Enhanced preprocessing with better normalization"""
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


# adapted from: https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
class SumSegmentTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree_size = 2 * capacity - 1
        self.tree = np.zeros(self.tree_size, dtype=np.float32)

    def _propagate(self, idx):
        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] = self.tree[parent * 2 + 1] + self.tree[parent * 2 + 2]
            parent = (parent - 1) // 2

    def update(self, idx, value):
        tree_idx = idx + self.capacity - 1
        self.tree[tree_idx] = value
        self._propagate(tree_idx)

    def total(self):
        return self.tree[0]

    def retrieve(self, value):
        idx = 0
        while idx * 2 + 1 < self.tree_size:
            left = idx * 2 + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        return idx - (self.capacity - 1)


# adapted from: https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
class MinSegmentTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree_size = 2 * capacity - 1
        self.tree = np.full(self.tree_size, float("inf"), dtype=np.float32)

    def _propagate(self, idx):
        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] = min(self.tree[parent * 2 + 1], self.tree[parent * 2 + 2])
            parent = (parent - 1) // 2

    def update(self, idx, value):
        tree_idx = idx + self.capacity - 1
        self.tree[tree_idx] = value
        self._propagate(tree_idx)

    def min(self):
        return self.tree[0]

PrioritizedBatch = namedtuple(
    "PrioritizedBatch", ["observations", "actions", "rewards", "next_observations", "dones", "indices", "weights"]
)

class PrioritizedReplayBuffer:
    """
    Enhanced Prioritized Experience Replay with Multi-Step Return support
    """
    def __init__(self, capacity, obs_shape, device, alpha=0.6, beta=0.4, n_step=3, gamma=0.99, eps=1e-6):
        ########## YOUR CODE HERE (for Task 3) ##########
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.n_step = n_step
        self.gamma = gamma
        self.eps = eps
        
        # Multi-step buffer - store transitions as tuples
        self.buffer_obs = np.zeros((capacity,) + obs_shape, dtype=np.uint8)
        self.buffer_next_obs = np.zeros((capacity,) + obs_shape, dtype=np.uint8)
        self.buffer_actions = np.zeros(capacity, dtype=np.int64)
        self.buffer_rewards = np.zeros(capacity, dtype=np.float32)
        self.buffer_dones = np.zeros(capacity, dtype=np.bool_)
        
        # Statistics
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

        self.sum_tree = SumSegmentTree(capacity)
        self.min_tree = MinSegmentTree(capacity)
        
        # For n-step returns
        self.n_step_buffer = deque(maxlen=n_step)
        ########## END OF YOUR CODE (for Task 3) ##########

    def _get_n_step_info(self):
        reward = 0.0
        next_obs = self.n_step_buffer[-1][3]
        done = self.n_step_buffer[-1][4]

        for i in range(len(self.n_step_buffer)):
            n_step_reward = self.n_step_buffer[i][2]
            reward += (self.gamma**i) * n_step_reward
            if self.n_step_buffer[i][4]:
                next_obs = self.n_step_buffer[i][3]
                done = True
                break
        return reward, next_obs, done
    
    def add(self, obs, action, reward, next_obs, done):
        ########## YOUR CODE HERE (for Task 3) ##########
        self.n_step_buffer.append((obs, action, reward, next_obs, done))

        if len(self.n_step_buffer) < self.n_step:
            return

        reward, next_obs, done = self._get_n_step_info()
        obs = self.n_step_buffer[0][0]
        action = self.n_step_buffer[0][1]

        idx = self.pos
        self.buffer_obs[idx] = obs
        self.buffer_next_obs[idx] = next_obs
        self.buffer_actions[idx] = action
        self.buffer_rewards[idx] = reward
        self.buffer_dones[idx] = done

        priority = self.max_priority**self.alpha
        self.sum_tree.update(idx, priority)
        self.min_tree.update(idx, priority)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        if done:
            self.n_step_buffer.clear()
        ########## END OF YOUR CODE (for Task 3) ##########

    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ##########
        indices = []
        p_total = self.sum_tree.total()
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = np.random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        samples = {
            "observations": torch.from_numpy(self.buffer_obs[indices]).to(self.device),
            "actions": torch.from_numpy(self.buffer_actions[indices]).to(self.device).unsqueeze(1),
            "rewards": torch.from_numpy(self.buffer_rewards[indices]).to(self.device).unsqueeze(1),
            "next_observations": torch.from_numpy(self.buffer_next_obs[indices]).to(self.device),
            "dones": torch.from_numpy(self.buffer_dones[indices]).to(self.device).unsqueeze(1),
        }

        probs = np.array([self.sum_tree.tree[idx + self.capacity - 1] for idx in indices])
        weights = (self.size * probs / p_total) ** -self.beta
        weights = weights / weights.max()
        samples["weights"] = torch.from_numpy(weights).to(self.device).unsqueeze(1)
        samples["indices"] = indices

        return PrioritizedBatch(**samples)
        ########## END OF YOUR CODE (for Task 3) ##########

    def update_priorities(self, indices, priorities):
        ########## YOUR CODE HERE (for Task 3) ##########
        priorities = np.abs(priorities) + self.eps
        self.max_priority = max(self.max_priority, priorities.max())

        for idx, priority in zip(indices, priorities):
            priority = priority**self.alpha
            self.sum_tree.update(idx, priority)
            self.min_tree.update(idx, priority)
        ########## END OF YOUR CODE (for Task 3) ##########


class EnhancedDQNAgent:
    def __init__(self, config: Config):
        self.config = config
        self.exp_name = f"{config.get('experiment.name', 'Rainbow-Agent')}_{os.path.basename(__file__)[: -len(".py")]}_{self.config.timestamp}"
        self.device = torch.device(config.get('device', 'cuda:0'))
        self.seed = config.get('seed', 42)
        print(f"Using device: {self.device}")

        # Environment setup
        env_name = config.get('environment.name', 'ALE/Pong-v5')
        self.num_envs = config.get('experiment.num_envs', 1)
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        # self.env = gym.vector.SyncVectorEnv(
        #     [make_env(env_name, self.seed + i, i, config.get('capture_video', False), self.exp_name) for i in range(self.num_envs)]
        # )        
        # self.num_actions = self.env.single_action_space.n
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.preprocessor = AtariPreprocessor(config.get('environment.frame_stack', 4))

        # C51
        self.n_atoms = config.get('model.n_atoms', 51)
        self.v_min = config.get('model.v_min', -10.0)
        self.v_max = config.get('model.v_max', 10.0)

        # Network setup
        self.q_net = DQN(self.num_actions, self.n_atoms, self.v_min, self.v_max).to(self.device)
        self.target_net = DQN(self.num_actions, self.n_atoms, self.v_min, self.v_max).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net.load_state_dict(self.q_net.state_dict())

        # Enable model compilation if requested (PyTorch 2.0+)
        if config.get('hardware.compile_model', False):
            print("🚀 Compiling model for faster training...")
            self.q_net = torch.compile(self.q_net, mode="max-autotune")
            self.target_net = torch.compile(self.target_net, mode="max-autotune")

        # Optimizer setup
        self.optimizer = optim.Adam(
            self.q_net.parameters(), 
            lr=float(config.get('training.lr', 2.5e-4)),
            eps=float(config.get('training.optimizer.eps', 1.5e-4)),
            weight_decay=float(config.get('training.optimizer.weight_decay', 0))
        )

        # Training parameters
        self.batch_size = config.get('training.batch_size', 32)
        self.gamma = config.get('training.gamma', 0.99)
        self.n_step = config.get('training.n_step', 3)

        # PER setup
        self.buffer_size = config.get('per.buffer_size', 1000000)
        self.per_replay_alpha = config.get('per.alpha', 0.5)
        self.per_replay_beta = config.get('per.beta', 0.4)
        self.per_replay_eps = config.get('per.eps', 1e-6)
        self.memory = PrioritizedReplayBuffer(
            capacity=self.buffer_size,
            # obs_shape=self.env.single_observation_space.shape,
            obs_shape=(4, 84, 84),  # Atari preprocessed shape
            device=self.device,
            alpha=self.per_replay_alpha,
            beta=self.per_replay_beta,
            eps=self.per_replay_eps,
            n_step=self.n_step,
            gamma=self.gamma,
        )
        print("✅ Using Prioritized Experience Replay")

        # Training state
        self.env_count = 0
        self.train_count = 0
        self.episode_count = 0
        self.best_score = -21.0  # Pong starts at -21
        self.recent_scores = deque(maxlen=100)
        
        # Training parameters
        self.max_episode_steps = config.get('environment.max_episode_steps', 10000)
        # self.replay_start_size = max(self.buffer_size // 10, 10000)  # Start training when 1/8 full, min 10k
        self.learning_starts = config.get('training.learning_starts', 10000) # 引用cleanRL的方法，取代原本replay_start_size
        self.target_update_frequency = config.get('training.target_update_freq', 8000)
        self.train_per_step = config.get('training.train_frequency', 4)
        
        # Evaluation parameters
        self.eval_frequency = config.get('evaluation.eval_frequency', 20000)
        self.eval_episodes = config.get('evaluation.eval_episodes', 20)

        # Logging parameters
        self.log_frequency = config.get('logging.log_frequency', 1000)
        self.checkpoint_frequency = self.config.get('logging.checkpoint_frequency', 10000)

        # # Setup mixed precision training
        # self.use_mixed_precision = config.get('hardware.mixed_precision', False)
        # if self.use_mixed_precision:
        #     self.scaler = torch.amp.GradScaler('cuda')
        #     print("✅ Mixed precision training enabled")

        # Setup W&B logging
        if config.get('logging.wandb_enabled', True):
            self._setup_wandb()

    def _setup_wandb(self):
        """Initialize Weights & Biases logging"""
        wandb.init(
            project=self.config.get('logging.wandb_project', 'DLP-Lab5-Task3-Enhanced-DQN'),
            group=self.config.get('logging.wandb_group', 'task3_experiments'),
            name=f"{self.exp_name}",
            tags=self.config.get('experiment.tags', []),
            config=self.config.data,
            save_code=True,
            dir=self.config.exp_dir
        )
        print(f"✅ W&B initialized: {wandb.run.name}")

    def _log_metrics(self, step: int, episode_reward: float, eval_scores: List[float] = None, training_metrics: Dict = None, fps: float = None):
        """Log comprehensive metrics to W&B"""
        if not self.config.get('logging.wandb_enabled', True):
            return
            
        log_data = {
            'step': step,
            'episode': self.episode_count,
            'training/learning_rate': self.optimizer.param_groups[0]['lr'],
            'algorithm/per_beta': self.memory.beta,
            'system/replay_buffer_size': self.memory.size,
        }
        
        # Episode metrics
        if episode_reward is not None:
            log_data['training/episode_reward'] = episode_reward
            
        # Evaluation metrics
        if eval_scores is not None:
            log_data.update({
                'performance/eval_score_mean': np.mean(eval_scores),
                'performance/eval_score_std': np.std(eval_scores),
                'performance/best_score': self.best_score,
                'performance/sample_efficiency': step / max(1, abs(self.best_score) + 21),
            })
            
        # Training metrics
        if training_metrics is not None:
            log_data.update({
                'training/loss': training_metrics['loss'],
                'training/q_value_mean': training_metrics['q_values'].mean().item(),
                'training/q_value_std': training_metrics['q_values'].std().item(),
                'training/per_loss_mean': training_metrics['td_errors'].mean().item(),
                'algorithm/priority_mean': training_metrics['priorities'].mean().item(),
                'algorithm/importance_weight_mean': training_metrics['priorities'].mean().item(),
                'algorithm/target_q_mean': training_metrics['target_q_mean'],
            })

        # FPS metrics
        if fps is not None:
            log_data['system/fps'] = fps
            
        # System metrics
        try:
            if torch.cuda.is_available():
                gpu_util = GPUtil.getGPUs()[0].load * 100
                memory_usage = psutil.virtual_memory().percent
                log_data.update({
                    'system/gpu_util': gpu_util,
                    'system/memory_usage': memory_usage,
                })
        except:
            pass
            
        wandb.log(log_data, step=step)

    def save_checkpoint(self, step: int, eval_scores: List[float], checkpoint_type: str = 'regular'):
        """Enhanced checkpoint saving"""
        # Prepare checkpoint data
        checkpoint = {
            # Model states
            'model_state_dict': self.q_net.state_dict(),
            'target_model_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            
            # Training state
            'step': step,
            'episode': self.episode_count,
            'env_count': self.env_count,
            'train_count': self.train_count,
            'per_beta': self.memory.beta,
            
            # Performance metrics
            'eval_scores': eval_scores,
            'best_score': self.best_score,
            'avg_score': np.mean(eval_scores) if eval_scores else 0,
            'recent_scores': list(self.recent_scores),

            # Rainbow特有參數
            'n_atoms': self.n_atoms,
            'v_min': self.v_min,
            'v_max': self.v_max,
            
            # Config and metadata
            'config': self.config.data,
            'timestamp': datetime.now().isoformat(),
            'checkpoint_type': checkpoint_type,
        }
        
        # Save different types of checkpoints
        if checkpoint_type == 'milestone':
            # Task requirement: specific step checkpoints
            path = self.config.get_checkpoint_path(step)
        elif checkpoint_type == 'best':
            path = self.config.get_best_model_path()
        else:
            path = self.config.get_latest_checkpoint_path()
            
        torch.save(checkpoint, path)
        print(f"💾 Saved {checkpoint_type} checkpoint: {os.path.basename(path)}")
        
        return path

    def train(self):
        """Enhanced training step with Double DQN, PER, and Multi-Step Return"""
        ########## YOUR CODE HERE ##########
        # if len(self.memory) < self.replay_start_size:
        #     return None
        # reset the noise for both networks
        self.q_net.reset_noise()
        self.target_net.reset_noise()
        data = self.memory.sample(self.batch_size)
        
        self.train_count += 1
        ########## END OF YOUR CODE ##########

        # # Convert to tensors
        # states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        # next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        # actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        # rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        # dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        ########## YOUR CODE HERE ##########
        # Forward pass
        with torch.no_grad():
            next_dist = self.target_net(data.next_observations)
            support = self.target_net.support
            next_q_values = torch.sum(next_dist * support, dim=2)  # [B, num_actions]

            # double q-learning
            next_dist_online = self.q_net(data.next_observations)  # [B, num_actions, n_atoms]
            next_q_online = torch.sum(next_dist_online * support, dim=2)  # [B, num_actions]
            best_actions = torch.argmax(next_q_online, dim=1)  # [B]
            next_pmfs = next_dist[torch.arange(self.batch_size), best_actions]  # [B, n_atoms]

            # compute the n-step Bellman update.
            gamma_n = self.gamma**self.n_step
            next_atoms = data.rewards + gamma_n * support * (1 - data.dones.float())
            tz = next_atoms.clamp(self.q_net.v_min, self.target_net.v_max)

            # projection
            delta_z = self.q_net.delta_z
            b = (tz - self.q_net.v_min) / delta_z  # shape: [B, n_atoms]
            l = b.floor().clamp(0, self.n_atoms - 1)
            u = b.ceil().clamp(0, self.n_atoms - 1)

            # (l == u).float() handles the case where bj is exactly an integer
            # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
            d_m_l = (u.float() + (l == b).float() - b) * next_pmfs  # [B, n_atoms]
            d_m_u = (b - l) * next_pmfs  # [B, n_atoms]

            target_pmfs = torch.zeros_like(next_pmfs)
            for i in range(target_pmfs.size(0)):
                target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])
        
        # Current distribution
        dist = self.q_net(data.observations)  # [B, num_actions, n_atoms]
        pred_dist = dist.gather(1, data.actions.unsqueeze(-1).expand(-1, -1, self.n_atoms)).squeeze(1)
        log_pred = torch.log(pred_dist.clamp(min=1e-5, max=1 - 1e-5))

        # Cross-entropy loss per sample
        loss_per_sample = -(target_pmfs * log_pred).sum(dim=1)
        loss = (loss_per_sample * data.weights.squeeze()).mean()

        # Update priorities in PER
        new_priorities = loss_per_sample.detach().cpu().numpy()
        self.memory.update_priorities(data.indices, new_priorities)

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        ########## END OF YOUR CODE ##########

        # Update target network
        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Calculate Q-values for logging
        q_values = (pred_dist * self.q_net.support).sum(dim=1)

        # Return training metrics
        return {
            'loss': loss.item(),
            'q_values': q_values.detach(),
            'td_errors': loss_per_sample.detach(),
            'priorities': data.weights.detach(),
            'target_q_mean': q_values.mean().item(),
        }

    def evaluate(self) -> float:
        """Comprehensive evaluation"""
        eval_scores = []
        
        for _ in range(self.eval_episodes):
            obs, _ = self.test_env.reset()
            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            
            while not done:
                action = self.select_action(state, eval_mode=True)
                next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = self.preprocessor.step(next_obs)
            
            eval_scores.append(total_reward)
        
        return eval_scores
    
    def select_action(self, state, eval_mode=False):
        """Enhanced action selection with evaluation mode"""        
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_dist = self.q_net(state_tensor)
            q_values = torch.sum(q_dist * self.q_net.support, dim=2)
        return q_values.argmax().item() ## CleanRL中有轉到cpu

    def run(self, total_steps: int = None):
        """Main training loop"""
        if total_steps is None:
            total_steps = self.config.get('training.total_steps', 1000000)
            
        print(f"🚀 Starting enhanced DQN training for {total_steps} steps...")
        print(f"📁 Experiment directory: {self.config.exp_dir}")
        
        milestone_steps = self.config.get('logging.milestone_steps', [200000, 400000, 600000, 800000, 1000000])
        target_score = self.config.get('early_stopping.target_score', 19.0)
        
        start_time = time.time()

        ## start the game
        while self.env_count < total_steps:
            # anneal PER beta to 1
            self.memory.beta = min(
                1.0, self.per_replay_beta + self.env_count * (1.0 - self.per_replay_beta) / total_steps
            )

            obs, _ = self.env.reset()
            state = self.preprocessor.reset(obs)
            done = False
            episode_reward = 0
            step_count = 0

            self.memory.n_step_buffer.clear()

            while not done and step_count < self.max_episode_steps and self.env_count < total_steps:
                # Select and execute action
                action = self.select_action(state)

                # Environment step
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = self.preprocessor.step(next_obs)
                
                # Store experience in buffer
                self.memory.add(state, action, reward, next_state, done)

                # Training step
                if self.env_count > self.learning_starts:
                    if self.env_count % self.train_per_step == 0:
                        training_metrics = self.train()
                    else:
                        training_metrics = None
                else:
                    training_metrics = None

                state = next_state
                episode_reward += reward
                self.env_count += 1
                step_count += 1

                # Logging
                if self.env_count % self.log_frequency == 0:
                    elapsed_time = time.time() - start_time
                    fps = self.env_count / elapsed_time
                    print(f"[Step {self.env_count:>7}] Ep: {self.episode_count:>4} | "
                          f"Reward: {episode_reward:>6.1f} | Beta: {self.memory.beta:.3f} | "
                          f"FPS: {fps:.1f} | Buffer: {self.memory.size:>6}")
                    
                    if training_metrics:
                        self._log_metrics(self.env_count, episode_reward, training_metrics=training_metrics, fps=fps)

                # Evaluation
                if self.env_count % self.eval_frequency == 0:
                    eval_scores = self.evaluate()
                    avg_score = np.mean(eval_scores)
                    
                    print(f"🎯 [Eval] Step {self.env_count} | Avg Score: {avg_score:.2f} ± {np.std(eval_scores):.2f}")
                    
                    # Update best score and save best model
                    if avg_score > self.best_score:
                        self.best_score = avg_score
                        self.save_checkpoint(self.env_count, eval_scores, 'best')
                        print(f"🏆 New best score: {self.best_score:.2f}")
                    
                    # Log evaluation metrics
                    self._log_metrics(self.env_count, episode_reward, eval_scores)
                    
                    # Check early stopping
                    if avg_score >= target_score:
                        print(f"🎉 Target score {target_score} reached! Early stopping...")
                        self.save_checkpoint(self.env_count, eval_scores, 'final')
                        return self.best_score

                # Save milestone checkpoints
                if self.env_count in milestone_steps:
                    eval_scores = self.evaluate()
                    self.save_checkpoint(self.env_count, eval_scores, 'milestone')

                # Regular checkpoint saving
                if self.env_count % self.checkpoint_frequency == 0:
                    self.save_checkpoint(self.env_count, [], 'regular')

            # End of episode
            self.episode_count += 1
            self.recent_scores.append(episode_reward)
            
        print(f"✅ Training completed! Best score: {self.best_score:.2f}")
        return self.best_score

if __name__ == "__main__":
    # Load configuration
    config, args = load_config_from_args()
    
    # Set random seeds for reproducibility
    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    print(f"🎲 Random seed set to: {seed}")
    
    # Create and run agent
    agent = DQNAgent(config)
    
    try:
        best_score = agent.run()
        print(f"🏁 Training finished with best score: {best_score:.2f}")
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        # Save emergency checkpoint
        agent.save_checkpoint(agent.env_count, [], 'emergency')
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        raise
    finally:
        if wandb.run:
            wandb.finish()