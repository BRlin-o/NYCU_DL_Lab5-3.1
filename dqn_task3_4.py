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
    """NoisyLinear layer for exploration - 關鍵缺失組件！"""
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        
        # Noise buffers
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x):
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                          self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)

class DQN(nn.Module):
    """
    Enhanced DQN with Dueling architecture and optional improvements
    """
    def __init__(self, num_actions, config):
        super(DQN, self).__init__()
        self.num_actions = num_actions
        self.dueling = config.get('model.dueling', True)
        self.hidden_dim = config.get('model.hidden_dim', 512)
        
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
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 額外層
            nn.ReLU()
        )
        
        # Calculate the size of flattened features
        # For 84x84 input: (84-8)/4+1=20, (20-4)/2+1=9, (9-3)/1+1=7
        # Feature size: 64 * 7 * 7 = 3136 (matches Enhanced-DQN)
        self.feature_size = 64 * 7 * 7
        
        if self.dueling:
            # Dueling DQN architecture
            # Value stream
            self.value_stream = nn.Sequential(
                nn.Linear(self.feature_size, self.hidden_dim),
                nn.ReLU(),
                NoisyLinear(self.hidden_dim, 1)
            )
            
            # Advantage stream  
            self.advantage_stream = nn.Sequential(
                nn.Linear(self.feature_size, self.hidden_dim),
                nn.ReLU(),
                NoisyLinear(self.hidden_dim, num_actions)
            )
        else:
            # Standard DQN
            self.network = nn.Sequential(
                nn.Linear(self.feature_size, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, num_actions)
            )
        
        ########## END OF YOUR CODE ##########

    def forward(self, x):
        ########## YOUR CODE HERE ##########
        # Normalize pixel values and forward pass
        x = x / 255.0
        
        # Convolutional features
        conv_out = self.conv_layers(x)
        features = conv_out.view(conv_out.size(0), -1)
        
        if self.dueling:
            # Dueling DQN forward pass
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            
            # Combine value and advantage using dueling formula
            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
            return q_values
        else:
            return self.network(features)
        ########## END OF YOUR CODE ##########
    
    def reset_noise(self):
        """Reset noise in all NoisyLinear layers"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


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


class SumTree:
    """Efficient Sum Tree implementation for Prioritized Experience Replay"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = np.full(capacity, None, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        if dataIdx < 0 or dataIdx >= self.capacity:
            return (idx, 0.0, None)
        if self.data[dataIdx] is None:
            return (idx, 0.0, None)
        return (idx, self.tree[idx], self.data[dataIdx])
    
    def __len__(self):
        return self.n_entries


class PrioritizedReplayBuffer:
    """
    Enhanced Prioritized Experience Replay with Multi-Step Return support
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, n_step=3, gamma=0.99, config=None):
        ########## YOUR CODE HERE (for Task 3) ##########
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.n_step = n_step
        self.gamma = gamma
        self.min_priority = float(config.get('per.min_priority', 1e-6) if config else 1e-6)
        
        # Multi-step buffer - store transitions as tuples
        self.n_step_buffer = deque(maxlen=n_step)
        
        # Statistics
        self.max_priority = 1.0
        self.eps = float(1e-8)  # Small epsilon for numerical stability
        ########## END OF YOUR CODE (for Task 3) ##########

    def add(self, transition, error=None):
        ########## YOUR CODE HERE (for Task 3) ##########
        # Convert Experience to tuple for consistent handling
        if hasattr(transition, 'state'):
            # It's an Experience namedtuple
            transition_tuple = (transition.state, transition.action, transition.reward, 
                              transition.next_state, transition.done)
        else:
            # It's already a tuple
            transition_tuple = transition
            
        # Add transition to n-step buffer
        self.n_step_buffer.append(transition_tuple)
        
        # Only add to main buffer when we have n steps
        if len(self.n_step_buffer) < self.n_step:
            return
            
        # Calculate n-step return
        n_step_transition = self._get_n_step_info()
        
        # Calculate priority
        if error is not None:
            priority = (abs(error) + self.min_priority) ** self.alpha
        else:
            priority = self.max_priority
            
        # Update max priority
        self.max_priority = max(self.max_priority, priority)
        
        # Add to tree
        self.tree.add(priority, n_step_transition)
        ########## END OF YOUR CODE (for Task 3) ##########

    def _get_n_step_info(self):
        """Calculate n-step return"""
        ########## YOUR CODE HERE (for Task 3) ##########
        # Get first transition info
        first_transition = self.n_step_buffer[0]
        state = first_transition[0]
        action = first_transition[1]
        
        # Calculate n-step reward
        n_step_reward = 0.0
        gamma_power = 1.0
        
        # Find actual episode termination point
        done_flag = False
        next_state = None
        
        for i, transition in enumerate(self.n_step_buffer):
            _, _, reward, next_state, done = transition
            n_step_reward += gamma_power * reward
            gamma_power *= self.gamma
            
            if done:
                # Episode terminated before n steps
                done_flag = True
                break
            else:
                # Full n-step return, episode didn't terminate
                next_state = self.n_step_buffer[-1][3]
                done_flag = self.n_step_buffer[-1][4]
        
        return Experience(state, action, n_step_reward, next_state, done_flag)
        ########## END OF YOUR CODE (for Task 3) ##########

    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ##########
        if len(self.tree) == 0:
            return [], [], []
            
        batch = []
        indices = []
        priorities = []
        
        # Sample from tree
        total_priority = self.tree.total()

        if total_priority <= 0:
            return [], [], []
        
        segment = total_priority / batch_size

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            
            # Validate data
            if data is not None:
                batch.append(data)
                indices.append(idx)
                priorities.append(max(priority, self.min_priority))
        
        if len(batch) == 0:
            return [], [], []
        
        # Calculate importance sampling weights with numerical stability
        priorities = np.array(priorities)
        sampling_probabilities = priorities / (total_priority + self.eps)
        
        # Avoid division by zero
        sampling_probabilities = np.maximum(sampling_probabilities, self.eps)
        
        # Calculate IS weights
        weights = np.power(len(self.tree) * sampling_probabilities, -self.beta)
        
        # Normalize weights to prevent gradient explosion
        # max_weight = weights.max()
        # if max_weight > 0:
        #     weights = weights / max_weight
        # else:
        #     weights = np.ones_like(weights)
        weights = weights / weights.max()
        
        return batch, indices, weights
        ########## END OF YOUR CODE (for Task 3) ##########

    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ##########
        for idx, error in zip(indices, errors):
            priority = (abs(error) + self.min_priority) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)
        ########## END OF YOUR CODE (for Task 3) ##########

    def update_beta(self, beta):
        """Update beta parameter for importance sampling"""
        self.beta = beta

    def __len__(self):
        # return self.tree.n_entries
        return len(self.tree)


class EnhancedDQNAgent:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda:0'))
        print(f"Using device: {self.device}")

        # Environment setup
        env_name = config.get('environment.name', 'ALE/Pong-v5')
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor(config.get('environment.frame_stack', 4))

        # Network setup
        self.q_net = DQN(self.num_actions, config).to(self.device)
        self.target_net = DQN(self.num_actions, config).to(self.device)
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
        self.gradient_clip = config.get('training.gradient_clipping', 10.0)

        # Exploration parameters
        self.epsilon = config.get('exploration.epsilon_start', 1.0)
        self.epsilon_final = config.get('exploration.epsilon_final', 0.01)
        self.epsilon_decay_steps = config.get('exploration.epsilon_decay_steps', 250000)
        self.epsilon_eval = config.get('exploration.epsilon_eval', 0.001)

        # PER setup
        buffer_size = config.get('per.buffer_size', 100000)
        per_enabled = config.get('per.enabled', True)
        
        if per_enabled:
            per_alpha = config.get('per.alpha', 0.6)
            per_beta_start = config.get('per.beta_start', 0.4)
            self.per_beta_final = config.get('per.beta_final', 1.0)
            self.per_beta_steps = config.get('per.beta_steps', 1000000)
            
            self.memory = PrioritizedReplayBuffer(
                capacity=buffer_size,
                alpha=per_alpha,
                beta=per_beta_start,
                n_step=self.n_step,
                gamma=self.gamma,
                config=config
            )
            self.use_per = True
            print("✅ Using Prioritized Experience Replay")
        else:
            # Fallback to uniform replay buffer
            self.memory = deque(maxlen=buffer_size)
            self.use_per = False
            print("⚠️ Using uniform replay buffer (PER disabled)")

        # Training state
        self.env_count = 0
        self.train_count = 0
        self.episode_count = 0
        self.best_score = -21.0  # Pong starts at -21
        self.recent_scores = deque(maxlen=100)
        
        # Training parameters
        self.max_episode_steps = config.get('environment.max_episode_steps', 108000) 
        self.replay_start_size = max(buffer_size // 10, 10000)  # Start training when 1/8 full, min 10k
        self.target_update_frequency = config.get('training.target_update_freq', 8000)
        self.train_per_step = config.get('training.train_frequency', 4)
        
        # Evaluation parameters
        self.eval_frequency = config.get('evaluation.eval_frequency', 20000)
        self.eval_episodes = config.get('evaluation.eval_episodes', 20)

        # Setup mixed precision training
        self.use_mixed_precision = config.get('hardware.mixed_precision', False)
        if self.use_mixed_precision:
            self.scaler = torch.amp.GradScaler('cuda')
            print("✅ Mixed precision training enabled")

        # Setup W&B logging
        if config.get('logging.wandb_enabled', True):
            self._setup_wandb()

    def _setup_wandb(self):
        """Initialize Weights & Biases logging"""
        wandb.init(
            project=self.config.get('logging.wandb_project', 'DLP-Lab5-Task3-Enhanced-DQN'),
            group=self.config.get('logging.wandb_group', 'task3_experiments'),
            name=f"{self.config.get('experiment.name')}_{self.config.timestamp}",
            tags=self.config.get('experiment.tags', []),
            config=self.config.data,
            save_code=True,
            dir=self.config.exp_dir
        )
        print(f"✅ W&B initialized: {wandb.run.name}")

    def select_action(self, state, eval_mode=False):
        """Enhanced action selection with evaluation mode"""
        if not eval_mode:
            epsilon = self.epsilon
            self.q_net.reset_noise()
        else:
            epsilon = self.epsilon_eval
        
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.use_mixed_precision:
                with torch.amp.autocast('cuda'):
                    q_values = self.q_net(state_tensor)
            else:
                q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def train(self):
        """Enhanced training step with Double DQN, PER, and Multi-Step Return"""
        if len(self.memory) < self.replay_start_size:
            return None
        
        # Update exploration
        if self.epsilon > self.epsilon_final:
            decay_rate = (self.epsilon - self.epsilon_final) / self.epsilon_decay_steps
            self.epsilon = max(self.epsilon_final, self.epsilon - decay_rate)
        
        # Update PER beta (only if using PER)
        if self.use_per and self.env_count < self.per_beta_steps:
            beta_progress = self.env_count / self.per_beta_steps
            current_beta = self.config.get('per.beta_start', 0.4) + \
                         beta_progress * (self.per_beta_final - self.config.get('per.beta_start', 0.4))
            self.memory.update_beta(current_beta)
        
        self.train_count += 1

        ########## YOUR CODE HERE ##########
        if self.use_per:
            # Sample batch from PER
            experiences, indices, weights = self.memory.sample(self.batch_size)
            
            if len(experiences) == 0:
                return None
            
            # Convert weights to tensor
            weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        else:
            # Uniform sampling from deque
            experiences = random.sample(self.memory, self.batch_size)
            indices = None
            weights = torch.ones(self.batch_size).to(self.device)  # Uniform weights
        
        # Unpack experiences - handle both Experience namedtuples and tuples
        if hasattr(experiences[0], 'state'):
            states = np.stack([exp.state for exp in experiences])
            actions = np.array([exp.action for exp in experiences])
            rewards = np.array([exp.reward for exp in experiences])
            next_states = np.stack([exp.next_state for exp in experiences])
            dones = np.array([exp.done for exp in experiences])
        else:
            states = np.stack([exp[0] for exp in experiences])
            actions = np.array([exp[1] for exp in experiences])
            rewards = np.array([exp[2] for exp in experiences])
            next_states = np.stack([exp[3] for exp in experiences])
            dones = np.array([exp[4] for exp in experiences])
        ########## END OF YOUR CODE ##########

        # Convert to tensors
        states = torch.from_numpy(states).float().to(self.device, non_blocking=True)
        next_states = torch.from_numpy(next_states).float().to(self.device, non_blocking=True)
        actions = torch.from_numpy(actions).long().to(self.device, non_blocking=True)
        rewards = torch.from_numpy(rewards).float().to(self.device, non_blocking=True)
        dones = torch.from_numpy(dones).float().to(self.device, non_blocking=True)

        ########## YOUR CODE HERE ##########
        # Forward pass
        if self.use_mixed_precision:
            with torch.amp.autocast('cuda'):
                current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                # Double DQN: Use main network to select actions, target network to evaluate
                with torch.no_grad():
                    next_actions = self.q_net(next_states).argmax(1)
                    next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    target_q_values = rewards + (self.gamma ** self.n_step) * next_q_values * (1 - dones)
                
                # Calculate TD errors for PER
                td_errors = current_q_values - target_q_values
                
                # # Weighted MSE loss
                # loss = (weights * td_errors.pow(2)).mean()

                # PER需要單獨計算每個樣本的loss
                elementwise_loss = F.smooth_l1_loss(
                    current_q_values, 
                    target_q_values.detach(), 
                    reduction='none'
                )
                loss = (weights * elementwise_loss).mean()
        else:
            current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Double DQN implementation
            with torch.no_grad():
                next_actions = self.q_net(next_states).argmax(1)
                next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target_q_values = rewards + (self.gamma ** self.n_step) * next_q_values * (1 - dones)
            
            # Calculate TD errors for PER
            td_errors = current_q_values - target_q_values
            
            # # Weighted MSE loss
            # loss = (weights * td_errors.pow(2)).mean()

            loss = F.smooth_l1_loss(current_q_values, target_q_values.detach())

        # Backward pass
        self.optimizer.zero_grad()
        
        if self.use_mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.gradient_clip)
            self.optimizer.step()

        # Update priorities in PER (only if using PER)
        if self.use_per and (indices is not None):
            td_errors_np = td_errors.detach().cpu().numpy()
            self.memory.update_priorities(indices, td_errors_np)
        ########## END OF YOUR CODE ##########

        # Update target network
        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        ## TODO: 看要不要加上
        self.q_net.reset_noise()

        # Return training metrics
        return {
            'loss': loss.item(),
            'q_values': current_q_values.detach(),
            'td_errors': td_errors.detach(),
            'priorities': weights.detach(),
            'target_q_mean': target_q_values.mean().item(),
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

    def _log_metrics(self, step: int, episode_reward: float, eval_scores: List[float] = None, training_metrics: Dict = None, fps: float = None):
        """Log comprehensive metrics to W&B"""
        if not self.config.get('logging.wandb_enabled', True):
            return
            
        log_data = {
            'step': step,
            'episode': self.episode_count,
            'exploration/epsilon': self.epsilon,
            'training/learning_rate': self.optimizer.param_groups[0]['lr'],
            'algorithm/per_beta': self.memory.beta if self.use_per else 0,
            'system/replay_buffer_size': len(self.memory),
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
                'training/td_error_mean': training_metrics['td_errors'].mean().item(),
                'algorithm/priority_mean': training_metrics['priorities'].mean().item(),
                'algorithm/importance_weight_mean': training_metrics['priorities'].mean().item(),
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
            'epsilon': self.epsilon,
            'per_beta': self.memory.beta,
            
            # Performance metrics
            'eval_scores': eval_scores,
            'best_score': self.best_score,
            'avg_score': np.mean(eval_scores) if eval_scores else 0,
            'recent_scores': list(self.recent_scores),
            
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

    def run(self, total_steps: int = None):
        """Main training loop"""
        if total_steps is None:
            total_steps = self.config.get('training.total_steps', 1000000)
            
        print(f"🚀 Starting enhanced DQN training for {total_steps} steps...")
        print(f"📁 Experiment directory: {self.config.exp_dir}")
        
        milestone_steps = self.config.get('logging.milestone_steps', [200000, 400000, 600000, 800000, 1000000])
        target_score = self.config.get('early_stopping.target_score', 19.0)
        
        start_time = time.time()
        
        while self.env_count < total_steps:
            # Start episode
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
                clipped_reward = np.clip(reward, -1.0, 1.0)  # Clip rewards to [-1, 1]
                done = terminated or truncated
                next_state = self.preprocessor.step(next_obs)
                
                # Store experience in buffer，存儲時使用 clipped_reward
                experience = (state, action, clipped_reward, next_state, done)
                if self.use_per:
                    self.memory.add(experience)
                else:
                    # Simple deque append for uniform sampling
                    self.memory.append(experience)

                # Training step
                if self.env_count % self.train_per_step == 0:
                    training_metrics = self.train()
                else:
                    training_metrics = None

                state = next_state
                episode_reward += reward
                self.env_count += 1
                step_count += 1

                # Logging
                if self.env_count % self.config.get('logging.log_frequency', 1000) == 0:
                    elapsed_time = time.time() - start_time
                    fps = self.env_count / elapsed_time
                    print(f"[Step {self.env_count:>7}] Ep: {self.episode_count:>4} | "
                          f"Reward: {episode_reward:>6.1f} | Eps: {self.epsilon:.4f} | "
                          f"FPS: {fps:.1f} | Buffer: {len(self.memory):>6}")
                    
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
                if self.env_count % self.config.get('logging.checkpoint_frequency', 10000) == 0:
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
    agent = EnhancedDQNAgent(config)
    
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