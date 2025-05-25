#!/usr/bin/env python3
"""
Task 3 Enhanced DQN Model Testing Script
Compatible with original test_model.py but adapted for enhanced DQN

Usage:
    python test_model_task3.py --model-path checkpoints/best_model.pt --episodes 20
    python test_model_task3.py --model-path LAB5_StudentID_task3_pong400000.pt --episodes 20 --seed 42
"""

import torch
import torch.nn as nn
import numpy as np
import random
import gymnasium as gym
import cv2
import imageio
import ale_py
import os
import sys
import argparse
from collections import deque
from datetime import datetime

# Add src to path for our modules
sys.path.append('src')

gym.register_envs(ale_py)


class DQN(nn.Module):
    """Enhanced DQN compatible with Task 3 implementation"""
    def __init__(self, num_actions, dueling=True, hidden_dim=512):
        super(DQN, self).__init__()
        self.num_actions = num_actions
        self.dueling = dueling
        self.hidden_dim = hidden_dim
        
        # Convolutional layers (following DeepMind DQN architecture)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )
        
        # Calculate the size of flattened features (for 84x84 input)
        self.feature_size = 64 * 7 * 7
        
        if self.dueling:
            # Dueling DQN architecture
            self.value_stream = nn.Sequential(
                nn.Linear(self.feature_size, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1)
            )
            
            self.advantage_stream = nn.Sequential(
                nn.Linear(self.feature_size, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, num_actions)
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

    def forward(self, x):
        # Normalize pixel values
        x = x / 255.0
        
        # Convolutional features
        conv_out = self.conv_layers(x)
        features = conv_out.view(conv_out.size(0), -1)
        
        if self.dueling:
            # Dueling DQN forward pass
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            
            # Combine value and advantage using dueling formula
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
            return q_values
        else:
            return self.network(features)


class AtariPreprocessor:
    """Enhanced Atari preprocessing"""
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame.copy())
        stacked = np.stack(self.frames, axis=0)
        return stacked


def load_model_with_config_detection(model_path: str, device: torch.device):
    """Load model with automatic config detection from checkpoint"""
    print(f"📂 Loading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract configuration from checkpoint if available
    config = checkpoint.get('config', {})
    
    # Model configuration with fallbacks
    model_config = config.get('model', {})
    dueling = model_config.get('dueling', True)
    hidden_dim = model_config.get('hidden_dim', 512)
    num_actions = 6  # Pong has 6 actions
    
    print(f"   Model config: Dueling={dueling}, Hidden={hidden_dim}")
    
    # Create model
    model = DQN(num_actions, dueling=dueling, hidden_dim=hidden_dim).to(device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Print checkpoint info if available
        if 'step' in checkpoint:
            print(f"   Checkpoint step: {checkpoint['step']:,}")
        if 'best_score' in checkpoint:
            print(f"   Best score: {checkpoint['best_score']:.2f}")
        if 'avg_score' in checkpoint:
            print(f"   Average score: {checkpoint['avg_score']:.2f}")
        if 'timestamp' in checkpoint:
            print(f"   Saved at: {checkpoint['timestamp']}")
    else:
        # Direct state dict (fallback)
        model.load_state_dict(checkpoint)
        print("   Loaded direct state dict (no metadata)")
    
    model.eval()
    return model


def evaluate_model(args):
    """Main evaluation function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")

    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Setup environment
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    preprocessor = AtariPreprocessor()

    # Load model
    model = load_model_with_config_detection(args.model_path, device)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n🎯 Starting evaluation:")
    print(f"   Episodes: {args.episodes}")
    print(f"   Seed: {args.seed}")
    print(f"   Output: {args.output_dir}")
    print(f"   Save videos: {args.save_videos}")

    all_scores = []
    episode_stats = []

    for ep in range(args.episodes):
        print(f"\n[Episode {ep+1:>2}/{args.episodes}] ", end="", flush=True)
        
        # Reset environment
        obs, _ = env.reset(seed=args.seed + ep)
        state = preprocessor.reset(obs)
        done = False
        episode_reward = 0
        frames = []
        step_count = 0
        
        # Episode statistics
        actions_taken = []
        q_values_history = []

        while not done and step_count < args.max_steps:
            # Render frame if saving videos
            if args.save_videos and (ep < args.video_episodes or args.save_all_videos):
                frame = env.render()
                frames.append(frame)

            # Select action
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
                action = q_values.argmax().item()
                
                # Store statistics
                actions_taken.append(action)
                q_values_history.append(q_values.cpu().numpy().flatten())

            # Execute action
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = preprocessor.step(next_obs)
            step_count += 1

        # Episode completed
        all_scores.append(episode_reward)
        episode_stats.append({
            'episode': ep + 1,
            'reward': episode_reward,
            'steps': step_count,
            'actions': actions_taken,
            'q_values': q_values_history
        })

        print(f"Reward: {episode_reward:>6.1f} | Steps: {step_count:>5} | ", end="")
        
        # Save video for this episode
        if args.save_videos and frames and (ep < args.video_episodes or args.save_all_videos):
            video_path = os.path.join(args.output_dir, f"eval_ep{ep+1:02d}.mp4")
            with imageio.get_writer(video_path, fps=30) as video:
                for frame in frames:
                    video.append_data(frame)
            print(f"Video: {os.path.basename(video_path)}")
        else:
            print("No video")

        # Progress update
        if ep > 0:
            current_avg = np.mean(all_scores)
            print(f"   Running average: {current_avg:>6.2f}")

    # Final statistics
    scores = np.array(all_scores)
    final_stats = {
        'model_path': args.model_path,
        'evaluation_date': datetime.now().isoformat(),
        'episodes': args.episodes,
        'seed': args.seed,
        'scores': scores.tolist(),
        'mean_score': float(np.mean(scores)),
        'std_score': float(np.std(scores)),
        'min_score': float(np.min(scores)),
        'max_score': float(np.max(scores)),
        'median_score': float(np.median(scores)),
        'success_rate': float(np.sum(scores >= 19) / len(scores)),
        'positive_rate': float(np.sum(scores > 0) / len(scores)),
        'episode_details': episode_stats
    }

    # Save detailed results
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    import json
    with open(results_path, 'w') as f:
        json.dump(final_stats, f, indent=2)

    # Print final summary
    print("\n" + "="*70)
    print("📊 EVALUATION SUMMARY")
    print("="*70)
    print(f"Model: {os.path.basename(args.model_path)}")
    print(f"Episodes: {args.episodes}")
    print(f"Mean Score: {final_stats['mean_score']:.2f} ± {final_stats['std_score']:.2f}")
    print(f"Score Range: [{final_stats['min_score']:.1f}, {final_stats['max_score']:.1f}]")
    print(f"Median Score: {final_stats['median_score']:.2f}")
    print(f"Success Rate (≥19): {final_stats['success_rate']*100:.1f}%")
    print(f"Positive Rate (>0): {final_stats['positive_rate']*100:.1f}%")
    print(f"Results saved to: {args.output_dir}")

    # Task assessment
    if final_stats['mean_score'] >= 19:
        print(f"✅ TARGET ACHIEVED! Mean score {final_stats['mean_score']:.2f} ≥ 19")
    else:
        print(f"❌ Target not reached. Mean score {final_stats['mean_score']:.2f} < 19")

    print("="*70)

    return final_stats


def main():
    parser = argparse.ArgumentParser(description='Test Enhanced DQN model for Task 3')
    
    # Required arguments
    parser.add_argument("--model-path", type=str, required=True, 
                       help="Path to trained .pt model checkpoint")
    
    # Evaluation settings
    parser.add_argument("--episodes", type=int, default=20, 
                       help="Number of episodes to evaluate")
    parser.add_argument("--seed", type=int, default=313551076, 
                       help="Random seed for evaluation")
    parser.add_argument("--max-steps", type=int, default=108000,
                       help="Maximum steps per episode")
    
    # Output settings
    parser.add_argument("--output-dir", type=str, default="./eval_results",
                       help="Output directory for results and videos")
    parser.add_argument("--save-videos", action="store_true",
                       help="Save evaluation videos")
    parser.add_argument("--video-episodes", type=int, default=3,
                       help="Number of episodes to record (if save-videos enabled)")
    parser.add_argument("--save-all-videos", action="store_true",
                       help="Save videos for all episodes")
    
    # Display settings
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output with detailed statistics")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return 1
    
    if args.episodes <= 0:
        print(f"❌ Invalid number of episodes: {args.episodes}")
        return 1
    
    try:
        # Run evaluation
        results = evaluate_model(args)
        
        # Return appropriate exit code
        if results['mean_score'] >= 19:
            return 0  # Success
        else:
            return 1  # Target not reached
            
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
        return 2
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    exit(main())