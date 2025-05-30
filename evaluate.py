#!/usr/bin/env python3
"""
Enhanced DQN Model Evaluation Script for Task 3

Usage:
    python evaluate.py --model experiments/2025-05-25_14-30-45/checkpoints/best_model.pt
    python evaluate.py --model experiments/*/checkpoints/best_model.pt --episodes 50
    python evaluate.py --experiment experiments/2025-05-25_14-30-45 --all-checkpoints
"""

import os
import sys
import torch
import numpy as np
import argparse
import json
import gymnasium as gym
import ale_py
from datetime import datetime
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append('src')

from dqn_task3_9 import DQN, AtariPreprocessor
from src.config import Config
from src.utils import create_evaluation_video, get_system_info

gym.register_envs(ale_py)

class ModelEvaluator:
    """Comprehensive model evaluation for Task 3"""
    
    def __init__(self, model_path: str, config_path: str = None, device: str = 'auto'):
        self.model_path = model_path
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else device)
        
        # Load config
        if config_path and os.path.exists(config_path):
            self.config = Config(config_path)
        else:
            # Create default config for evaluation
            self.config = self._create_default_config()
        
        # Setup environment
        self.env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
        self.preprocessor = AtariPreprocessor()
        
        # Load model parameters from checkpoint first
        self._load_checkpoint_info()
        
        # Load model
        self.model = self._load_model()
        
        print(f"✅ Model evaluator initialized")
        print(f"   Model: {os.path.basename(model_path)}")
        print(f"   Device: {self.device}")
        print(f"   Environment: Pong-v5")
        print(f"   Model type: Distributional DQN (C51)")
        print(f"   Atoms: {self.n_atoms}, V-range: [{self.v_min}, {self.v_max}]")
    
    def _create_default_config(self):
        """Create default config for evaluation"""
        class DefaultConfig:
            def get(self, key, default=None):
                defaults = {
                    'model.dueling': True,
                    'model.hidden_dim': 512,
                    'model.input_channels': 4,
                    'model.n_atoms': 51,
                    'model.v_min': -10.0,
                    'model.v_max': 10.0,
                    'exploration.epsilon_eval': 0.001,
                }
                return defaults.get(key, default)
        
        return DefaultConfig()
    
    def _load_checkpoint_info(self):
        """Load checkpoint to extract model parameters"""
        print(f"📂 Loading checkpoint info: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        
        # Extract distributional DQN parameters from checkpoint
        if 'n_atoms' in checkpoint:
            self.n_atoms = checkpoint['n_atoms']
            self.v_min = checkpoint['v_min']
            self.v_max = checkpoint['v_max']
            print(f"   📋 Found C51 parameters in checkpoint")
        else:
            # Use config or defaults
            self.n_atoms = self.config.get('model.n_atoms', 51)
            self.v_min = self.config.get('model.v_min', -10.0)
            self.v_max = self.config.get('model.v_max', 10.0)
            print(f"   📋 Using default C51 parameters")
        
        # Print checkpoint info
        if 'step' in checkpoint:
            print(f"   Step: {checkpoint.get('step', 'N/A')}")
        if 'episode' in checkpoint:
            print(f"   Episode: {checkpoint.get('episode', 'N/A')}")
        if 'best_score' in checkpoint:
            print(f"   Best Score: {checkpoint.get('best_score', 'N/A')}")
        if 'timestamp' in checkpoint:
            print(f"   Saved: {checkpoint['timestamp']}")
    
    def _load_model(self) -> torch.nn.Module:
        """Load model from checkpoint with correct parameters"""
        # Create model with extracted parameters
        model = DQN(
            num_actions=self.env.action_space.n,
            n_atoms=self.n_atoms,
            v_min=self.v_min,
            v_max=self.v_max,
            hidden_dim=self.config.get('model.hidden_dim', 512)
        ).to(self.device)
        
        # Load checkpoint
        print(f"🔄 Loading model state...")
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Direct state dict (fallback for older checkpoints)
            model.load_state_dict(checkpoint)
        
        model.eval()
        print(f"✅ Model loaded successfully")
        return model
    
    def evaluate_single_episode(self, render_frames: bool = False, max_steps: int = 108000) -> Tuple[float, List, Dict]:
        """Evaluate a single episode"""
        obs, _ = self.env.reset()
        state = self.preprocessor.reset(obs)
        done = False
        total_reward = 0
        step_count = 0
        frames = []
        
        episode_stats = {
            'actions': [],
            'rewards': [],
            'q_values': [],
            'q_distributions': [],  # For distributional analysis
        }
        
        epsilon = self.config.get('exploration.epsilon_eval', 0.001)
        
        while not done and step_count < max_steps:
            # Render frame if requested
            if render_frames:
                frame = self.env.render()
                frames.append(frame)
            
            # Select action
            if np.random.random() < epsilon:
                action = np.random.randint(0, self.env.action_space.n)
                q_values = None
                q_dist = None
            else:
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    # Get distribution from model
                    q_dist_raw = self.model(state_tensor)  # [1, num_actions, n_atoms]
                    
                    # Convert to Q-values by taking expectation
                    q_values = torch.sum(q_dist_raw * self.model.support, dim=2)  # [1, num_actions]
                    action = q_values.argmax().item()
                    
                    # Store for analysis
                    q_values = q_values.cpu().numpy().flatten()
                    q_dist = q_dist_raw.cpu().numpy()
            
            # Step environment
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = self.preprocessor.step(next_obs)
            step_count += 1
            
            # Record stats
            episode_stats['actions'].append(action)
            episode_stats['rewards'].append(reward)
            if q_values is not None:
                episode_stats['q_values'].append(q_values.copy())
            if q_dist is not None:
                episode_stats['q_distributions'].append(q_dist.copy())
        
        return total_reward, frames, episode_stats
    
    def evaluate_multiple_episodes(self, num_episodes: int = 20, save_videos: bool = False, 
                                 video_episodes: int = 3) -> Dict[str, Any]:
        """Evaluate multiple episodes and return comprehensive stats"""
        print(f"🎯 Evaluating {num_episodes} episodes...")
        
        scores = []
        all_episode_stats = []
        video_frames = []
        
        for episode in range(num_episodes):
            render_frames = save_videos and episode < video_episodes
            score, frames, stats = self.evaluate_single_episode(render_frames=render_frames)
            
            scores.append(score)
            all_episode_stats.append(stats)
            
            if render_frames:
                video_frames.extend(frames)
            
            # Progress update
            if (episode + 1) % 5 == 0 or episode == num_episodes - 1:
                current_avg = np.mean(scores)
                print(f"   Episode {episode + 1:>3}/{num_episodes}: Score = {score:>6.1f}, "
                      f"Running Avg = {current_avg:>6.2f}")
        
        # Calculate comprehensive statistics
        scores = np.array(scores)
        results = {
            'num_episodes': num_episodes,
            'scores': scores.tolist(),
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'median_score': float(np.median(scores)),
            'q25_score': float(np.percentile(scores, 25)),
            'q75_score': float(np.percentile(scores, 75)),
            'success_rate': float(np.sum(scores >= 19) / len(scores)),  # Task target
            'positive_rate': float(np.sum(scores > 0) / len(scores)),
            'evaluation_date': datetime.now().isoformat(),
            'model_path': self.model_path,
            'model_type': 'Rainbow',
            'model_params': {
                'n_atoms': self.n_atoms,
                'v_min': self.v_min,
                'v_max': self.v_max,
            }
        }
        
        # Analyze episode statistics
        if all_episode_stats:
            all_actions = []
            all_q_values = []
            all_q_distributions = []
            
            for stats in all_episode_stats:
                all_actions.extend(stats['actions'])
                if stats['q_values']:
                    all_q_values.extend(stats['q_values'])
                if stats['q_distributions']:
                    all_q_distributions.extend(stats['q_distributions'])
            
            if all_actions:
                action_counts = np.bincount(all_actions, minlength=self.env.action_space.n)
                results['action_distribution'] = action_counts.tolist()
                results['action_entropy'] = float(-np.sum(
                    (action_counts / len(all_actions)) * np.log(action_counts / len(all_actions) + 1e-8)
                ))
            
            if all_q_values:
                all_q_values = np.array(all_q_values)
                results['q_value_stats'] = {
                    'mean': float(np.mean(all_q_values)),
                    'std': float(np.std(all_q_values)),
                    'min': float(np.min(all_q_values)),
                    'max': float(np.max(all_q_values)),
                }
            
            # Distributional analysis
            if all_q_distributions:
                all_q_distributions = np.array(all_q_distributions)
                results['distributional_stats'] = {
                    'mean_distribution_entropy': float(np.mean([
                        -np.sum(dist * np.log(dist + 1e-8)) 
                        for dist in all_q_distributions.reshape(-1, self.n_atoms)
                    ])),
                    'distribution_shape': all_q_distributions.shape,
                }
        
        return results, video_frames
    
    def save_evaluation_report(self, results: Dict[str, Any], output_dir: str):
        """Save comprehensive evaluation report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON report
        report_path = os.path.join(output_dir, 'evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create visualization
        self._create_evaluation_plots(results, output_dir)
        
        # Create text summary
        self._create_text_summary(results, output_dir)
        
        print(f"📊 Evaluation report saved to: {output_dir}")
    
    def _create_evaluation_plots(self, results: Dict[str, Any], output_dir: str):
        """Create evaluation visualization plots"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Distributional DQN Evaluation Report\nMean Score: {results["mean_score"]:.2f} ± {results["std_score"]:.2f}', fontsize=16)
        
        # Score distribution
        axes[0, 0].hist(results['scores'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(results['mean_score'], color='red', linestyle='--', label=f'Mean: {results["mean_score"]:.2f}')
        axes[0, 0].axvline(19, color='green', linestyle='--', label='Target: 19')
        axes[0, 0].set_xlabel('Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Score Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Score over episodes
        axes[0, 1].plot(results['scores'], alpha=0.7, marker='o', markersize=3)
        axes[0, 1].axhline(results['mean_score'], color='red', linestyle='--', alpha=0.7)
        axes[0, 1].axhline(19, color='green', linestyle='--', alpha=0.7)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Score Progression')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Action distribution
        if 'action_distribution' in results:
            action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
            action_counts = results['action_distribution']
            bars = axes[1, 0].bar(range(len(action_counts)), action_counts, alpha=0.7)
            axes[1, 0].set_xlabel('Action')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title(f'Action Distribution (Entropy: {results.get("action_entropy", 0):.3f})')
            axes[1, 0].set_xticks(range(len(action_names)))
            axes[1, 0].set_xticklabels(action_names, rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, count in zip(bars, action_counts):
                if count > 0:
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(action_counts)*0.01,
                                   f'{count}', ha='center', va='bottom', fontsize=8)
        
        # Summary statistics
        axes[1, 1].axis('off')
        stats_text = f"""
Distributional DQN Performance Summary:

Model Type: {results.get('model_type', 'Unknown')}
Episodes Evaluated: {results['num_episodes']}
Mean Score: {results['mean_score']:.2f} ± {results['std_score']:.2f}
Median Score: {results['median_score']:.2f}
Min/Max Score: {results['min_score']:.1f} / {results['max_score']:.1f}
Q25/Q75: {results['q25_score']:.1f} / {results['q75_score']:.1f}

Success Rate (≥19): {results['success_rate']*100:.1f}%
Positive Rate (>0): {results['positive_rate']*100:.1f}%

Task Assessment:
Target Score (19): {'✅ ACHIEVED' if results['mean_score'] >= 19 else '❌ NOT REACHED'}

Model Parameters:
Atoms: {results.get('model_params', {}).get('n_atoms', 'N/A')}
V-range: [{results.get('model_params', {}).get('v_min', 'N/A')}, {results.get('model_params', {}).get('v_max', 'N/A')}]
"""
        
        if 'q_value_stats' in results:
            q_stats = results['q_value_stats']
            stats_text += f"""
Q-Value Statistics:
Mean: {q_stats['mean']:.3f}
Std: {q_stats['std']:.3f}
Range: [{q_stats['min']:.3f}, {q_stats['max']:.3f}]
"""
        
        if 'distributional_stats' in results:
            dist_stats = results['distributional_stats']
            stats_text += f"""
Distributional Stats:
Avg Entropy: {dist_stats['mean_distribution_entropy']:.3f}
"""
        
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'evaluation_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_text_summary(self, results: Dict[str, Any], output_dir: str):
        """Create text summary report"""
        summary_path = os.path.join(output_dir, 'evaluation_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DISTRIBUTIONAL DQN MODEL EVALUATION REPORT - TASK 3\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Model Path: {results['model_path']}\n")
            f.write(f"Model Type: {results.get('model_type', 'Unknown')}\n")
            f.write(f"Evaluation Date: {results['evaluation_date']}\n")
            f.write(f"Episodes Evaluated: {results['num_episodes']}\n\n")
            
            # Model parameters
            if 'model_params' in results:
                f.write("MODEL PARAMETERS:\n")
                f.write("-" * 40 + "\n")
                params = results['model_params']
                f.write(f"Atoms (n_atoms):     {params.get('n_atoms', 'N/A'):>8}\n")
                f.write(f"V-min:               {params.get('v_min', 'N/A'):>8}\n")
                f.write(f"V-max:               {params.get('v_max', 'N/A'):>8}\n\n")
            
            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Mean Score:          {results['mean_score']:>8.2f} ± {results['std_score']:.2f}\n")
            f.write(f"Median Score:        {results['median_score']:>8.2f}\n")
            f.write(f"Min Score:           {results['min_score']:>8.1f}\n")
            f.write(f"Max Score:           {results['max_score']:>8.1f}\n")
            f.write(f"25th Percentile:     {results['q25_score']:>8.1f}\n")
            f.write(f"75th Percentile:     {results['q75_score']:>8.1f}\n\n")
            
            f.write("SUCCESS METRICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Success Rate (≥19):  {results['success_rate']*100:>7.1f}%\n")
            f.write(f"Positive Rate (>0):  {results['positive_rate']*100:>7.1f}%\n\n")
            
            f.write("TASK ASSESSMENT:\n")
            f.write("-" * 40 + "\n")
            target_status = "✅ ACHIEVED" if results['mean_score'] >= 19 else "❌ NOT REACHED"
            f.write(f"Target Score (19):   {target_status}\n")
            
            if results['mean_score'] >= 19:
                f.write(f"Performance Level:   EXCELLENT\n")
            elif results['mean_score'] >= 15:
                f.write(f"Performance Level:   GOOD\n")
            elif results['mean_score'] >= 10:
                f.write(f"Performance Level:   FAIR\n")
            else:
                f.write(f"Performance Level:   NEEDS IMPROVEMENT\n")
            
            if 'action_distribution' in results:
                f.write(f"\nACTION ANALYSIS:\n")
                f.write("-" * 40 + "\n")
                action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
                total_actions = sum(results['action_distribution'])
                for i, (name, count) in enumerate(zip(action_names, results['action_distribution'])):
                    percentage = count / total_actions * 100 if total_actions > 0 else 0
                    f.write(f"{name:<12}: {count:>8} ({percentage:>5.1f}%)\n")
                f.write(f"Action Entropy:      {results.get('action_entropy', 0):>8.3f}\n")
            
            if 'q_value_stats' in results:
                f.write(f"\nQ-VALUE STATISTICS:\n")
                f.write("-" * 40 + "\n")
                q_stats = results['q_value_stats']
                f.write(f"Mean Q-Value:        {q_stats['mean']:>8.3f}\n")
                f.write(f"Q-Value Std:         {q_stats['std']:>8.3f}\n")
                f.write(f"Q-Value Range:       [{q_stats['min']:.3f}, {q_stats['max']:.3f}]\n")
            
            if 'distributional_stats' in results:
                f.write(f"\nDISTRIBUTIONAL ANALYSIS:\n")
                f.write("-" * 40 + "\n")
                dist_stats = results['distributional_stats']
                f.write(f"Avg Distribution Entropy: {dist_stats['mean_distribution_entropy']:>8.3f}\n")

def evaluate_all_checkpoints(exp_dir: str, episodes_per_checkpoint: int = 10) -> Dict[str, Dict]:
    """Evaluate all checkpoints in an experiment directory"""
    checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
    config_path = os.path.join(exp_dir, 'config.json')
    
    if not os.path.exists(checkpoints_dir):
        print(f"❌ Checkpoints directory not found: {checkpoints_dir}")
        return {}
    
    # Find all checkpoint files
    checkpoint_files = []
    for filename in os.listdir(checkpoints_dir):
        if filename.endswith('.pt'):
            checkpoint_files.append(os.path.join(checkpoints_dir, filename))
    
    if not checkpoint_files:
        print(f"❌ No checkpoint files found in: {checkpoints_dir}")
        return {}
    
    # Sort files
    checkpoint_files.sort()
    
    results = {}
    print(f"🔍 Evaluating {len(checkpoint_files)} checkpoints...")
    
    for i, checkpoint_path in enumerate(checkpoint_files):
        filename = os.path.basename(checkpoint_path)
        print(f"\n[{i+1}/{len(checkpoint_files)}] Evaluating: {filename}")
        
        try:
            evaluator = ModelEvaluator(checkpoint_path, config_path)
            eval_results, _ = evaluator.evaluate_multiple_episodes(episodes_per_checkpoint)
            results[filename] = eval_results
            
            print(f"   Mean Score: {eval_results['mean_score']:.2f} ± {eval_results['std_score']:.2f}")
            
        except Exception as e:
            print(f"   ❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate Distributional DQN models for Task 3')
    parser.add_argument('--model', type=str, help='Path to model checkpoint (.pt file)')
    parser.add_argument('--experiment', type=str, help='Path to experiment directory')
    parser.add_argument('--all-checkpoints', action='store_true', help='Evaluate all checkpoints in experiment')
    parser.add_argument('--episodes', type=int, default=20, help='Number of episodes to evaluate')
    parser.add_argument('--save-videos', action='store_true', help='Save evaluation videos')
    parser.add_argument('--video-episodes', type=int, default=3, help='Number of episodes to record')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    if not args.model and not args.experiment:
        print("❌ Must specify either --model or --experiment")
        return
    
    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    elif args.experiment:
        output_dir = os.path.join(args.experiment, 'evaluation_results')
    else:
        model_dir = os.path.dirname(args.model)
        output_dir = os.path.join(model_dir, '..', 'evaluation_results')
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        if args.all_checkpoints and args.experiment:
            # Evaluate all checkpoints
            print("🔍 Evaluating all checkpoints in experiment...")
            results = evaluate_all_checkpoints(args.experiment, args.episodes)
            
            # Save comparison results
            comparison_path = os.path.join(output_dir, 'checkpoint_comparison.json')
            with open(comparison_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n📊 Checkpoint comparison saved to: {comparison_path}")
            
            # Print summary
            print("\n" + "="*80)
            print("CHECKPOINT COMPARISON SUMMARY")
            print("="*80)
            print(f"{'Checkpoint':<40} {'Mean Score':<12} {'Success Rate':<12}")
            print("-" * 80)
            
            for filename, result in results.items():
                mean_score = result['mean_score']
                success_rate = result['success_rate'] * 100
                print(f"{filename:<40} {mean_score:>10.2f} {success_rate:>10.1f}%")
        
        else:
            # Evaluate single model
            model_path = args.model
            config_path = None
            
            if args.experiment:
                config_path = os.path.join(args.experiment, 'config.json')
                if not model_path:
                    model_path = os.path.join(args.experiment, 'checkpoints', 'best_model.pt')
            
            if not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path}")
                return
            
            print(f"🎯 Evaluating Distributional DQN model: {os.path.basename(model_path)}")
            
            # Create evaluator and run evaluation
            evaluator = ModelEvaluator(model_path, config_path, args.device)
            results, video_frames = evaluator.evaluate_multiple_episodes(
                args.episodes, args.save_videos, args.video_episodes
            )
            
            # Save results
            evaluator.save_evaluation_report(results, output_dir)
            
            # Save video if requested
            if args.save_videos and video_frames:
                video_path = os.path.join(output_dir, 'evaluation_video.mp4')
                print(f"🎬 Saving evaluation video...")
                try:
                    import imageio
                    with imageio.get_writer(video_path, fps=30) as writer:
                        for frame in video_frames:
                            writer.append_data(frame)
                    print(f"✅ Video saved: {video_path}")
                except ImportError:
                    print("⚠️ imageio not available, skipping video save")
            
            # Print summary
            print("\n" + "="*60)
            print("EVALUATION SUMMARY")
            print("="*60)
            print(f"Model Type:      {results.get('model_type', 'Unknown')}")
            print(f"Episodes:        {results['num_episodes']}")
            print(f"Mean Score:      {results['mean_score']:.2f} ± {results['std_score']:.2f}")
            print(f"Success Rate:    {results['success_rate']*100:.1f}% (≥19 score)")
            print(f"Target Status:   {'✅ ACHIEVED' if results['mean_score'] >= 19 else '❌ NOT REACHED'}")
            print("="*60)
    
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()