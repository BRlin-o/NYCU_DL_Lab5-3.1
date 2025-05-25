"""
Utility functions for Task 3 Enhanced DQN
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import psutil
import GPUtil
from collections import defaultdict
import cv2
import imageio

def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information"""
    info = {
        'timestamp': datetime.now().isoformat(),
        'cpu': {
            'count': psutil.cpu_count(),
            'percent': psutil.cpu_percent(),
            'freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        },
        'memory': {
            'total': psutil.virtual_memory().total,
            'available': psutil.virtual_memory().available,
            'percent': psutil.virtual_memory().percent,
        },
        'disk': {
            'total': psutil.disk_usage('/').total,
            'free': psutil.disk_usage('/').free,
            'percent': psutil.disk_usage('/').percent,
        }
    }
    
    # GPU information
    if torch.cuda.is_available():
        info['cuda'] = {
            'available': True,
            'device_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device(),
            'devices': []
        }
        
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                info['cuda']['devices'].append({
                    'id': i,
                    'name': gpu.name,
                    'memory_total': gpu.memoryTotal,
                    'memory_used': gpu.memoryUsed,
                    'memory_free': gpu.memoryFree,
                    'load': gpu.load,
                    'temperature': gpu.temperature,
                })
        except:
            # Fallback to torch info
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info['cuda']['devices'].append({
                    'id': i,
                    'name': props.name,
                    'memory_total': props.total_memory // 1024**2,  # MB
                    'major': props.major,
                    'minor': props.minor,
                })
    else:
        info['cuda'] = {'available': False}
    
    return info

def get_gpu_utilization() -> float:
    """Get GPU utilization percentage"""
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            return gpus[0].load * 100
    except:
        pass
    return 0.0

def get_memory_usage() -> float:
    """Get system memory usage percentage"""
    return psutil.virtual_memory().percent

def save_training_curves(exp_dir: str, metrics_history: Dict[str, List], save_format: str = 'png'):
    """Save training curves as plots"""
    plots_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Performance curves
    if 'eval_score_mean' in metrics_history:
        plt.figure(figsize=(12, 8))
        
        steps = list(range(len(metrics_history['eval_score_mean'])))
        scores = metrics_history['eval_score_mean']
        
        plt.subplot(2, 2, 1)
        plt.plot(steps, scores, linewidth=2, alpha=0.8)
        plt.axhline(y=19, color='red', linestyle='--', alpha=0.7, label='Target (19)')
        plt.title('Evaluation Score vs Steps')
        plt.xlabel('Evaluation Steps')
        plt.ylabel('Average Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Training loss
        if 'loss' in metrics_history:
            plt.subplot(2, 2, 2)
            plt.plot(metrics_history['loss'], linewidth=1, alpha=0.7)
            plt.title('Training Loss')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
        
        # Exploration
        if 'epsilon' in metrics_history:
            plt.subplot(2, 2, 3)
            plt.plot(metrics_history['epsilon'], linewidth=2, alpha=0.8)
            plt.title('Exploration (Epsilon)')
            plt.xlabel('Steps')
            plt.ylabel('Epsilon')
            plt.grid(True, alpha=0.3)
        
        # Q-values
        if 'q_value_mean' in metrics_history:
            plt.subplot(2, 2, 4)
            plt.plot(metrics_history['q_value_mean'], linewidth=1, alpha=0.7)
            plt.title('Q-Value Mean')
            plt.xlabel('Training Steps')
            plt.ylabel('Q-Value')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'training_curves.{save_format}'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # Sample efficiency plot
    if 'sample_efficiency' in metrics_history:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_history['sample_efficiency'], linewidth=2, alpha=0.8)
        plt.title('Sample Efficiency Over Time')
        plt.xlabel('Steps')
        plt.ylabel('Sample Efficiency')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, f'sample_efficiency.{save_format}'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"📊 Training curves saved to: {plots_dir}")

def analyze_experiment_results(exp_dir: str) -> Dict[str, Any]:
    """Analyze experiment results and generate summary"""
    # Load training log
    log_path = os.path.join(exp_dir, 'training_log.json')
    config_path = os.path.join(exp_dir, 'config.json')
    
    if not os.path.exists(log_path):
        print(f"❌ Training log not found: {log_path}")
        return {}
    
    with open(log_path, 'r') as f:
        logs = json.load(f)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract metrics
    steps = [log['step'] for log in logs]
    eval_scores = [log.get('eval_score_mean', 0) for log in logs if 'eval_score_mean' in log]
    
    if not eval_scores:
        print("⚠️ No evaluation scores found in logs")
        return {}
    
    # Calculate statistics
    best_score = max(eval_scores)
    final_score = eval_scores[-1] if eval_scores else 0
    
    # Find when target was reached
    target_score = config.get('early_stopping', {}).get('target_score', 19.0)
    target_reached_step = None
    for i, score in enumerate(eval_scores):
        if score >= target_score:
            target_reached_step = steps[i] if i < len(steps) else None
            break
    
    # Sample efficiency analysis
    sample_efficiency = None
    if target_reached_step:
        sample_efficiency = target_reached_step
    
    # Task score estimation
    task_score = "0%"
    if sample_efficiency:
        if sample_efficiency <= 200000:
            task_score = "15%"
        elif sample_efficiency <= 400000:
            task_score = "12%"
        elif sample_efficiency <= 600000:
            task_score = "10%"
        elif sample_efficiency <= 800000:
            task_score = "8%"
        elif sample_efficiency <= 1000000:
            task_score = "6%"
        else:
            task_score = "3%"
    
    analysis = {
        'experiment_name': config.get('experiment', {}).get('name', 'Unknown'),
        'total_steps': max(steps) if steps else 0,
        'total_episodes': len([log for log in logs if 'episode' in log]),
        'best_score': best_score,
        'final_score': final_score,
        'target_score': target_score,
        'target_reached': target_reached_step is not None,
        'target_reached_step': target_reached_step,
        'sample_efficiency': sample_efficiency,
        'estimated_task_score': task_score,
        'config_summary': {
            'batch_size': config.get('training', {}).get('batch_size'),
            'learning_rate': config.get('training', {}).get('lr'),
            'buffer_size': config.get('per', {}).get('buffer_size'),
            'n_step': config.get('training', {}).get('n_step'),
        }
    }
    
    return analysis

def compare_experiments(exp_dirs: List[str]) -> None:
    """Compare multiple experiments"""
    results = []
    
    for exp_dir in exp_dirs:
        if os.path.exists(exp_dir):
            analysis = analyze_experiment_results(exp_dir)
            if analysis:
                results.append({
                    'dir': exp_dir,
                    'name': analysis['experiment_name'],
                    **analysis
                })
    
    if not results:
        print("❌ No valid experiments found")
        return
    
    # Create comparison table
    print("\n" + "="*120)
    print("🔍 EXPERIMENT COMPARISON")
    print("="*120)
    
    header = f"{'Experiment':<30} {'Best Score':<12} {'Target':<8} {'Steps':<12} {'Task Score':<10} {'Batch':<8} {'LR':<10}"
    print(header)
    print("-" * 120)
    
    for result in results:
        name = result['name'][:28] if len(result['name']) > 28 else result['name']
        best_score = f"{result['best_score']:.2f}"
        target = "✅" if result['target_reached'] else "❌"
        steps = f"{result['target_reached_step']:,}" if result['target_reached_step'] else "N/A"
        task_score = result['estimated_task_score']
        batch_size = str(result['config_summary']['batch_size'])
        lr = f"{result['config_summary']['learning_rate']:.0e}"
        
        row = f"{name:<30} {best_score:<12} {target:<8} {steps:<12} {task_score:<10} {batch_size:<8} {lr:<10}"
        print(row)
    
    print("="*120)

def create_evaluation_video(model_path: str, output_path: str, episodes: int = 5, 
                          config_path: str = None) -> None:
    """Create evaluation video from trained model"""
    import gymnasium as gym
    import ale_py
    from dqn_task3 import DQN, AtariPreprocessor
    
    # Setup environment
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    preprocessor = AtariPreprocessor()
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model with default config if not provided
    if config_path and os.path.exists(config_path):
        from src.config import Config
        config = Config(config_path)
    else:
        # Default config for video generation
        config = type('Config', (), {
            'get': lambda self, key, default=None: {
                'model.dueling': True,
                'model.hidden_dim': 512,
            }.get(key, default)
        })()
    
    model = DQN(env.action_space.n, config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Record episodes
    all_frames = []
    total_rewards = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        state = preprocessor.reset(obs)
        done = False
        total_reward = 0
        episode_frames = []
        
        while not done and len(episode_frames) < 10000:  # Limit frames per episode
            # Render frame
            frame = env.render()
            episode_frames.append(frame)
            
            # Select action
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()
            
            # Step environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = preprocessor.step(next_obs)
        
        all_frames.extend(episode_frames)
        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Reward = {total_reward:.1f}, Frames = {len(episode_frames)}")
    
    # Save video
    print(f"💾 Saving video with {len(all_frames)} frames...")
    with imageio.get_writer(output_path, fps=30) as writer:
        for frame in all_frames:
            writer.append_data(frame)
    
    avg_reward = np.mean(total_rewards)
    print(f"✅ Video saved: {output_path}")
    print(f"📊 Average reward across {episodes} episodes: {avg_reward:.2f}")
    
    env.close()

def validate_checkpoint(checkpoint_path: str) -> bool:
    """Validate checkpoint file integrity"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        required_keys = ['model_state_dict', 'step', 'config']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        
        if missing_keys:
            print(f"❌ Missing keys in checkpoint: {missing_keys}")
            return False
        
        print(f"✅ Checkpoint validation passed: {checkpoint_path}")
        print(f"   Step: {checkpoint['step']}")
        print(f"   Best Score: {checkpoint.get('best_score', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Checkpoint validation failed: {e}")
        return False

def cleanup_experiment_dir(exp_dir: str, keep_best: bool = True) -> None:
    """Clean up experiment directory to save space"""
    checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
    
    if not os.path.exists(checkpoints_dir):
        return
    
    # Files to keep
    keep_files = {'best_model.pt', 'latest.pt'}
    
    # Add milestone checkpoints
    milestone_pattern = 'LAB5_*_task3_pong*.pt'
    
    # Remove unnecessary files
    removed_count = 0
    for filename in os.listdir(checkpoints_dir):
        filepath = os.path.join(checkpoints_dir, filename)
        
        if filename not in keep_files and not filename.startswith('LAB5_'):
            if filename.endswith('.pt'):
                os.remove(filepath)
                removed_count += 1
    
    print(f"🧹 Cleaned up {removed_count} checkpoint files from {exp_dir}")

if __name__ == "__main__":
    # Test system info
    info = get_system_info()
    print("🖥️ System Information:")
    print(f"   CPU: {info['cpu']['count']} cores")
    print(f"   Memory: {info['memory']['total'] / 1e9:.1f}GB")
    if info['cuda']['available']:
        print(f"   GPU: {len(info['cuda']['devices'])} device(s)")
        for gpu in info['cuda']['devices']:
            print(f"      {gpu['name']}: {gpu.get('memory_total', 'N/A')}MB")
    
    print(f"\n📊 Current Usage:")
    print(f"   GPU Utilization: {get_gpu_utilization():.1f}%")
    print(f"   Memory Usage: {get_memory_usage():.1f}%")