#!/usr/bin/env python3
"""
Task 3 Enhanced DQN Training Script
Usage:
    python train.py --config configs/rainbow_fast.yaml --device cuda:0
    python train.py --config configs/rainbow_stable.yaml --batch-size 64
    python train.py --config configs/base.yaml --resume experiments/2025-05-25_14-30-45
"""

import os
import sys
import torch
import numpy as np
import random
from datetime import datetime
import traceback

# Add src to path
sys.path.append('src')

from dqn_task3_5 import EnhancedDQNAgent
from src.config import Config, load_config_from_args

def setup_environment():
    """Setup training environment"""
    # Ensure required directories exist
    os.makedirs('experiments', exist_ok=True)
    os.makedirs('configs', exist_ok=True)
    
    # Set CUDA settings for optimal performance
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f"🔥 CUDA available: {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("⚠️ CUDA not available, using CPU")

def load_checkpoint_if_resume(config, args, agent):
    """Load checkpoint if resuming training"""
    if args.resume:
        checkpoint_path = os.path.join(args.resume, 'checkpoints', 'latest.pt')
        
        if os.path.exists(checkpoint_path):
            print(f"📂 Loading checkpoint from: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=agent.device)
            
            # Load model states
            agent.q_net.load_state_dict(checkpoint['model_state_dict'])
            agent.target_net.load_state_dict(checkpoint['target_model_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load training state
            agent.env_count = checkpoint.get('env_count', 0)
            agent.train_count = checkpoint.get('train_count', 0)
            agent.episode_count = checkpoint.get('episode', 0)
            agent.epsilon = checkpoint.get('epsilon', agent.epsilon)
            agent.best_score = checkpoint.get('best_score', -21.0)
            
            # Update PER beta
            if 'per_beta' in checkpoint:
                agent.memory.update_beta(checkpoint['per_beta'])
            
            print(f"✅ Resumed from step {agent.env_count}, episode {agent.episode_count}")
            print(f"   Best score: {agent.best_score:.2f}, Epsilon: {agent.epsilon:.4f}")
            
            return True
        else:
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return False
    
    return False

def print_training_info(config, agent):
    """Print comprehensive training information"""
    print("\n" + "="*80)
    print("🚀 ENHANCED DQN TRAINING - TASK 3")
    print("="*80)
    
    print(f"📋 Experiment: {config.get('experiment.name')}")
    print(f"📝 Description: {config.get('experiment.description')}")
    print(f"📁 Directory: {config.exp_dir}")
    print(f"🏷️ Tags: {config.get('experiment.tags', [])}")
    
    print("\n📊 MODEL CONFIGURATION:")
    print(f"   Architecture: {config.get('model.architecture')}")
    print(f"   Dueling: {config.get('model.dueling')}")
    print(f"   Hidden Dim: {config.get('model.hidden_dim')}")
    print(f"   Input Channels: {config.get('model.input_channels')}")
    print(f"   Output Actions: {agent.num_actions}")
    
    print("\n🎯 TRAINING CONFIGURATION:")
    print(f"   Total Steps: {config.get('training.total_steps'):,}")
    print(f"   Batch Size: {config.get('training.batch_size')}")
    print(f"   Learning Rate: {config.get('training.lr')}")
    print(f"   Gamma: {config.get('training.gamma')}")
    print(f"   N-Step: {config.get('training.n_step')}")
    print(f"   Target Update Freq: {config.get('training.target_update_freq'):,}")
    
    print("\n🔍 EXPLORATION:")
    print(f"   Epsilon Start: {config.get('exploration.epsilon_start')}")
    print(f"   Epsilon Final: {config.get('exploration.epsilon_final')}")
    print(f"   Decay Steps: {config.get('exploration.epsilon_decay_steps'):,}")
    
    print("\n💾 PRIORITIZED EXPERIENCE REPLAY:")
    print(f"   Buffer Size: {config.get('per.buffer_size'):,}")
    print(f"   Alpha: {config.get('per.alpha')}")
    print(f"   Beta Start: {config.get('per.beta_start')}")
    print(f"   Beta Final: {config.get('per.beta_final')}")
    
    print("\n📈 EVALUATION:")
    print(f"   Eval Frequency: {config.get('evaluation.eval_frequency'):,} steps")
    print(f"   Eval Episodes: {config.get('evaluation.eval_episodes')}")
    print(f"   Target Score: {config.get('early_stopping.target_score')}")
    
    print("\n💻 HARDWARE:")
    print(f"   Device: {config.get('device', 'auto')}")
    print(f"   Mixed Precision: {config.get('hardware.mixed_precision')}")
    print(f"   Model Compilation: {config.get('hardware.compile_model')}")
    print(f"   Workers: {config.get('hardware.num_workers')}")
    
    print("\n📊 MONITORING:")
    print(f"   W&B Enabled: {config.get('logging.wandb_enabled')}")
    if config.get('logging.wandb_enabled'):
        print(f"   W&B Project: {config.get('logging.wandb_project')}")
        print(f"   W&B Group: {config.get('logging.wandb_group')}")
    
    milestone_steps = config.get('logging.milestone_steps', [])
    print(f"   Milestone Steps: {milestone_steps}")
    
    print("="*80 + "\n")

def estimate_training_time(config):
    """Estimate training time based on configuration"""
    total_steps = config.get('training.total_steps', 1000000)
    batch_size = config.get('training.batch_size', 32)
    device = config.get('device', 'cuda:0')
    
    # Rough estimates based on hardware (very approximate)
    if 'cuda' in device:
        if batch_size >= 128:  # RTX3090 setup
            steps_per_hour = 50000
        elif batch_size >= 64:  # RTX4080 setup
            steps_per_hour = 35000
        else:  # Conservative setup
            steps_per_hour = 25000
    else:
        steps_per_hour = 5000  # CPU
    
    estimated_hours = total_steps / steps_per_hour
    
    print(f"⏰ ESTIMATED TRAINING TIME:")
    print(f"   Total Steps: {total_steps:,}")
    print(f"   Estimated Speed: {steps_per_hour:,} steps/hour")
    print(f"   Estimated Time: {estimated_hours:.1f} hours ({estimated_hours/24:.1f} days)")
    
    # Milestone estimates
    milestone_steps = config.get('logging.milestone_steps', [200000, 400000, 600000, 800000, 1000000])
    print(f"   Milestone Estimates:")
    for step in milestone_steps:
        hours = step / steps_per_hour
        print(f"      {step:>7,} steps: ~{hours:.1f}h")
    
    print()

def main():
    """Main training function"""
    try:
        # Setup environment
        setup_environment()
        
        # Load configuration
        print("📋 Loading configuration...")
        config, args = load_config_from_args()
        
        # Set random seeds for reproducibility
        seed = config.get('seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        print(f"🎲 Random seed set to: {seed}")
        
        # Create agent
        print("🤖 Initializing Enhanced DQN Agent...")
        agent = EnhancedDQNAgent(config)
        
        # Load checkpoint if resuming
        resumed = load_checkpoint_if_resume(config, args, agent)
        
        # Print training information
        print_training_info(config, agent)
        estimate_training_time(config)
        
        # Confirm before starting
        if not config.get('debug', False) and not resumed:
            response = input("🚀 Ready to start training? [Y/n]: ")
            if response.lower() in ['n', 'no']:
                print("❌ Training cancelled.")
                return
        
        # Start training
        start_time = datetime.now()
        print(f"🏁 Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 80)
        
        best_score = agent.run()
        
        # Training completed
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*80)
        print("🎉 TRAINING COMPLETED!")
        print("="*80)
        print(f"🏆 Best Score Achieved: {best_score:.2f}")
        print(f"⏱️ Training Duration: {duration}")
        print(f"📁 Results saved in: {config.exp_dir}")
        print(f"💾 Best model: {config.get_best_model_path()}")
        
        # Check if target achieved
        target_score = config.get('early_stopping.target_score', 19.0)
        if best_score >= target_score:
            print(f"✅ TARGET ACHIEVED! Score {best_score:.2f} >= {target_score}")
            
            # Estimate task score based on steps
            total_steps = agent.env_count
            if total_steps <= 200000:
                task_score = "15%"
            elif total_steps <= 400000:
                task_score = "12%"
            elif total_steps <= 600000:
                task_score = "10%"
            elif total_steps <= 800000:
                task_score = "8%"
            elif total_steps <= 1000000:
                task_score = "6%"
            else:
                task_score = "3%"
            
            print(f"📊 Estimated Task Score: {task_score} ({total_steps:,} steps)")
        else:
            print(f"⚠️ Target not reached. Score {best_score:.2f} < {target_score}")
        
        print("="*80)
        
        return best_score
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user (Ctrl+C)")
        if 'agent' in locals():
            print("💾 Saving emergency checkpoint...")
            agent.save_checkpoint(agent.env_count, [], 'emergency')
        return None
        
    except Exception as e:
        print(f"\n❌ Training failed with error:")
        print(f"   {type(e).__name__}: {e}")
        print("\n📄 Full traceback:")
        traceback.print_exc()
        
        if 'agent' in locals():
            print("\n💾 Saving emergency checkpoint...")
            try:
                agent.save_checkpoint(agent.env_count, [], 'error')
            except:
                print("❌ Failed to save emergency checkpoint")
        
        return None
    
    finally:
        # Cleanup
        try:
            import wandb
            if wandb.run:
                wandb.finish()
        except:
            pass

if __name__ == "__main__":
    best_score = main()
    
    if best_score is not None:
        print(f"\n🎯 Final Result: {best_score:.2f}")
        if best_score >= 19.0:
            exit(0)  # Success
        else:
            exit(1)  # Target not reached
    else:
        exit(2)  # Training failed or interrupted