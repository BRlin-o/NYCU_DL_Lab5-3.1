#!/usr/bin/env python3
"""
Task 3 Enhanced DQN - Quick Setup and Run Script

This script provides easy commands to set up and run Task 3 experiments:

1. Setup environment and dependencies
2. Run training with different configurations  
3. Evaluate trained models
4. Generate reports and videos

Usage:
    python run_task3.py setup
    python run_task3.py train --config fast --device cuda:0
    python run_task3.py evaluate --experiment experiments/2025-05-25_14-30-45
    python run_task3.py ablation --device cuda:1
"""

import os
import sys
import subprocess
import argparse
import json
import glob
from datetime import datetime
from pathlib import Path

def setup_environment():
    """Setup environment and check dependencies"""
    print("🔧 Setting up Task 3 Enhanced DQN environment...")
    
    # Create directory structure
    dirs = [
        'experiments',
        'configs', 
        'src',
        'logs',
        'videos',
        'plots'
    ]
    
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"   ✅ Created directory: {dir_name}")
    
    # Check Python packages
    required_packages = [
        'torch', 'torchvision', 'numpy', 'gymnasium', 'ale-py',
        'opencv-python', 'wandb', 'matplotlib', 'seaborn', 
        'imageio', 'psutil', 'gputil', 'pyyaml'
    ]
    
    print("\n📦 Checking required packages...")
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"\n🔥 CUDA available: {gpu_count} GPU(s)")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"   GPU {i}: {gpu_name}")
        else:
            print("\n⚠️ CUDA not available - training will be slow on CPU")
    except:
        print("\n❌ Could not check CUDA availability")
    
    print("\n✅ Environment setup complete!")
    return True

def run_training(config_name: str, device: str = 'auto', extra_args: list = None):
    """Run training with specified configuration"""
    
    # Config mapping
    config_map = {
        'fast': 'configs/rainbow_fast.yaml',
        'stable': 'configs/rainbow_stable.yaml', 
        'base': 'configs/base.yaml',
        'debug': 'configs/debug.yaml'
    }
    
    config_path = config_map.get(config_name, config_name)
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    print(f"🚀 Starting training with config: {config_name}")
    print(f"   Config file: {config_path}")
    print(f"   Device: {device}")
    
    # Build command
    cmd = [
        sys.executable, 'train.py',
        '--config', config_path,
        '--device', device
    ]
    
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"   Command: {' '.join(cmd)}")
    print("-" * 60)
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return False

def run_evaluation(experiment_dir: str = None, model_path: str = None, episodes: int = 20):
    """Run model evaluation"""
    
    if experiment_dir:
        if not os.path.exists(experiment_dir):
            print(f"❌ Experiment directory not found: {experiment_dir}")
            return False
        print(f"🎯 Evaluating experiment: {experiment_dir}")
        cmd = [
            sys.executable, 'evaluate.py',
            '--experiment', experiment_dir,
            '--episodes', str(episodes),
            '--save-videos'
        ]
    elif model_path:
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
        print(f"🎯 Evaluating model: {model_path}")
        cmd = [
            sys.executable, 'evaluate.py',
            '--model', model_path,
            '--episodes', str(episodes),
            '--save-videos'
        ]
    else:
        # Find latest experiment
        exp_dirs = glob.glob('experiments/2*')
        if not exp_dirs:
            print("❌ No experiments found")
            return False
        
        latest_exp = max(exp_dirs, key=os.path.getctime)
        print(f"🎯 Evaluating latest experiment: {latest_exp}")
        cmd = [
            sys.executable, 'evaluate.py',
            '--experiment', latest_exp,
            '--episodes', str(episodes),
            '--save-videos'
        ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Evaluation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Evaluation failed with exit code: {e.returncode}")
        return False

def run_ablation_studies(device: str = 'auto'):
    """Run ablation studies"""
    
    ablation_configs = [
        'configs/ablation_no_per.yaml',
        'configs/ablation_no_double.yaml', 
        'configs/ablation_no_multistep.yaml',
        # 'configs/vanilla_dqn.yaml'  # Add if needed
    ]
    
    print("🔬 Running ablation studies...")
    print(f"   Configurations: {len(ablation_configs)}")
    print(f"   Device: {device}")
    
    results = {}
    
    for i, config_path in enumerate(ablation_configs):
        if not os.path.exists(config_path):
            print(f"⚠️ Config not found: {config_path}")
            continue
            
        config_name = os.path.basename(config_path).replace('.yaml', '')
        print(f"\n[{i+1}/{len(ablation_configs)}] Running: {config_name}")
        
        cmd = [
            sys.executable, 'train.py',
            '--config', config_path,
            '--device', device
        ]
        
        try:
            subprocess.run(cmd, check=True)
            results[config_name] = 'SUCCESS'
            print(f"✅ {config_name} completed")
        except subprocess.CalledProcessError:
            results[config_name] = 'FAILED'
            print(f"❌ {config_name} failed")
        except KeyboardInterrupt:
            print(f"\n⚠️ Ablation interrupted during {config_name}")
            break
    
    # Summary
    print(f"\n📊 Ablation Study Results:")
    for config, status in results.items():
        status_icon = "✅" if status == "SUCCESS" else "❌"
        print(f"   {status_icon} {config}: {status}")
    
    return results

def generate_final_report():
    """Generate final report with all results"""
    print("📋 Generating final Task 3 report...")
    
    # Find all experiments
    exp_dirs = glob.glob('experiments/2*')
    if not exp_dirs:
        print("❌ No experiments found")
        return False
    
    print(f"   Found {len(exp_dirs)} experiments")
    
    # Analyze each experiment
    all_results = {}
    
    for exp_dir in exp_dirs:
        config_path = os.path.join(exp_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            exp_name = config.get('experiment', {}).get('name', os.path.basename(exp_dir))
            
            # Check for results
            best_model = os.path.join(exp_dir, 'checkpoints', 'best_model.pt')
            if os.path.exists(best_model):
                all_results[exp_name] = {
                    'dir': exp_dir,
                    'config': config,
                    'has_model': True
                }
    
    # Create report
    report_path = f"Task3_Final_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(report_path, 'w') as f:
        f.write("# Task 3 Enhanced DQN - Final Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        f.write(f"- Total Experiments: {len(all_results)}\n")
        f.write(f"- Experiment Types: {', '.join(all_results.keys())}\n\n")
        
        # Individual results
        f.write("## Experiment Results\n\n")
        
        for exp_name, info in all_results.items():
            f.write(f"### {exp_name}\n\n")
            f.write(f"- Directory: `{info['dir']}`\n")
            f.write(f"- Configuration: {info['config'].get('experiment', {}).get('description', 'N/A')}\n")
            f.write(f"- Model Available: {'✅' if info['has_model'] else '❌'}\n\n")
    
    print(f"✅ Report generated: {report_path}")
    return True

def show_status():
    """Show current status of experiments"""
    print("📊 Task 3 Enhanced DQN - Current Status")
    print("=" * 50)
    
    # Check experiments
    exp_dirs = glob.glob('experiments/2*')
    print(f"Experiments: {len(exp_dirs)}")
    
    for exp_dir in sorted(exp_dirs):
        exp_name = os.path.basename(exp_dir)
        config_path = os.path.join(exp_dir, 'config.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            name = config.get('experiment', {}).get('name', 'Unknown')
            
            # Check checkpoints
            checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
            checkpoint_count = 0
            if os.path.exists(checkpoints_dir):
                checkpoint_count = len([f for f in os.listdir(checkpoints_dir) if f.endswith('.pt')])
            
            print(f"   📁 {exp_name}")
            print(f"      Name: {name}")
            print(f"      Checkpoints: {checkpoint_count}")
            
            # Check if best model exists
            best_model = os.path.join(checkpoints_dir, 'best_model.pt')
            if os.path.exists(best_model):
                print(f"      Best Model: ✅")
            else:
                print(f"      Best Model: ❌")
    
    print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description='Task 3 Enhanced DQN - Quick Setup and Run')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup environment')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Run training')
    train_parser.add_argument('--config', type=str, default='fast',
                             choices=['fast', 'stable', 'base', 'debug'],
                             help='Training configuration')
    train_parser.add_argument('--device', type=str, default='auto',
                             help='Training device')
    train_parser.add_argument('--batch-size', type=int, help='Override batch size')
    train_parser.add_argument('--lr', type=float, help='Override learning rate')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--experiment', type=str, help='Experiment directory')
    eval_parser.add_argument('--model', type=str, help='Model checkpoint path')
    eval_parser.add_argument('--episodes', type=int, default=20, help='Number of episodes')
    
    # Ablation command
    ablation_parser = subparsers.add_parser('ablation', help='Run ablation studies')
    ablation_parser.add_argument('--device', type=str, default='auto', help='Training device')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate final report')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show current status')
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        success = setup_environment()
        exit(0 if success else 1)
    
    elif args.command == 'train':
        extra_args = []
        if args.batch_size:
            extra_args.extend(['--batch-size', str(args.batch_size)])
        if args.lr:
            extra_args.extend(['--lr', str(args.lr)])
        
        success = run_training(args.config, args.device, extra_args)
        exit(0 if success else 1)
    
    elif args.command == 'evaluate':
        success = run_evaluation(args.experiment, args.model, args.episodes)
        exit(0 if success else 1)
    
    elif args.command == 'ablation':
        results = run_ablation_studies(args.device)
        success = all(status == 'SUCCESS' for status in results.values())
        exit(0 if success else 1)
    
    elif args.command == 'report':
        success = generate_final_report()
        exit(0 if success else 1)
    
    elif args.command == 'status':
        show_status()
        exit(0)
    
    else:
        print("❌ No command specified. Use --help for available commands.")
        parser.print_help()
        exit(1)

if __name__ == "__main__":
    main()