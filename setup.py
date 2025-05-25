#!/usr/bin/env python3
"""
Task 3 Enhanced DQN - Automatic Setup Script

This script automatically sets up the environment for Task 3 Enhanced DQN:
1. Checks system requirements
2. Installs Python dependencies  
3. Verifies installation
4. Creates necessary directories
5. Downloads any required files

Usage:
    python setup.py
    python setup.py --minimal  # Install only essential packages
    python setup.py --dev      # Install development dependencies too
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path

def print_header():
    """Print setup header"""
    print("="*60)
    print("🚀 Task 3 Enhanced DQN - Automatic Setup")
    print("="*60)

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("   Required: Python 3.8 or higher")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_system_requirements():
    """Check system requirements"""
    print("\n💻 Checking system requirements...")
    
    # Check OS
    os_name = platform.system()
    print(f"   OS: {os_name} {platform.release()}")
    
    # Check available memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"   RAM: {memory_gb:.1f} GB")
        
        if memory_gb < 8:
            print("⚠️  Warning: Less than 8GB RAM detected")
            print("   Recommend: 16GB+ for optimal performance")
    except ImportError:
        print("   RAM: Unable to check (psutil not installed)")
    
    # Check disk space
    try:
        disk_free = os.statvfs('.').f_frsize * os.statvfs('.').f_available / (1024**3)
        print(f"   Disk: {disk_free:.1f} GB free")
        
        if disk_free < 10:
            print("⚠️  Warning: Less than 10GB free space")
            print("   Recommend: 20GB+ for experiments and checkpoints")
    except (AttributeError, OSError):
        # Windows doesn't have statvfs
        print("   Disk: Unable to check free space")
    
    return True

def check_cuda_support():
    """Check CUDA support"""
    print("\n🔥 Checking CUDA support...")
    
    try:
        # Try to run nvidia-smi
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            
            # Parse GPU info from nvidia-smi output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GeForce' in line or 'RTX' in line or 'GTX' in line:
                    # Extract GPU name
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'GeForce' in part or 'RTX' in part or 'GTX' in part:
                            gpu_name = ' '.join(parts[i:i+3])
                            print(f"   GPU: {gpu_name}")
                            break
            
            return True
        else:
            print("❌ nvidia-smi failed")
            return False
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ NVIDIA GPU not detected or nvidia-smi not available")
        print("   Training will use CPU (much slower)")
        return False

def create_virtual_environment():
    """Create a virtual environment"""
    print("\n🔧 Creating virtual environment...")
    
    venv_path = ".venv"
    
    # Check if venv already exists
    if os.path.exists(venv_path):
        print(f"   ⚠️ Virtual environment already exists at {venv_path}")
        return True, venv_path
    
    # Create virtual environment
    try:
        subprocess.run([sys.executable, '-m', 'venv', venv_path], check=True)
        print(f"✅ Virtual environment created at {venv_path}")
        return True, venv_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False, None

def install_pytorch_cuda():
    """Install PyTorch with CUDA support"""
    print("\n🔥 Installing PyTorch with CUDA support...")
    
    # Detect CUDA version or use default
    cuda_version = "cu126"  # Default to CUDA 11.8
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            # Try to parse CUDA version from nvidia-smi
            output = result.stdout
            if "CUDA Version: 12." in output:
                cuda_version = "cu126"
                print("   Detected CUDA 12.x")
            elif "CUDA Version: 11.8" in output:
                cuda_version = "cu118"
                print("   Detected CUDA 11.8")
            else:
                print("   Using default CUDA 12.6")
    except:
        print("   Using default CUDA 12.6")
    
    # Install PyTorch
    cmd = [
        sys.executable, '-m', 'pip', 'install',
        'torch', 'torchvision',
        '--index-url', f'https://download.pytorch.org/whl/{cuda_version}'
    ]
    
    print(f"   Installing: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ PyTorch with CUDA installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ PyTorch installation failed: {e}")
        return False

def install_requirements(minimal=False, dev=False):
    """Install Python requirements"""
    print(f"\n📦 Installing Python dependencies...")
    
    # Choose requirements file
    if minimal:
        req_file = "requirements-minimal.txt"
        print("   Using minimal dependencies")
    else:
        req_file = "requirements.txt"
        print("   Using full dependencies")
    
    if not os.path.exists(req_file):
        print(f"❌ Requirements file not found: {req_file}")
        return False
    
    # Install requirements
    cmd = [sys.executable, '-m', 'pip', 'install', '-r', req_file]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Dependencies installed from {req_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Dependency installation failed: {e}")
        return False
    
    # Install development dependencies if requested
    if dev:
        dev_packages = [
            'black', 'isort', 'flake8', 'pytest',
            'jupyter', 'notebook', 'ipywidgets'
        ]
        
        print("   Installing development dependencies...")
        cmd = [sys.executable, '-m', 'pip', 'install'] + dev_packages
        
        try:
            subprocess.run(cmd, check=True)
            print("✅ Development dependencies installed")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Some development dependencies failed: {e}")
    
    return True

def create_directory_structure():
    """Create necessary directories"""
    print("\n📁 Creating directory structure...")
    
    directories = [
        'experiments',
        'configs',
        'src',
        'logs',
        'videos',
        'plots',
        'checkpoints',
        'eval_results'
    ]
    
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"   ✅ {dir_name}/")
    
    return True

def verify_installation():
    """Verify that all dependencies are properly installed"""
    print("\n🔍 Verifying installation...")
    
    # Core imports
    test_imports = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('numpy', 'NumPy'),
        ('gymnasium', 'Gymnasium'),
        ('ale_py', 'ALE-Py'),
        ('cv2', 'OpenCV'),
        ('wandb', 'Weights & Biases'),
        ('matplotlib', 'Matplotlib'),
        ('yaml', 'PyYAML'),
        ('psutil', 'PSUtil'),
    ]
    
    failed_imports = []
    
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError as e:
            print(f"   ❌ {name}: {e}")
            failed_imports.append(name)
    
    # Test PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"   ✅ PyTorch CUDA: {gpu_count} GPU(s) available")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"      GPU {i}: {gpu_name}")
        else:
            print("   ⚠️ PyTorch CUDA: Not available (CPU only)")
    except ImportError:
        print("   ❌ PyTorch: Not available")
        failed_imports.append('PyTorch')
    
    # Test Atari environment
    try:
        import gymnasium as gym
        import ale_py
        gym.register_envs(ale_py)
        env = gym.make('ALE/Pong-v5')
        env.close()
        print("   ✅ Atari Environment: Pong-v5 working")
    except Exception as e:
        print(f"   ❌ Atari Environment: {e}")
        failed_imports.append('Atari Environment')
    
    if failed_imports:
        print(f"\n❌ Verification failed for: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✅ All dependencies verified successfully!")
        return True

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("🎉 Setup Complete! Next Steps:")
    print("="*60)
    print("1. Quick test:")
    print("   python run_task3.py status")
    print()
    print("2. Start training:")
    print("   python run_task3.py train --config fast --device cuda:0")
    print()
    print("3. Monitor progress:")
    print("   - Check console output")
    print("   - View W&B dashboard")
    print()
    print("4. Evaluate results:")
    print("   python run_task3.py evaluate")
    print()
    print("📚 Documentation: README.md")
    print("🐛 Troubleshooting: Check GPU memory and CUDA drivers")
    print("="*60)

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description='Task 3 Enhanced DQN Setup')
    parser.add_argument('--minimal', action='store_true', 
                       help='Install only minimal dependencies')
    parser.add_argument('--dev', action='store_true',
                       help='Install development dependencies')
    parser.add_argument('--skip-pytorch', action='store_true',
                       help='Skip PyTorch installation')
    parser.add_argument('--skip-verify', action='store_true',
                       help='Skip installation verification')
    
    args = parser.parse_args()
    
    print_header()
    
    # Step 1: Check system requirements
    if not check_python_version():
        print("\n❌ Setup failed: Python version incompatible")
        return 1
    
    if not check_system_requirements():
        print("\n❌ Setup failed: System requirements not met")
        return 1
    
    # Step 2: Check CUDA support
    has_cuda = check_cuda_support()
    
    # Step 3: Install PyTorch with CUDA
    if not args.skip_pytorch:
        if has_cuda:
            if not install_pytorch_cuda():
                print("\n❌ Setup failed: PyTorch installation failed")
                return 1
        else:
            print("\n⚠️ Installing CPU-only PyTorch (training will be slow)")
            cmd = [sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio']
            try:
                subprocess.run(cmd, check=True)
                print("✅ CPU PyTorch installed")
            except subprocess.CalledProcessError as e:
                print(f"❌ PyTorch installation failed: {e}")
                return 1
    
    # Step 4: Install other dependencies
    if not install_requirements(minimal=args.minimal, dev=args.dev):
        print("\n❌ Setup failed: Dependency installation failed")
        return 1
    
    # Step 5: Create directories
    if not create_directory_structure():
        print("\n❌ Setup failed: Directory creation failed")
        return 1
    
    # Step 6: Verify installation
    if not args.skip_verify:
        if not verify_installation():
            print("\n⚠️ Setup completed with some issues")
            print("   You may need to fix the failed dependencies manually")
    
    # Step 7: Print next steps
    print_next_steps()
    
    print("🎯 Ready to start Task 3 Enhanced DQN training!")
    return 0

if __name__ == "__main__":
    exit(main())