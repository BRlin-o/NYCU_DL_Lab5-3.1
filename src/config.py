import yaml
import json
import os
import shutil
from datetime import datetime
from typing import Dict, Any, Optional
import argparse

class Config:
    """簡單但強大的配置管理類"""
    
    def __init__(self, config_path: str, **overrides):
        self.yaml_path = config_path
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # 讀取YAML配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.data = yaml.safe_load(f)
        
        # 處理配置繼承
        if '_base_' in self.data:
            self._load_base_config()
        
        # 應用命令行覆蓋
        for key, value in overrides.items():
            self._set_nested(key.replace('__', '.'), value)
        
        # 創建實驗目錄
        self.exp_dir = self._create_experiment_dir()
        
    def _load_base_config(self):
        """載入基礎配置文件"""
        base_path = self.data.pop('_base_')
        base_config_path = os.path.join(os.path.dirname(self.yaml_path), base_path)
        
        with open(base_config_path, 'r', encoding='utf-8') as f:
            base_data = yaml.safe_load(f)
        
        # 深度合併配置
        self.data = self._deep_merge(base_data, self.data)
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """深度合併兩個字典"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _set_nested(self, key: str, value: Any):
        """設置嵌套字典值"""
        keys = key.split('.')
        d = self.data
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        
        # 自動類型轉換
        if isinstance(value, str):
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.replace('.', '').replace('-', '').isdigit():
                value = float(value) if '.' in value else int(value)
        
        d[keys[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """獲取嵌套配置值"""
        keys = key.split('.')
        d = self.data
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return default
        return d
    
    def _create_experiment_dir(self) -> str:
        """創建實驗目錄"""
        exp_dir = f"experiments/{self.timestamp}"
        
        # 創建目錄結構
        subdirs = ['checkpoints', 'plots', 'videos', 'wandb_logs']
        for subdir in [''] + subdirs:
            os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)
        
        # 保存配置文件
        self.save_config(exp_dir)
        
        return exp_dir
    
    def save_config(self, exp_dir: str):
        """保存完整配置"""
        # 保存JSON格式配置
        config_path = os.path.join(exp_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        
        # 複製原始YAML
        original_path = os.path.join(exp_dir, 'original_config.yaml')
        shutil.copy(self.yaml_path, original_path)
        
        print(f"✅ 配置已保存到: {exp_dir}")
    
    def get_checkpoint_path(self, step: int) -> str:
        """獲取檢查點路徑"""
        student_id = self.get('experiment.student_id', 'STUDENT_ID')
        filename = f"LAB5_{student_id}_task3_pong{step}.pt"
        return os.path.join(self.exp_dir, 'checkpoints', filename)
    
    def get_best_model_path(self) -> str:
        """獲取最佳模型路徑"""
        return os.path.join(self.exp_dir, 'checkpoints', 'best_model.pt')
    
    def get_latest_checkpoint_path(self) -> str:
        """獲取最新檢查點路徑"""
        return os.path.join(self.exp_dir, 'checkpoints', 'latest.pt')
    
    def update_training_log(self, step: int, metrics: Dict[str, Any]):
        """更新訓練日誌"""
        log_path = os.path.join(self.exp_dir, 'training_log.json')
        
        # 讀取現有日誌
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # 添加新記錄
        log_entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        logs.append(log_entry)
        
        # 保存日誌
        with open(log_path, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def __str__(self):
        return f"Config(experiment={self.get('experiment.name')}, exp_dir={self.exp_dir})"

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='Task 3 Enhanced DQN Training')
    
    # 基本參數
    parser.add_argument('--config', type=str, default='configs/base.yaml',
                       help='配置文件路徑')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='訓練設備')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢復訓練的實驗目錄路径')
    
    # 可覆蓋的訓練參數
    parser.add_argument('--batch-size', type=int, help='批次大小')
    parser.add_argument('--lr', type=float, help='學習率')
    parser.add_argument('--total-steps', type=int, help='總訓練步數')
    parser.add_argument('--buffer-size', type=int, help='重播緩衝區大小')
    
    # 實驗控制
    parser.add_argument('--no-wandb', action='store_true', help='禁用W&B記錄')
    parser.add_argument('--debug', action='store_true', help='調試模式')
    
    return parser.parse_args()

def load_config_from_args():
    """從命令行參數載入配置"""
    args = parse_args()
    
    # 準備覆蓋參數
    overrides = {}
    if args.batch_size is not None:
        overrides['training__batch_size'] = args.batch_size
    if args.lr is not None:
        overrides['training__lr'] = args.lr  
    if args.total_steps is not None:
        overrides['training__total_steps'] = args.total_steps
    if args.buffer_size is not None:
        overrides['per__buffer_size'] = args.buffer_size
    
    # 其他設置
    overrides['device'] = args.device
    overrides['logging__wandb_enabled'] = not args.no_wandb
    overrides['debug'] = args.debug
    
    # 載入配置
    config = Config(args.config, **overrides)
    
    return config, args

if __name__ == "__main__":
    # 測試配置系統
    config, args = load_config_from_args()
    print(f"✅ 配置載入成功: {config}")
    print(f"📁 實驗目錄: {config.exp_dir}")
    print(f"🎯 實驗名稱: {config.get('experiment.name')}")
    print(f"📝 批次大小: {config.get('training.batch_size')}")
    print(f"🔧 設備: {config.get('device')}")