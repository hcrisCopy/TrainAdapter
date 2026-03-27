"""
主训练脚本：训练Coordinate Adapter
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# 添加src到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.adapter import CoordinateAdapter, LightweightCoordinateAdapter
from data.dataset import CoordinateDataset, collate_fn_pad_batch
from loss.hungarian_loss import HungarianPointLoss
from training.trainer import CoordinateAdapterTrainer, create_optimizer_and_scheduler
from training.config import get_config, CONFIG_PRESETS
from utils.coordinate_parser import CoordinateParser


def setup_transforms(config):
    """
    设置数据变换
    
    Args:
        config: 配置对象
        
    Returns:
        train_transform, val_transform
    """
    if config.data.use_data_augmentation:
        train_transform = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05
            ) if config.data.color_jitter else nn.Identity(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    # 验证集变换（不使用数据增强）
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_transform


def load_qwen_model(model_path, device):
    """
    加载Qwen2.5-VL模型
    
    Args:
        model_path: 模型路径
        device: 设备
        
    Returns:
        qwen_model, tokenizer
    """
    print(f"Loading Qwen2.5-VL from {model_path}")
    
    # 加载模型（简化版本，实际使用时需要完整加载）
    # 注意：这里需要根据Qwen2.5-VL的实际结构进行调整
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device
        )
        
        processor = AutoProcessor.from_pretrained(model_path)
        # 为模型挂载tokenizer引用，以便trainer可以直接调用
        model.tokenizer = processor.tokenizer
        
        print("Qwen2.5-VL loaded successfully")
        return model, processor.tokenizer
        
    except Exception as e:
        print(f"Warning: Failed to load Qwen2.5-VL: {e}")
        print("Creating mock model for testing")
        
        # 创建模拟模型（用于测试）
        class MockQwenModel:
            def __init__(self):
                self.device = device
                self.dtype = torch.float32
            
            def to(self, device):
                self.device = device
                return self
            
            def eval(self):
                pass
            
            def vision_encoder(self, images):
                # 模拟视觉编码器
                B = images.shape[0]
                return torch.randn(B, 196, 768).to(self.device)
            
            def text_encoder(self, input_ids, attention_mask):
                # 模拟文本编码器
                B = input_ids.shape[0]
                return torch.randn(B, 50, 768).to(self.device)
            
            def generate(self, **kwargs):
                # 模拟生成
                inputs_embeds = kwargs.get('inputs_embeds')
                B = inputs_embeds.shape[0]
                # 生成模拟文本（包含坐标）
                return torch.randint(0, 1000, (B, 20)).to(self.device)
            
            def get_tokenizer(self):
                from transformers import AutoTokenizer
                return AutoTokenizer.from_pretrained('gpt2')
        
        return MockQwenModel(), None


def create_adapter(config):
    """
    创建Coordinate Adapter
    
    Args:
        config: 配置对象
        
    Returns:
        adapter
    """
    if config.model.adapter_type == 'lightweight':
        adapter = LightweightCoordinateAdapter(
            visual_dim=config.model.visual_dim,
            grid_feature_dim=config.model.grid_feature_dim,
            hidden_dim=config.model.hidden_dim,
            num_heads=config.model.num_heads,
            num_grid_tokens=config.model.num_grid_tokens,
            dropout=config.model.dropout
        )
    else:
        adapter = CoordinateAdapter(
            visual_dim=config.model.visual_dim,
            grid_feature_dim=config.model.grid_feature_dim,
            hidden_dim=config.model.hidden_dim,
            num_heads=config.model.num_heads,
            num_grid_tokens=config.model.num_grid_tokens,
            dropout=config.model.dropout
        )
    
    return adapter


def create_dataloaders(config, train_transform, val_transform):
    """
    创建数据加载器
    
    Args:
        config: 配置对象
        train_transform: 训练变换
        val_transform: 验证变换
        
    Returns:
        train_dataloader, val_dataloader
    """
    # 训练数据集
    train_dataset = CoordinateDataset(
        data_root=config.data.data_root,
        annotation_file=config.data.annotation_file,
        image_dir=config.data.image_dir,
        grid_image_dir=config.data.grid_image_dir,
        tokenizer_path=config.model.qwen_model_path,
        image_size=config.data.image_size,
        max_length=config.data.max_length,
        transform=train_transform
    )
    
    # 验证数据集（如果有验证集）
    val_dataset = None
    val_dataloader = None
    
    if os.path.exists(os.path.join(config.data.data_root, 'val_grefs_with_grids.json')):
        val_dataset = CoordinateDataset(
            data_root=config.data.data_root,
            annotation_file='val_grefs_with_grids.json',
            image_dir=config.data.image_dir,
            grid_image_dir=config.data.grid_image_dir,
            tokenizer_path=config.model.qwen_model_path,
            image_size=config.data.image_size,
            max_length=config.data.max_length,
            transform=val_transform
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn_pad_batch
        )
    
    # 训练数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn_pad_batch
    )
    
    return train_dataloader, val_dataloader


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Train Coordinate Adapter')
    
    # 基本参数
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--preset', type=str, default='default', 
                       choices=list(CONFIG_PRESETS.keys()), help='预置配置')
    parser.add_argument('--save_dir', type=str, default='/root/autodl-tmp/Data/train_outputs', help='保存目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--resume', type=str, default=None, help='从检查点恢复')
    
    # 模型参数
    parser.add_argument('--adapter_type', type=str, choices=['standard', 'lightweight'], help='Adapter类型')
    parser.add_argument('--grid_feature_dim', type=int, help='网格特征维度')
    parser.add_argument('--hidden_dim', type=int, help='隐藏层维度')
    parser.add_argument('--num_heads', type=int, help='注意力头数')
    parser.add_argument('--num_grid_tokens', type=int, help='网格token数量')
    parser.add_argument('--dropout', type=float, help='Dropout率')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, help='数据根目录')
    parser.add_argument('--annotation_file', type=str, help='标注文件')
    parser.add_argument('--batch_size', type=int, help='Batch大小')
    parser.add_argument('--use_negative_samples', action='store_true', help='使用负样本')
    
    # 训练参数
    parser.add_argument('--lr', type=float, help='学习率')
    parser.add_argument('--weight_decay', type=float, help='权重衰减')
    parser.add_argument('--num_epochs', type=int, help='训练轮数')
    parser.add_argument('--gradient_accumulation_steps', type=int, help='梯度累积步数')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config and os.path.exists(args.config):
        config = get_config().load(args.config)
        print(f"Loaded config from {args.config}")
    else:
        config = get_config(args.preset)
        print(f"Using {args.preset} preset")
    
    # 从命令行参数更新配置
    config.update_from_args(vars(args))
    
    # 设置保存目录
    config.logging.save_dir = args.save_dir
    
    # 设置设备
    if args.device:
        config.device = args.device
    
    # 设置恢复检查点
    if args.resume:
        config.resume_from = args.resume
    
    # 打印配置
    print("=" * 50)
    print("Training Configuration:")
    print("=" * 50)
    print(f"Adapter type: {config.model.adapter_type}")
    print(f"Grid feature dim: {config.model.grid_feature_dim}")
    print(f"Hidden dim: {config.model.hidden_dim}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.lr}")
    print(f"Num epochs: {config.training.num_epochs}")
    print(f"Save dir: {config.logging.save_dir}")
    print(f"Device: {config.device}")
    print("=" * 50)
    
    # 创建保存目录
    os.makedirs(config.logging.save_dir, exist_ok=True)
    
    # 保存配置
    config_path = os.path.join(config.logging.save_dir, 'config.json')
    config.save(config_path)
    
    # 设置设备
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载Qwen2.5-VL模型
    qwen_model, tokenizer = load_qwen_model(config.model.qwen_model_path, device)
    
    # 创建Coordinate Adapter
    adapter = create_adapter(config)
    adapter.to(device)
    
    print(f"Adapter parameters: {adapter.get_parameter_count()}")
    
    # 设置数据变换
    train_transform, val_transform = setup_transforms(config)
    
    # 创建数据加载器
    train_dataloader, val_dataloader = create_dataloaders(
        config, train_transform, val_transform
    )
    
    print(f"Train dataset size: {len(train_dataloader.dataset)}")
    if val_dataloader:
        print(f"Val dataset size: {len(val_dataloader.dataset)}")
    
    # 创建损失函数
    loss_fn = HungarianPointLoss(
        inside_bbox_weight=config.training.inside_bbox_weight,
        outside_bbox_weight=config.training.outside_bbox_weight,
        match_cost=config.training.match_cost,
        boundary_penalty_weight=config.training.boundary_penalty_weight
    )
    
    # 创建优化器和调度器
    total_steps = len(train_dataloader) * config.training.num_epochs
    config.training.total_steps = total_steps
    
    optimizer, scheduler = create_optimizer_and_scheduler(
        adapter, 
        config.training.__dict__
    )
    
    # 创建训练器
    trainer = CoordinateAdapterTrainer(
        adapter=adapter,
        qwen_model=qwen_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        save_dir=config.logging.save_dir,
        max_grad_norm=config.training.max_grad_norm,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        log_interval=config.logging.log_interval,
        eval_interval=config.logging.eval_interval,
        save_interval=config.logging.save_interval
    )
    
    # 开始训练
    print("\n" + "=" * 50)
    print("Starting Training...")
    print("=" * 50 + "\n")
    
    trainer.train(
        num_epochs=config.training.num_epochs,
        resume_from=config.resume_from
    )
    
    print("\n" + "=" * 50)
    print("Training Completed!")
    print("=" * 50)


if __name__ == '__main__':
    main()
