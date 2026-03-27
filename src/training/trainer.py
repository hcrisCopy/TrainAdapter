"""
训练器：负责训练Coordinate Adapter
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import json
import logging
from datetime import datetime


class CoordinateAdapterTrainer:
    """
    Coordinate Adapter训练器
    """
    def __init__(self, 
                 adapter,
                 qwen_model,
                 train_dataloader,
                 val_dataloader=None,
                 optimizer=None,
                 scheduler=None,
                 loss_fn=None,
                 device='cuda',
                 save_dir='outputs',
                 max_grad_norm=1.0,
                 gradient_accumulation_steps=1,
                 log_interval=10,
                 eval_interval=500,
                 save_interval=1000):
        """
        Args:
            adapter: Coordinate Adapter模型
            qwen_model: Qwen2.5-VL模型（冻结）
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            optimizer: 优化器
            scheduler: 学习率调度器
            loss_fn: 损失函数
            device: 设备
            save_dir: 保存目录
            max_grad_norm: 梯度裁剪阈值
            gradient_accumulation_steps: 梯度累积步数
            log_interval: 日志间隔
            eval_interval: 验证间隔
            save_interval: 保存间隔
        """
        self.adapter = adapter.to(device)
        self.qwen_model = qwen_model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.save_dir = save_dir
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'train_outputs'), exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 记录模型信息
        self.logger.info(f"Adapter parameters: {self.adapter.get_parameter_count()}")
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # 冻结Qwen模型
        self._freeze_qwen_model()
    
    def _setup_logging(self):
        """设置日志"""
        log_file = os.path.join(
            self.save_dir, 
            'logs', 
            f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _freeze_qwen_model(self):
        """冻结Qwen2.5-VL模型参数"""
        for param in self.qwen_model.parameters():
            param.requires_grad = False
        
        self.qwen_model.eval()
        self.logger.info("Qwen2.5-VL model frozen and set to eval mode")
    
    def save_checkpoint(self, step, loss, is_best=False):
        """保存检查点"""
        checkpoint = {
            'step': step,
            'epoch': self.epoch,
            'model_state_dict': self.adapter.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss': loss,
            'best_loss': self.best_loss
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.save_dir, 'checkpoints', f'checkpoint_step_{step}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.save_dir, 'checkpoints', 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model at step {step} with loss {loss:.4f}")
        
        self.logger.info(f"Saved checkpoint at step {step}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.adapter.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.global_step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def generate_coordinates(self, batch):
        """
        生成坐标：使用Qwen2.5-VL生成文本，然后解析坐标
        
        Args:
            batch: 批次数据
            
        Returns:
            generated_texts: 生成的文本列表
            pred_points_list: 解析的坐标点列表
        """
        # 提取批次数据
        images = batch['image'].to(self.device)
        grid_images = batch['grid_image'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        batch_size = images.shape[0]
        
        # 1. 视觉编码（冻结）
        with torch.no_grad():
            # Qwen2.5-VL视觉编码器，注意Qwen2.5-VL真实网络层是 visual
            visual_features = self.qwen_model.visual(images)  # [B, N, D]
            # grid_visual_features = self.qwen_model.visual(grid_images)  # [B, N, D]
        
        # 2. Adapter增强（可训练）
        enhanced_features = self.adapter(images, grid_images, visual_features)
        
        # 3. 文本编码（冻结）
        with torch.no_grad():
            text_embeddings = self.qwen_model.text_encoder(input_ids, attention_mask)
        
        # 4. 融合特征
        # 将文本特征与增强的视觉特征结合
        fused_features = torch.cat([text_embeddings, enhanced_features], dim=1)
        
        # 5. 生成文本（冻结）
        with torch.no_grad():
            outputs = self.qwen_model.generate(
                inputs_embeds=fused_features,
                attention_mask=attention_mask,
                max_length=100,
                do_sample=True,
                temperature=0.7
            )
        
        # 6. 解码文本
        generated_texts = []
        for i in range(batch_size):
            generated_text = self.qwen_model.tokenizer.decode(
                outputs[i], 
                skip_special_tokens=True
            )
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def train_step(self, batch):
        """
        单步训练
        
        Args:
            batch: 批次数据
            
        Returns:
            loss: 损失值
        """
        # 设置为训练模式
        self.adapter.train()
        
        # 生成坐标文本
        generated_texts = self.generate_coordinates(batch)
        
        # 获取真值数据
        gt_points_list = batch['gt_points']
        image_sizes = batch['image_size']
        
        # 计算损失
        loss, match_info = self.loss_fn(
            pred_texts=generated_texts,
            gt_points_list=gt_points_list,
            image_sizes=image_sizes
        )
        
        # 反向传播
        loss.backward()
        
        # 梯度累积
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            # 梯度裁剪
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.adapter.parameters(), 
                    self.max_grad_norm
                )
            
            # 更新参数
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()
        
        return loss.item(), match_info
    
    def evaluate(self):
        """
        验证模型
        
        Returns:
            avg_loss: 平均损失
            metrics: 评估指标
        """
        if self.val_dataloader is None:
            return None, None
        
        self.adapter.eval()
        total_loss = 0.0
        total_samples = 0
        
        all_match_info = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc='Evaluating'):
                # 生成坐标
                generated_texts = self.generate_coordinates(batch)
                
                # 获取真值数据
                gt_points_list = batch['gt_points']
                image_sizes = batch['image_size']
                
                # 计算损失
                loss, match_info = self.loss_fn(
                    pred_texts=generated_texts,
                    gt_points_list=gt_points_list,
                    image_sizes=image_sizes
                )
                
                total_loss += loss.item()
                total_samples += len(batch['image'])
                all_match_info.extend(match_info)
        
        avg_loss = total_loss / len(self.val_dataloader)
        
        # 计算评估指标
        metrics = self._compute_metrics(all_match_info)
        
        return avg_loss, metrics
    
    def _compute_metrics(self, match_info):
        """
        计算评估指标
        
        Args:
            match_info: 匹配信息列表
            
        Returns:
            metrics: 指标字典
        """
        total_l1_error = 0.0
        total_samples = 0
        acc_5 = 0
        acc_10 = 0
        
        for info in match_info:
            pred_points = info['pred_points']
            gt_points = info['gt_points']
            image_size = info.get('image_size', (500, 500))
            
            if len(pred_points) == 0 or len(gt_points) == 0:
                continue
            
            # 计算最近距离
            for gt_point in gt_points:
                min_dist = float('inf')
                for pred_point in pred_points:
                    dist = np.linalg.norm(np.array(pred_point) - np.array(gt_point))
                    min_dist = min(min_dist, dist)
                
                total_l1_error += min_dist
                total_samples += 1
                
                # 计算准确率
                max_size = max(image_size)
                if min_dist < 0.05 * max_size:  # 5%范围内
                    acc_5 += 1
                if min_dist < 0.1 * max_size:   # 10%范围内
                    acc_10 += 1
        
        metrics = {
            'l1_error': total_l1_error / total_samples if total_samples > 0 else 0.0,
            'acc_5': acc_5 / total_samples if total_samples > 0 else 0.0,
            'acc_10': acc_10 / total_samples if total_samples > 0 else 0.0,
            'total_samples': total_samples
        }
        
        return metrics
    
    def train(self, num_epochs, resume_from=None):
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
            resume_from: 从检查点恢复训练
        """
        if resume_from:
            self.load_checkpoint(resume_from)
        
        self.logger.info(f"Start training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            epoch_loss = 0.0
            num_batches = 0
            
            # 训练
            for batch_idx, batch in enumerate(tqdm(self.train_dataloader, desc=f'Training Epoch {epoch + 1}')):
                try:
                    loss, match_info = self.train_step(batch)
                    
                    epoch_loss += loss
                    num_batches += 1
                    self.global_step += 1
                    
                    # 日志
                    if self.global_step % self.log_interval == 0:
                        self.logger.info(
                            f"Step {self.global_step}, Loss: {loss:.4f}, "
                            f"Avg Loss: {epoch_loss / num_batches:.4f}"
                        )
                    
                    # 验证
                    if self.val_dataloader and self.global_step % self.eval_interval == 0:
                        val_loss, metrics = self.evaluate()
                        if val_loss is not None:
                            self.logger.info(
                                f"Validation - Loss: {val_loss:.4f}, "
                                f"L1 Error: {metrics['l1_error']:.2f}, "
                                f"Acc@5: {metrics['acc_5']:.2%}, "
                                f"Acc@10: {metrics['acc_10']:.2%}"
                            )
                            
                            # 保存最佳模型
                            if val_loss < self.best_loss:
                                self.best_loss = val_loss
                                self.save_checkpoint(self.global_step, val_loss, is_best=True)
                    
                    # 保存检查点
                    if self.global_step % self.save_interval == 0:
                        self.save_checkpoint(self.global_step, loss)
                
                except Exception as e:
                    self.logger.error(f"Error at step {self.global_step}: {str(e)}")
                    continue
            
            #  epoch结束
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            self.logger.info(f"Epoch {epoch + 1} completed, Avg Loss: {avg_epoch_loss:.4f}")
            
            # 保存epoch检查点
            self.save_checkpoint(self.global_step, avg_epoch_loss)
        
        self.logger.info("Training completed!")


def create_optimizer_and_scheduler(adapter, train_config):
    """
    创建优化器和学习率调度器
    
    Args:
        adapter: Adapter模型
        train_config: 训练配置
        
    Returns:
        optimizer, scheduler
    """
    # 优化器
    optimizer = AdamW(
        adapter.get_trainable_parameters(),
        lr=train_config['lr'],
        weight_decay=train_config['weight_decay'],
        betas=train_config.get('betas', (0.9, 0.999))
    )
    
    # 学习率调度器
    scheduler = None
    if train_config.get('use_scheduler', True):
        warmup_steps = train_config.get('warmup_steps', 500)
        total_steps = train_config.get('total_steps', 10000)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + torch.cos(torch.tensor(np.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler


if __name__ == "__main__":
    # 测试代码
    print("=== 测试训练器 ===")
    
    # 模拟配置
    train_config = {
        'lr': 1e-4,
        'weight_decay': 0.01,
        'betas': (0.9, 0.999),
        'use_scheduler': True,
        'warmup_steps': 500,
        'total_steps': 10000
    }
    
    # 创建模拟模型（需要替换为实际模型）
    class MockAdapter:
        def get_trainable_parameters(self):
            return [torch.nn.Parameter(torch.randn(10, 10))]
        
        def get_parameter_count(self):
            return {'total': 100, 'trainable': 100}
        
        def to(self, device):
            return self
        
        def train(self):
            pass
        
        def eval(self):
            pass
    
    class MockQwenModel:
        def to(self, device):
            return self
        
        def eval(self):
            pass
    
    # 创建组件
    adapter = MockAdapter()
    qwen_model = MockQwenModel()
    
    # 创建优化器和调度器
    optimizer, scheduler = create_optimizer_and_scheduler(adapter, train_config)
    
    print(f"Optimizer: {optimizer}")
    print(f"Scheduler: {scheduler}")
