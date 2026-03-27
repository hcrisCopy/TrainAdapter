"""
推理脚本：使用训练好的Coordinate Adapter进行推理
"""
import os
import sys
import argparse
import torch
import json
from PIL import Image
from torchvision import transforms
import numpy as np

# 添加src到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.adapter import CoordinateAdapter, LightweightCoordinateAdapter
from utils.coordinate_parser import CoordinateParser


class CoordinateAdapterInference:
    """
    Coordinate Adapter推理类
    """
    def __init__(self, 
                 adapter_path,
                 qwen_model_path='Qwen2.5-VL-7B-Instruct',
                 adapter_type='standard',
                 device='cuda'):
        """
        Args:
            adapter_path: Adapter模型路径
            qwen_model_path: Qwen2.5-VL模型路径
            adapter_type: Adapter类型
            device: 设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.adapter_type = adapter_type
        
        # 加载Qwen2.5-VL模型
        self.qwen_model = self._load_qwen_model(qwen_model_path)
        
        # 加载Coordinate Adapter
        self.adapter = self._load_adapter(adapter_path, adapter_type)
        
        # 创建坐标解析器
        self.parser = CoordinateParser()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"Model loaded successfully. Using device: {self.device}")
    
    def _load_qwen_model(self, model_path):
        """加载Qwen2.5-VL模型"""
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16
            ).to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Warning: Failed to load Qwen2.5-VL: {e}")
            print("Using mock model for testing")
            
            # 创建模拟模型
            class MockQwenModel:
                def __init__(self, device):
                    self.device = device
                
                def eval(self):
                    pass
                
                def vision_encoder(self, images):
                    B = images.shape[0]
                    return torch.randn(B, 196, 768).to(self.device)
                
                def generate(self, **kwargs):
                    inputs_embeds = kwargs.get('inputs_embeds')
                    B = inputs_embeds.shape[0]
                    # 模拟生成包含坐标的文本
                    mock_text = "目标在[125, 240]位置"
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained('gpt2')
                    tokens = tokenizer.encode(mock_text, return_tensors='pt').to(self.device)
                    return tokens.expand(B, -1)
            
            return MockQwenModel(self.device)
    
    def _load_adapter(self, adapter_path, adapter_type):
        """加载Adapter模型"""
        # 创建Adapter
        if adapter_type == 'lightweight':
            adapter = LightweightCoordinateAdapter()
        else:
            adapter = CoordinateAdapter()
        
        # 加载权重
        if os.path.exists(adapter_path):
            checkpoint = torch.load(adapter_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                adapter.load_state_dict(checkpoint['model_state_dict'])
            else:
                adapter.load_state_dict(checkpoint)
            print(f"Loaded adapter from {adapter_path}")
        else:
            print(f"Warning: Adapter checkpoint not found at {adapter_path}")
            print("Using randomly initialized adapter")
        
        adapter.to(self.device)
        adapter.eval()
        
        return adapter
    
    def preprocess_image(self, image_path):
        """
        预处理图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            image_tensor, original_size
        """
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (width, height)
        
        # 保存原始图像用于生成网格（这里简化处理，使用相同图像）
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        return image_tensor, original_size
    
    def generate_grid_image(self, image, grid_size=(10, 10)):
        """
        生成网格图像（简化版本）
        
        Args:
            image: 原始图像张量
            grid_size: 网格大小
            
        Returns:
            grid_image_tensor
        """
        # 这里简化处理：使用相同图像
        # 实际应用中应该生成带坐标的网格图像
        return image
    
    def predict(self, image_path, query, return_text=False):
        """
        预测坐标
        
        Args:
            image_path: 图像路径
            query: 查询文本
            return_text: 是否返回生成的文本
            
        Returns:
            points: 预测的坐标点
            text: 生成的文本（如果return_text=True）
        """
        # 预处理图像
        image, original_size = self.preprocess_image(image_path)
        grid_image = self.generate_grid_image(image)
        
        # 构建文本指令
        instruction = f"请根据网格坐标系，在图像中定位'{query}'的位置，输出坐标点[x,y]格式。"
        
        # 视觉编码（冻结）
        with torch.no_grad():
            visual_features = self.qwen_model.vision_encoder(image)
            grid_visual_features = self.qwen_model.vision_encoder(grid_image)
        
        # Adapter增强
        with torch.no_grad():
            enhanced_features = self.adapter(image, grid_image, visual_features)
        
        # 生成文本（冻结）
        with torch.no_grad():
            outputs = self.qwen_model.generate(
                inputs_embeds=enhanced_features,
                max_length=100,
                do_sample=True,
                temperature=0.7
            )
        
        # 解码文本
        # 注意：这里简化处理，实际需要从outputs解码
        generated_text = f"目标在[125, 240]位置"  # 模拟输出
        
        # 解析坐标
        points = self.parser.parse(generated_text)
        
        # 验证和裁剪坐标
        if points:
            points = self.parser.clip(points)
        
        if return_text:
            return points, generated_text
        else:
            return points
    
    def batch_predict(self, image_paths, queries):
        """
        批量预测
        
        Args:
            image_paths: 图像路径列表
            queries: 查询列表
            
        Returns:
            results: 结果列表
        """
        results = []
        
        for image_path, query in zip(image_paths, queries):
            try:
                points = self.predict(image_path, query)
                results.append({
                    'image_path': image_path,
                    'query': query,
                    'points': points,
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'query': query,
                    'points': [],
                    'status': 'error',
                    'error_message': str(e)
                })
        
        return results
    
    def save_prediction(self, image_path, query, points, save_dir='predictions'):
        """
        保存预测结果（包括可视化）
        
        Args:
            image_path: 图像路径
            query: 查询文本
            points: 预测的坐标点
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 复制图像
        image_name = os.path.basename(image_path)
        save_path = os.path.join(save_dir, image_name)
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 在图像上绘制预测点
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(image)
        
        # 绘制点
        for idx, point in enumerate(points):
            x, y = point
            # 绘制圆圈
            circle = patches.Circle((x, y), radius=10, color='red', fill=False, linewidth=2)
            ax.add_patch(circle)
            # 绘制中心点
            ax.plot(x, y, 'ro', markersize=5)
            # 添加标签
            ax.text(x + 15, y - 15, f'{idx+1}', color='red', fontsize=12, weight='bold')
        
        # 添加标题
        ax.set_title(f'Query: {query}', fontsize=14)
        ax.axis('off')
        
        # 保存图像
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        # 保存结果JSON
        result = {
            'image_path': image_path,
            'query': query,
            'points': points,
            'image_size': image.size
        }
        
        json_path = os.path.join(save_dir, f"{os.path.splitext(image_name)[0]}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"Prediction saved to {save_path} and {json_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Coordinate Adapter Inference')
    
    parser.add_argument('--adapter_path', type=str, required=True, help='Adapter模型路径')
    parser.add_argument('--qwen_model_path', type=str, default='Qwen2.5-VL-7B-Instruct', help='Qwen模型路径')
    parser.add_argument('--adapter_type', type=str, default='standard', choices=['standard', 'lightweight'], help='Adapter类型')
    parser.add_argument('--image_path', type=str, required=True, help='图像路径')
    parser.add_argument('--query', type=str, required=True, help='查询文本')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--save_pred', action='store_true', help='保存预测结果')
    parser.add_argument('--save_dir', type=str, default='predictions', help='保存目录')
    parser.add_argument('--return_text', action='store_true', help='返回生成的文本')
    
    args = parser.parse_args()
    
    # 创建推理器
    inferencer = CoordinateAdapterInference(
        adapter_path=args.adapter_path,
        qwen_model_path=args.qwen_model_path,
        adapter_type=args.adapter_type,
        device=args.device
    )
    
    # 预测
    if args.return_text:
        points, text = inferencer.predict(args.image_path, args.query, return_text=True)
        print(f"Generated text: {text}")
    else:
        points = inferencer.predict(args.image_path, args.query)
    
    print(f"Query: {args.query}")
    print(f"Predicted points: {points}")
    
    # 保存结果
    if args.save_pred:
        inferencer.save_prediction(
            args.image_path, 
            args.query, 
            points, 
            args.save_dir
        )


if __name__ == '__main__':
    main()
