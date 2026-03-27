"""
坐标解析工具：从文本中提取和验证坐标
"""
import re
import torch
import numpy as np
from typing import List, Tuple, Union, Optional


def extract_coordinates(text: str, 
                       patterns: List[str] = None) -> List[List[float]]:
    """
    从文本中提取坐标点
    
    Args:
        text: 输入文本
        patterns: 正则表达式模式列表（可选）
        
    Returns:
        坐标点列表 [[x1, y1], [x2, y2], ...]
    """
    if patterns is None:
        patterns = [
            r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]',  # [x,y]
            r'\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)',  # (x,y)
            r'x\s*[:\-]\s*(\d+(?:\.\d+)?)\s*,\s*y\s*[:\-]\s*(\d+(?:\.\d+)?)',  # x:123, y:456
            r'(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)',  # x, y
        ]
    
    points = []
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                x = float(match[0])
                y = float(match[1])
                points.append([x, y])
            except (ValueError, IndexError):
                continue
    
    # 如果通过正则没提取到，尝试提取所有数字对
    if len(points) == 0:
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        if len(numbers) >= 2 and len(numbers) % 2 == 0:
            for i in range(0, len(numbers), 2):
                try:
                    x = float(numbers[i])
                    y = float(numbers[i+1])
                    points.append([x, y])
                except (ValueError, IndexError):
                    continue
    
    return points


def validate_coordinates(points: List[List[float]], 
                        image_width: int, 
                        image_height: int,
                        strict: bool = False) -> Tuple[List[List[float]], List[bool]]:
    """
    验证坐标是否在有效范围内
    
    Args:
        points: 坐标点列表
        image_width: 图像宽度
        image_height: 图像高度
        strict: 是否严格检查（必须完全在图像内）
        
    Returns:
        valid_points: 有效的坐标点
        validity_mask: 每个点是否有效的布尔列表
    """
    valid_points = []
    validity_mask = []
    
    for point in points:
        x, y = point
        
        if strict:
            # 严格检查：必须在图像范围内
            is_valid = (0 <= x <= image_width) and (0 <= y <= image_height)
        else:
            # 宽松检查：在合理范围内（允许稍微超出）
            is_valid = (0 <= x <= image_width * 1.2) and (0 <= y <= image_height * 1.2)
        
        validity_mask.append(is_valid)
        if is_valid:
            valid_points.append(point)
    
    return valid_points, validity_mask


def normalize_coordinates(points: List[List[float]], 
                         image_width: int, 
                         image_height: int,
                         to_range: Tuple[float, float] = (0, 1)) -> List[List[float]]:
    """
    坐标归一化
    
    Args:
        points: 原始坐标点
        image_width: 图像宽度
        image_height: 图像高度
        to_range: 归一化后的范围 (min, max)
        
    Returns:
        归一化后的坐标点
    """
    min_val, max_val = to_range
    normalized_points = []
    
    for point in points:
        x, y = point
        norm_x = (x / image_width) * (max_val - min_val) + min_val
        norm_y = (y / image_height) * (max_val - min_val) + min_val
        normalized_points.append([norm_x, norm_y])
    
    return normalized_points


def denormalize_coordinates(points: List[List[float]], 
                           image_width: int, 
                           image_height: int,
                           from_range: Tuple[float, float] = (0, 1)) -> List[List[float]]:
    """
    坐标反归一化
    
    Args:
        points: 归一化的坐标点
        image_width: 图像宽度
        image_height: 图像高度
        from_range: 归一化的范围 (min, max)
        
    Returns:
        反归一化后的坐标点
    """
    min_val, max_val = from_range
    denormalized_points = []
    
    for point in points:
        x, y = point
        denorm_x = (x - min_val) / (max_val - min_val) * image_width
        denorm_y = (y - min_val) / (max_val - min_val) * image_height
        denormalized_points.append([denorm_x, denorm_y])
    
    return denormalized_points


def clip_coordinates(points: List[List[float]], 
                    image_width: int, 
                    image_height: int) -> List[List[float]]:
    """
    将坐标裁剪到图像范围内
    
    Args:
        points: 原始坐标点
        image_width: 图像宽度
        image_height: 图像高度
        
    Returns:
        裁剪后的坐标点
    """
    clipped_points = []
    
    for point in points:
        x, y = point
        clipped_x = max(0, min(x, image_width))
        clipped_y = max(0, min(y, image_height))
        clipped_points.append([clipped_x, clipped_y])
    
    return clipped_points


def coordinates_to_tensor(points: List[List[float]], 
                         dtype: torch.dtype = torch.float32,
                         device: str = 'cpu') -> torch.Tensor:
    """
    坐标列表转换为张量
    
    Args:
        points: 坐标点列表
        dtype: 数据类型
        device: 设备
        
    Returns:
        坐标张量 [N, 2]
    """
    return torch.tensor(points, dtype=dtype, device=device)


def tensor_to_coordinates(tensor: torch.Tensor) -> List[List[float]]:
    """
    张量转换为坐标列表
    
    Args:
        tensor: 坐标张量 [N, 2]
        
    Returns:
        坐标点列表
    """
    return tensor.cpu().numpy().tolist()


def filter_duplicate_points(points: List[List[float]], 
                           threshold: float = 1.0) -> List[List[float]]:
    """
    过滤重复的点（距离小于阈值）
    
    Args:
        points: 坐标点列表
        threshold: 距离阈值
        
    Returns:
        过滤后的坐标点
    """
    if len(points) <= 1:
        return points
    
    filtered_points = []
    for point in points:
        is_duplicate = False
        for filtered_point in filtered_points:
            dist = np.linalg.norm(np.array(point) - np.array(filtered_point))
            if dist < threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered_points.append(point)
    
    return filtered_points


def sort_points_by_confidence(points: List[List[float]], 
                             confidences: List[float]) -> List[List[float]]:
    """
    按置信度排序点
    
    Args:
        points: 坐标点列表
        confidences: 置信度列表
        
    Returns:
        按置信度排序的坐标点
    """
    sorted_pairs = sorted(zip(confidences, points), reverse=True)
    return [point for _, point in sorted_pairs]


def compute_spatial_distribution(points: List[List[float]], 
                                image_width: int, 
                                image_height: int,
                                grid_size: Tuple[int, int] = (10, 10)) -> np.ndarray:
    """
    计算点的空间分布（用于分析）
    
    Args:
        points: 坐标点列表
        image_width: 图像宽度
        image_height: 图像高度
        grid_size: 网格大小 (grid_h, grid_w)
        
    Returns:
        空间分布矩阵 [grid_h, grid_w]
    """
    grid_h, grid_w = grid_size
    distribution = np.zeros((grid_h, grid_w))
    
    for point in points:
        x, y = point
        grid_x = int(x / image_width * grid_w)
        grid_y = int(y / image_height * grid_h)
        grid_x = min(grid_x, grid_w - 1)
        grid_y = min(grid_y, grid_h - 1)
        distribution[grid_y, grid_x] += 1
    
    return distribution


class CoordinateParser:
    """
    坐标解析器类：封装所有坐标解析和验证功能
    """
    def __init__(self, 
                 image_width: int = 448,
                 image_height: int = 448,
                 strict_validation: bool = False):
        """
        Args:
            image_width: 默认图像宽度
            image_height: 默认图像高度
            strict_validation: 是否严格验证
        """
        self.image_width = image_width
        self.image_height = image_height
        self.strict_validation = strict_validation
    
    def parse(self, text: str) -> List[List[float]]:
        """解析文本中的坐标"""
        return extract_coordinates(text)
    
    def validate(self, points: List[List[float]]) -> Tuple[List[List[float]], List[bool]]:
        """验证坐标有效性"""
        return validate_coordinates(points, self.image_width, self.image_height, self.strict_validation)
    
    def normalize(self, points: List[List[float]], to_range: Tuple[float, float] = (0, 1)) -> List[List[float]]:
        """归一化坐标"""
        return normalize_coordinates(points, self.image_width, self.image_height, to_range)
    
    def denormalize(self, points: List[List[float]], from_range: Tuple[float, float] = (0, 1)) -> List[List[float]]:
        """反归一化坐标"""
        return denormalize_coordinates(points, self.image_width, self.image_height, from_range)
    
    def clip(self, points: List[List[float]]) -> List[List[float]]:
        """裁剪坐标到图像范围内"""
        return clip_coordinates(points, self.image_width, self.image_height)
    
    def filter_duplicates(self, points: List[List[float]], threshold: float = 1.0) -> List[List[float]]:
        """过滤重复坐标"""
        return filter_duplicate_points(points, threshold)
    
    def to_tensor(self, points: List[List[float]], dtype: torch.dtype = torch.float32, device: str = 'cpu') -> torch.Tensor:
        """转换为张量"""
        return coordinates_to_tensor(points, dtype, device)
    
    def from_tensor(self, tensor: torch.Tensor) -> List[List[float]]:
        """从张量转换"""
        return tensor_to_coordinates(tensor)


# 测试代码
if __name__ == "__main__":
    print("=== 测试坐标解析 ===")
    
    test_texts = [
        "目标位置在[125, 240]和[300, 410]",
        "坐标是(150.5, 320.7)",
        "x: 200, y: 450",
        "位置在100, 200处",
        "没有坐标信息",
        "找到目标在[100,200][300,400][500,600]"
    ]
    
    for text in test_texts:
        points = extract_coordinates(text)
        print(f"Text: '{text}' -> Points: {points}")
    
    print("\n=== 测试坐标验证 ===")
    
    test_points = [[100, 150], [200, 300], [-10, 50], [500, 600]]
    valid_points, validity_mask = validate_coordinates(test_points, 448, 448, strict=False)
    print(f"Original points: {test_points}")
    print(f"Valid points: {valid_points}")
    print(f"Validity mask: {validity_mask}")
    
    print("\n=== 测试归一化 ===")
    
    points = [[224, 224], [100, 100], [400, 400]]
    normalized = normalize_coordinates(points, 448, 448, (0, 1))
    print(f"Original points: {points}")
    print(f"Normalized points: {normalized}")
    
    denormalized = denormalize_coordinates(normalized, 448, 448, (0, 1))
    print(f"Denormalized points: {denormalized}")
    
    print("\n=== 测试CoordinateParser类 ===")
    
    parser = CoordinateParser(image_width=448, image_height=448)
    
    text = "目标在[125.5, 240.7]和[300, 410]位置"
    points = parser.parse(text)
    print(f"Parsed points: {points}")
    
    valid_points, validity_mask = parser.validate(points)
    print(f"Valid points: {valid_points}")
    
    tensor = parser.to_tensor(points)
    print(f"Tensor shape: {tensor.shape}")
    
    points_back = parser.from_tensor(tensor)
    print(f"Points from tensor: {points_back}")
