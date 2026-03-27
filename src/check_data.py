import json
import os
from PIL import Image

def check_data(data_root='../Data', annotation_file='grefs_with_grids.json'):
    """检查数据完整性"""
    
    annotation_path = os.path.join(data_root, annotation_file)
    
    # 检查标注文件
    if not os.path.exists(annotation_path):
        print(f"❌ 标注文件不存在: {annotation_path}")
        return False
    
    with open(annotation_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ 标注文件加载成功，共 {len(data)} 条记录")
    
    # 检查每条记录
    missing_images = []
    missing_grid_images = []
    invalid_points = []
    
    for idx, item in enumerate(data):
        # 检查原始图像
        image_name = item.get('file_name') or str(item['image_id'])
        image_path = os.path.join(data_root, 'images', image_name)
        if not os.path.exists(image_path):
            missing_images.append(image_name)
        
        # 检查网格图像
        grid_path = os.path.join(data_root, 'grid_images', 
                                os.path.basename(item.get('grid_image_path', '')))
        if not os.path.exists(grid_path):
            missing_grid_images.append(item.get('grid_image_path', ''))
        
        # 检查坐标点
        points = item.get('grid_points', [])
        if not points or not all(len(p) == 2 for p in points):
            invalid_points.append(item['image_id'])
    
    # 报告结果
    if missing_images:
        print(f"❌ 缺失原始图像: {len(missing_images)} 张")
        for img in missing_images[:5]:  # 只显示前5个
            print(f"   - {img}")
    else:
        print("✅ 所有原始图像都存在")
    
    if missing_grid_images:
        print(f"❌ 缺失网格图像: {len(missing_grid_images)} 张")
        for img in missing_grid_images[:5]:
            print(f"   - {img}")
    else:
        print("✅ 所有网格图像都存在")
    
    if invalid_points:
        print(f"❌ 无效坐标点: {len(invalid_points)} 条记录")
    else:
        print("✅ 所有坐标点都有效")
    
    # 统计信息
    total_points = sum(len(item.get('grid_points', [])) for item in data)
    print(f"\n📊 数据统计:")
    print(f"   - 总样本数: {len(data)}")
    print(f"   - 总坐标点数: {total_points}")
    print(f"   - 平均每图点数: {total_points / len(data):.2f}")
    
    return len(missing_images) == 0 and len(missing_grid_images) == 0

if __name__ == '__main__':
    check_data()