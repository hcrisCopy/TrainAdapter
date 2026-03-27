import json
import sys
import os

def query_image(filename):
    instances_path = "/root/autodl-tmp/gRefCOCO/data/gRefCOCO/instances.json"
    grefs_path = "/root/autodl-tmp/gRefCOCO/data/gRefCOCO/grefs(unc).json"
    
    if not os.path.exists(instances_path) or not os.path.exists(grefs_path):
        print("找不到对应的 JSON 文件，请检查路径。")
        return
    
    print(f"正在读取 {instances_path} ...")
    with open(instances_path, 'r') as f:
        instances_data = json.load(f)
        
    print(f"正在读取 {grefs_path} ...\n")
    with open(grefs_path, 'r') as f:
        grefs_data = json.load(f)
        
    images = instances_data.get('images', [])
    exact_img = None
    
    for img in images:
        if img['file_name'] == filename:
            exact_img = img
            break
            
    if not exact_img:
        print(f"[-] 在 instances.json 中没有找到完全匹配的文件名: {filename}")
        return
        
    img_id = exact_img['id']
    print("="*50)
    print(f"[+] 找到匹配图片: {filename}")
    print(f"    Image ID: {img_id}")
    print(f"    原始宽高: {exact_img.get('width')} x {exact_img.get('height')}")
    print("="*50)
    
    # 查找 instances_data 中的标注
    annotations = [ann for ann in instances_data.get('annotations', []) if ann.get('image_id') == img_id]
    print(f"\n[Instances.json 相关标注信息] (共 {len(annotations)} 个实体):")
    for ann in annotations:
        print(f"  - Ann ID: {ann.get('id')}, Category ID: {ann.get('category_id')}, BBox: {ann.get('bbox')}, Has Segmentation: {'segmentation' in ann}")

    # 查找 grefs(unc).json 中的信息
    matched_grefs = [g for g in grefs_data if g.get('image_id') == img_id or g.get('file_name') == filename]
    print(f"\n[grefs(unc).json 相关指代信息] (共 {len(matched_grefs)} 条指令):")
    for gidx, g in enumerate(matched_grefs, 1):
        print(f"  [{gidx}] Ref ID: {g.get('ref_id')}")
        print(f"      No Target: {g.get('no_target')}")
        print(f"      Target Ann IDs: {g.get('ann_id')}")
        print("      Sentences:")
        sentences = g.get('sentences', [])
        for s in sentences:
            print(f"        - {s.get('raw')}")
            
    print("="*50)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        query_image(sys.argv[1])
    else:
        print("用法: python query_image_in_json.py <图片文件名>")
        print("示例: python query_image_in_json.py COCO_train2014_000000000072.jpg")
