import json
import os
import numpy as np
from PIL import Image, ImageDraw

# ================= 配置参数 =================
MASK_SCALE_FACTOR = 0.85
# ============================================

def scale_mask_region(mask_img, scale_factor=0.98):
    if scale_factor >= 1.0:
        return mask_img
        
    mask_np = np.array(mask_img)
    rows = np.any(mask_np, axis=1)
    cols = np.any(mask_np, axis=0)
    
    if not np.any(rows):
        return mask_img  
        
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    w = cmax - cmin + 1
    h = rmax - rmin + 1
    
    roi = mask_img.crop((cmin, rmin, cmax + 1, rmax + 1))
    
    new_w = max(1, int(w * scale_factor))
    new_h = max(1, int(h * scale_factor))
    
    scaled_roi = roi.resize((new_w, new_h), Image.Resampling.NEAREST)
    
    new_mask = Image.new('L', mask_img.size, 0)
    
    offset_x = cmin + (w - new_w) // 2
    offset_y = rmin + (h - new_h) // 2
    
    new_mask.paste(scaled_roi, (offset_x, offset_y))
    return new_mask

def mask_from_polygons(polygons, width, height):
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        if len(poly) >= 6:
            draw.polygon(poly, outline=1, fill=1)
            
    mask = scale_mask_region(mask, MASK_SCALE_FACTOR)
    return mask

def process_data():
    grefs_path = "/root/autodl-tmp/Data/grefs(unc).json"
    instances_path = "/root/autodl-tmp/gRefCOCO/data/gRefCOCO/instances.json"
    output_grefs_path = "/root/autodl-tmp/Data/grefs_with_grids.json"

    with open(grefs_path, 'r') as f:
        grefs = json.load(f)

    with open(instances_path, 'r') as f:
        instances = json.load(f)

    images_dict = {img['id']: img for img in instances.get('images', [])}
    ann_dict = {ann['id']: ann for ann in instances.get('annotations', [])}

    print(f"Total queries (grefs) to process: {len(grefs)}")
    
    # 提前为所有的 instance 算好由于图像和bbox已知而固定的网格点。
    # 因为很多grefs重复引用相同的 ann_id 
    print("Pre-computing grid points for all valid instances...")
    valid_ann_ids = set()
    for g in grefs:
        aids = g.get('ann_id', [])
        if not isinstance(aids, list):
            aids = [aids]
        for a in aids:
            valid_ann_ids.add(a)

    precomputed_points = {}
    total_anns = len(instances.get('annotations', []))
    processed_anns = 0
    for ann in instances.get('annotations', []):
        aid = ann['id']
        if aid not in valid_ann_ids:
            continue
            
        processed_anns += 1
        if processed_anns % 10000 == 0:
            print(f"  Precomputed {processed_anns} instances...")
            
        if 'segmentation' not in ann or not isinstance(ann['segmentation'], list):
            precomputed_points[aid] = []
            continue
            
        img_info = images_dict.get(ann['image_id'])
        if not img_info:
            precomputed_points[aid] = []
            continue
            
        orig_width, orig_height = img_info['width'], img_info['height']
        new_width = (orig_width // 28) * 28
        new_height = (orig_height // 28) * 28
        
        if new_width <= 0 or new_height <= 0:
            precomputed_points[aid] = []
            continue
            
        mask = mask_from_polygons(ann['segmentation'], orig_width, orig_height)
        mask_resized = mask.resize((new_width, new_height), Image.Resampling.NEAREST)
        mask_resized_np = np.array(mask_resized)
        
        target_grid_points = []
        for i in range(11):
            x = i * (new_width / 10.0)
            ix = int(round(x))
            if ix >= new_width: ix = new_width - 1
            if ix < 0: ix = 0
            
            for j in range(11):
                y = j * (new_height / 10.0)
                iy = int(round(y))
                if iy >= new_height: iy = new_height - 1
                if iy < 0: iy = 0
                
                if mask_resized_np[iy, ix] == 1:
                    target_grid_points.append([i, j])
                    
        precomputed_points[aid] = target_grid_points

    print("Pre-computation done. Assigning to grefs...")
    
    new_grefs = []
    
    # 赋值给 Grefs
    for g in grefs:
        new_g = g.copy()
        target_ids = g.get('ann_id', [])
        if not isinstance(target_ids, list):
            target_ids = [target_ids]
            
        gref_grid_points = [] 
        
        if not g.get('no_target'):
            for aid in target_ids:
                if aid in precomputed_points:
                    gref_grid_points.append(precomputed_points[aid])
                else:
                    gref_grid_points.append([])

        new_g['grid_points'] = gref_grid_points
        new_grefs.append(new_g)

    with open(output_grefs_path, 'w') as f:
        json.dump(new_grefs, f, indent=2)
        
    print(f"Processed complete. Saved final merged grefs to {output_grefs_path}")

if __name__ == "__main__":
    process_data()
