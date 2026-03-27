import json

def check_grefs():
    grefs_path = "/root/autodl-tmp/gRefCOCO/data/gRefCOCO/grefs(unc).json"
    with open(grefs_path, 'r') as f:
        grefs = json.load(f)
        
    print(f"Total grefs: {len(grefs)}")
    
    img_to_grefs = {}
    for g in grefs:
        img_id = g['image_id']
        if img_id not in img_to_grefs:
            img_to_grefs[img_id] = []
        img_to_grefs[img_id].append(g)
        
    print(f"Total unique images in grefs: {len(img_to_grefs)}")
    
    multi_gref_imgs = [img_id for img_id, g_list in img_to_grefs.items() if len(g_list) > 1]
    print(f"Images with multiple grefs (text queries): {len(multi_gref_imgs)}")
    
    if multi_gref_imgs:
        sample_img_id = multi_gref_imgs[0]
        print(f"\nSample Image ID with multiple grefs: {sample_img_id}")
        sample_grefs = img_to_grefs[sample_img_id]
        for idx, g in enumerate(sample_grefs):
            print(f"Gref {idx}: ann_id={g.get('ann_id')}, sentences: {[s['raw'] for s in g.get('sentences', [])]}")

if __name__ == "__main__":
    check_grefs()
