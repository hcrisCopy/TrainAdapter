import os
from PIL import Image

def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    count = 0
    if not os.path.exists(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        return

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                with Image.open(input_path) as img:
                    width, height = img.size
                    new_width = (width // 28) * 28
                    new_height = (height // 28) * 28
                    
                    if new_width == 0 or new_height == 0:
                        continue
                        
                    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    resized_img.save(output_path)
                    count += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
    print(f"一共处理了 {count} 张图片。")

if __name__ == "__main__":
    input_directory = "/root/autodl-tmp/gRefCOCO/Data/coco/train2014"
    output_directory = "/root/autodl-tmp/Data/images"
    process_images(input_directory, output_directory)
