import os
from PIL import Image, ImageDraw, ImageFont

def add_grid_to_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    border_size = 28 # 保证加上白边后仍是28的倍数
    
    # 尝试加载可缩放的系统字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except IOError:
        print("未找到指定字体，正使用无法调整大小的默认字体。建议安装系统字体。")
        font = ImageFont.load_default()
            
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
                    
                    # 创建带白边的新图像 (左、上分别加28像素的白边，右、下也加上保持整体对称和尺寸，也可以只加上左，但需要保证总是28的倍数)
                    # 此处我们在四周都加上28像素的白边
                    new_width = width + border_size * 2
                    new_height = height + border_size * 2
                    new_img = Image.new('RGB', (new_width, new_height), 'white')
                    
                    # 将原图粘贴到中间
                    new_img.paste(img, (border_size, border_size))
                    
                    draw = ImageDraw.Draw(new_img)
                    
                    # 添加网格线及数字
                    for i in range(11):
                        # x, y 坐标在原图像上的位置
                        x = border_size + i * (width / 10)
                        y = border_size + i * (height / 10)
                        
                        # 画垂直线
                        draw.line([(x, border_size), (x, border_size + height)], fill='black', width=1)
                        # 画水平线
                        draw.line([(border_size, y), (border_size + width, y)], fill='black', width=1)
                        
                        # 绘制顶部数字
                        text = str(i)
                        bbox = draw.textbbox((0, 0), text, font=font)
                        text_w = bbox[2] - bbox[0]
                        text_h = bbox[3] - bbox[1]
                        # 尽量靠近图像：y = border_size - text_h - 2
                        draw.text((x - text_w / 2, border_size - text_h - 5), text, fill='black', font=font)
                        
                        # 绘制左侧数字
                        # 尽量靠近图像：x = border_size - text_w - 2
                        draw.text((border_size - text_w - 5, y - text_h / 2), text, fill='black', font=font)
                        
                    new_img.save(output_path)
                    count += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
    print(f"一共处理了 {count} 张添加网格的图片。")

if __name__ == "__main__":
    input_directory = "/root/autodl-tmp/Data/images"
    output_directory = "/root/autodl-tmp/Data/grid_images"
    add_grid_to_images(input_directory, output_directory)
