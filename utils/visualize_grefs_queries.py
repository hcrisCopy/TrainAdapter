import os
import json
from PIL import Image, ImageDraw, ImageFont

def batch_visualize_queries():
    grefs_path = "/root/autodl-tmp/Data/grefs_with_grids.json"
    grid_img_dir = "/root/autodl-tmp/Data/grid_images"
    output_dir = "/root/autodl-tmp/Data/vis_grefs_output"
    
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(grefs_path):
        print(f"找不到 {grefs_path}！")
        return

    print("加载 JSON 数据...")
    with open(grefs_path, 'r') as f:
        grefs = json.load(f)

    # 有许多gref可能对应的是同一张图，如果一张一张图去保存会导致覆盖。
    # 按照实际查询任务，不同的查询应该生成不同的可视化图，为了避免重名，保存格式为 "原图名_gref编号.jpg"
    
    # 尝试加载可缩放的系统字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        large_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
        large_font = ImageFont.load_default()

    print("开始批量处理查询并可视化...")
    count = 0
    skipped = 0
    colors = ['red', 'blue', 'green', 'magenta', 'cyan', 'yellow', 'orange', 'purple', 'deeppink']

    # 我们设定一个可视化的总数限制，因为有94050条gref，如果全部生成图片会导致IO爆炸。
    # 用户可以在脚本里改这个值，暂时只前生成50张做样例和验证。
    MAX_VISUALIZE = 50 
    
    # 找到所有的带网格的文件，以便不空读取
    valid_files = set(os.listdir(grid_img_dir))

    for g_idx, g in enumerate(grefs):
        if count >= MAX_VISUALIZE:
            break
            
        filename = g.get('file_name')
        if not filename or filename not in valid_files:
            skipped += 1
            continue
            
        gref_points = g.get('grid_points', [])
        # 如果既不是 no_target 也没有坐标，我们跳过
        if not g.get('no_target') and not gref_points:
            skipped += 1
            continue

        img_path = os.path.join(grid_img_dir, filename)
        # 用原图名 + gref中的 ref_id 确保唯一保存
        output_name = f"vis_{filename.replace('.jpg','')}_ref{g.get('ref_id')}.jpg"
        output_path = os.path.join(output_dir, output_name)

        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                draw = ImageDraw.Draw(img)
                
                # 计算中心尺寸
                border_size = 28
                width, height = img.size
                new_width = width - border_size * 2
                new_height = height - border_size * 2
                
                if new_width <= 0 or new_height <= 0:
                    continue

                # 绘制每个目标的点集。 gref_points 结构是三层的 [[ [x,y], [x,y] ], [ [x,y] ]]
                # 外层是多个实体目标
                if not g.get('no_target'):
                    for tgt_idx, target_pts in enumerate(gref_points):
                        color = colors[tgt_idx % len(colors)]
                        for pt in target_pts:
                            i, j = pt
                            x = border_size + i * (new_width / 10.0)
                            y = border_size + j * (new_height / 10.0)
                            
                            r = 6
                            draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline='black', width=1)
                
                # 绘制顶部查询文本 (Sentences)
                sentences = g.get('sentences', [])
                texts_to_draw = []
                
                # 提取出主要的查询
                for s in sentences:
                    texts_to_draw.append(s.get('raw', ''))
                
                # 准备文本画板 (放在图片底部白边区域或直接叠在图像左上角)
                # 我们将其醒目地绘制在于顶部
                y_text = 10
                draw.text((10, y_text), f"Queries (Ref {g.get('ref_id')}):", fill='red', font=large_font)
                y_text += 25
                
                if g.get('no_target'):
                    draw.text((10, y_text), "[ NO TARGET ]", fill='blue', font=large_font)
                    y_text += 25
                    
                for txt in texts_to_draw:
                    # 使用描边提高可读性
                    draw.text((10, y_text), f"- {txt}", fill='red', font=font,
                             stroke_width=1, stroke_fill='white')
                    y_text += 25
                        
                img.save(output_path)
                count += 1
                    
        except Exception as e:
            print(f"处理 {filename} (Ref {g.get('ref_id')}) 时出错: {e}")

    print(f"\n批量可视化样例生成完成！共成功生成 {count} 张独立的文本查询对应图。")
    print(f"保存在: {output_dir}")
    print(f"(注：为了防止磁盘撑爆，当前限制最多生成 {MAX_VISUALIZE} 张，可在代码中调节 MAX_VISUALIZE)")

if __name__ == '__main__':
    batch_visualize_queries()
