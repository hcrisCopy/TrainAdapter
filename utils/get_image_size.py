from PIL import Image
import sys

def get_image_size(image_path):
    """
    根据图片路径获取图片的宽度和高度
    :param image_path: 图片路径（本地绝对/相对路径）
    :return: (宽度, 高度)
    """
    try:
        # 打开图片
        with Image.open(image_path) as img:
            width, height = img.size
            return width, height
    except Exception as e:
        print(f"错误：无法读取图片 -> {e}")
        return None

if __name__ == "__main__":
    # 判断是否传入了图片路径
    if len(sys.argv) < 2:
        print("使用方法：python get_image_size.py 图片路径")
        print("示例：python get_image_size.py test.jpg")
    else:
        img_path = sys.argv[1]
        size = get_image_size(img_path)
        if size:
            w, h = size
            print(f"图片宽度：{w} 像素")
            print(f"图片高度：{h} 像素")