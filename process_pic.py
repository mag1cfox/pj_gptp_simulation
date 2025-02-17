from PIL import Image
import os

# 定义输入文件夹和输出文件夹
input_folder = 'pic'  # 存放原始图片的文件夹
output_folder = 'pic2'  # 存放合并后图片的文件夹

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 每5张图片为一组
group_size = 5
total_images = 100  # 假设总共有100张图片

# 遍历每一组图片
for i in range(0, total_images, group_size):
    # 获取当前组的图片文件名
    group_images = [f"{j+1}_跳TE结果.jpg" for j in range(i, i + group_size)]

    # 打开每组中的图片
    images = []
    for img_name in group_images:
        img_path = os.path.join(input_folder, img_name)
        if os.path.exists(img_path):  # 确保图片存在
            images.append(Image.open(img_path))

    if not images:
        continue

    # 计算合并后图片的宽度和高度
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)

    # 创建一张空白图片
    combined_image = Image.new('RGB', (total_width, max_height))

    # 将每张图片粘贴到合并后的图片上
    x_offset = 0
    for img in images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    # 保存合并后的图片
    output_path = os.path.join(output_folder, f'combined_{i // group_size + 1}.jpg')
    combined_image.save(output_path)
    print(f'Saved: {output_path}')

print("所有图片合并完成！")