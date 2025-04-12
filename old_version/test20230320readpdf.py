"""
**************************************
*  @Author  ：   mag1cfox
*  @Time    ：   2025/3/20 11:44
*  @Project :   pj_gptp_simulation
*  @Description :   从PDF文件中提取所有图像并以600 DPI的分辨率保存到指定文件夹：
*  @FileName:   test20230320readpdf.py
**************************************
"""
import os
import pdfplumber
from PIL import Image
import io


def extract_images_from_pdf(pdf_path, output_folder, dpi=600):
    """
    从PDF文件提取所有图像并保存到指定文件夹

    参数:
        pdf_path (str): PDF文件的路径
        output_folder (str): 保存图像的文件夹路径
        dpi (int): 输出图像的DPI分辨率
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开PDF文件
    with pdfplumber.open(pdf_path) as pdf:
        image_count = 0

        # 遍历PDF的每一页
        for page_num, page in enumerate(pdf.pages):
            # 获取页面上的图像
            images = page.images

            # 遍历页面上的每个图像
            for img_index, img in enumerate(images):
                # 获取图像数据
                image_bytes = img["stream"].get_data()

                try:
                    # 使用PIL打开图像
                    image = Image.open(io.BytesIO(image_bytes))

                    # 确定图像格式
                    format = image.format.lower() if image.format else 'png'

                    # 调整DPI (不改变像素大小，只改变元数据)
                    image.info['dpi'] = (dpi, dpi)

                    # 保存图像
                    image_filename = f"{output_folder}/image_{page_num + 1}_{img_index + 1}.{format}"
                    image.save(image_filename, dpi=(dpi, dpi))

                    image_count += 1
                    print(f"保存图像: {image_filename}")
                except Exception as e:
                    print(f"处理图像时出错: {e}")

    print(f"从PDF中提取了 {image_count} 张图像，保存到 {output_folder} 文件夹")


if __name__ == "__main__":
    pdf_path = input(r"D:\tmp\test.pdf")
    output_folder = input(r"D:\tmp")

    try:
        extract_images_from_pdf(pdf_path, output_folder, dpi=600)
    except Exception as e:
        print(f"发生错误: {e}")