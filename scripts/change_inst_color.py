import numpy as np
from PIL import Image
import os
from tqdm import tqdm  # 如果没安装，可以直接删除相关的代码

def batch_replace_color(input_dir, output_dir):
    # 1. 定义颜色和容差
    target_color = np.array([212, 0, 0])
    replacement_color = np.array([0, 255, 255])
    tolerance = 20  # 稍微调大一点点，确保边缘红边也能覆盖
    
    # 2. 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 3. 遍历文件
    files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    print(f"找到 {len(files)} 个待处理文件。")

    for filename in tqdm(files, desc="处理中"):
        # 读取图像
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path).convert('RGB')
        data = np.array(img)

        # 颜色匹配与替换
        distances = np.abs(data - target_color).sum(axis=-1)
        mask = distances <= tolerance
        data[mask] = replacement_color

        # 保存结果
        result_img = Image.fromarray(data)
        result_img.save(os.path.join(output_dir, filename))

    print(f"\n批量处理完成！结果保存在: {output_dir}")

# --- 配置路径 ---
base_path = '/home/xuyang/chuanheng/cosmos-transfer2.5-20260104/test_videos/1120_DV/inst_seg'
out_path = os.path.join(base_path, 'modified_vis') # 结果放在 inst_seg/modified_vis 下

batch_replace_color(base_path, out_path)