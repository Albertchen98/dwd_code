import numpy as np
from PIL import Image
import os
from pathlib import Path

def visualize_label(label_map):
    # 定义色表 (19个类别 + 1个填充色)
    palette = np.array([
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
        [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
        [0, 80, 100], [0, 0, 230], [119, 11, 32], [255, 255, 255]
    ], dtype=np.uint8)

    label_indices = label_map.astype(np.int64)
    color_image = palette[label_indices]
    return color_image

def process_folder(input_dir, output_dir=None):
    # 如果没有指定输出路径，则默认保存在原路径
    if output_dir is None:
        output_dir = input_dir
    
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 遍历文件夹中所有 .tif 图片 (你可以换成 .png, .jpg)
    extensions = ('*.tif', '*.tiff')
    files = []
    for ext in extensions:
        files.extend(Path(input_dir).glob(ext))

    print(f"找到 {len(files)} 个文件，开始处理...")

    for img_path in files:
        # 读取图片
        img = Image.open(img_path)
        img_array = np.array(img)

        # 转换为彩色图
        color_img = visualize_label(img_array)

        # 构建保存路径: 文件名_vis.png
        save_name = f"{img_path.stem}_vis.png"
        save_path = os.path.join(output_dir, save_name)

        # 保存图片
        Image.fromarray(color_img).save(save_path)
        print(f"已保存: {save_name}")

    print("全部处理完成！")

# --- 使用示例 ---
input_folder = '/starmap/nas/dataset/processed/CarlaData30hr/highresx4_pca8_conv_mergeframe/vid2vid_2B_control_720p_t24_dino_prep_control_layer4_cr1_embedding_rectified_flow_train_keyframe/guidance3_iter000022000_seed2025/dino_cw1.0/sempred/Town02_Route0049_0/'
# depth
input_folder = '/starmap/nas/dataset/processed/CarlaData30hr/cosmos_results/depth_result/sempred/Town02_Route0049_0/'

# pca8
# input_folder = '/starmap/nas/dataset/processed/CarlaData30hr/pca8_result/vid2vid_2B_control_720p_t24_dinocontrol_layer4_cr1_embedding_rectified_flow_train_keyframe/guidance3_iter000016000_seed2025/dino_cw1.0/sempred/Town05_Route0035_10/'
process_folder(input_folder)