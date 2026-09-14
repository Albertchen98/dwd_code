import cv2
import os
import re
import glob

def create_video_from_frames(folder_path, output_video_path, fps=16.0):
    """
    将文件夹中的图片按照文件名中的索引排序，并串联成视频。
    
    Args:
        folder_path (str): 包含图片的文件夹路径。
        output_video_path (str): 输出视频的完整路径和文件名（例如: 'output.mp4'）。
        fps (float): 视频的帧率，默认为 16.0。
    """
    # 1. 定义匹配文件名的正则表达式，提取索引（00113）部分
    # 匹配 frame_后面跟着一组数字（索引），然后是_000.png 或其他扩展名
    # pattern: r'frame_(\d+)_000\.(png|jpg|jpeg)'
    # 
    # 考虑到用户给的例子，只提取第一个数字串
    file_pattern = os.path.join(folder_path, 'frame_*.png')
    all_files = glob.glob(file_pattern)

    if not all_files:
        print(f"Error: No .png files found in {folder_path} matching 'frame_*.png'")
        return

    # 2. 提取索引并排序
    # 存储 (索引, 文件路径)
    frames_info = []
    
    # 正则表达式用于提取第一个数字序列作为索引
    index_pattern = re.compile(r'frame_(\d+)_')
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        match = index_pattern.search(filename)
        
        if match:
            # 提取数字串并转换为整数进行排序
            index = int(match.group(1))
            frames_info.append((index, file_path))
        else:
            print(f"Warning: Skipped file due to naming mismatch: {filename}")

    # 按照提取的索引进行排序
    frames_info.sort(key=lambda x: x[0])
    sorted_frames = [info[1] for info in frames_info]

    if not sorted_frames:
        print("Error: Could not extract valid indices from any file.")
        return

    # 3. 读取第一张图片以获取尺寸
    first_frame = cv2.imread(sorted_frames[0])
    if first_frame is None:
        print(f"Error: Could not read the first frame: {sorted_frames[0]}")
        return
        
    height, width, layers = first_frame.shape
    size = (width, height)

    # 4. 配置 VideoWriter
    # 视频编码器 (FourCC)。'mp4v' 或 'XVID' 常用，但兼容性最好的通常是 'mp4v' (H.264)
    # 对于 .mp4 文件，通常使用 H.264 (即 'mp4v' 或 'avc1')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(output_video_path, fourcc, fps, size)

    # 5. 写入所有帧
    print(f"Processing {len(sorted_frames)} frames...")
    for frame_path in sorted_frames:
        frame = cv2.imread(frame_path)
        if frame is not None:
            video.write(frame)
        else:
            print(f"Warning: Failed to read frame {frame_path}")

    # 6. 释放资源
    video.release()
    print(f"\n✅ Video successfully created at: {output_video_path}")
    print(f"Total frames: {len(sorted_frames)}, FPS: {fps}")

# --- 使用示例 ---
if __name__ == '__main__':
    # ⚠️ 请将以下路径修改为您自己的文件夹路径和输出文件名
    
    # 示例: 假设图片在当前目录下的 'image_folder' 中
    # 并假设 'image_folder' 包含 frame_00100_000.png, frame_00101_000.png 等文件
# results/results/dinotok_0f_23t_13849332693800388551_960_000_980_000_0
    IMAGE_FOLDER = 'results/dinotok_4t64_23t64_v4_color_jitter_1120_DV_video_4_bf/img2img_2B_control_720p_dinocontrol_layer4_t5_embedding/' 
    OUTPUT_FILE = 'dinotok_4t64_23t64_v4_color_jitter_1120_DV_video_4_bf.mp4'
    FRAME_RATE = 4.0
    
    # 运行前，请先创建一些测试图片来模拟您的文件结构
    create_video_from_frames(IMAGE_FOLDER, OUTPUT_FILE, FRAME_RATE)
    # print("请将脚本中的 IMAGE_FOLDER 和 OUTPUT_FILE 变量修改为您的实际路径，并运行脚本。")