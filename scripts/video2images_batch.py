import cv2
import os

def convert_videos_to_images(video_root, output_subdir="images"):
    """
    读取指定目录下所有视频，将其转换为图像并保存
    """
    # 1. 设置并创建输出路径
    output_path = os.path.join(video_root, output_subdir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"已创建输出目录: {output_path}")

    # 支持的视频后缀
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')
    
    # 获取目录下所有文件
    all_files = os.listdir(video_root)
    video_files = [f for f in all_files if f.lower().endswith(video_extensions)]

    if not video_files:
        print("未在该目录下找到任何视频文件。")
        return

    print(f"找到 {len(video_files)} 个视频，准备开始转换...")

    for video_name in sorted(video_files):
        video_full_path = os.path.join(video_root, video_name)
        # 获取不带后缀的文件名作为前缀
        video_prefix = os.path.splitext(video_name)[0]
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_full_path)
        if not cap.isOpened():
            print(f"  [!] 无法打开视频: {video_name}")
            continue

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 2. 构造图像文件名：视频文件名 + _ + 6位帧序号 (例如: Video01_000001.jpg)
            # 使用 zfill(6) 保证序号对齐，方便后续你代码中的前缀匹配
            img_name = f"{video_prefix}_{str(frame_count).zfill(6)}.jpg"
            img_save_path = os.path.join(output_path, img_name)
            if os.path.exists(img_save_path): continue
            
            # 3. 保存图像
            cv2.imwrite(img_save_path, frame)
            frame_count += 1

        cap.release()
        print(f"  [√] 完成: {video_name} -> 提取了 {frame_count} 帧")

    print("\n所有任务处理完毕！")

# --- 执行脚本 ---
target_root = "/starmap/nas/dataset/processed/CarlaData30hr/fresco_result"
convert_videos_to_images(target_root)