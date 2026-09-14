import cv2
import os

def extract_frames_multi_folders(folders_with_labels, prefix, frame_indices, base_output_dir):
    """
    遍历多个文件夹，找到匹配前缀的视频，并提取指定帧。
    """
    # 确保基础输出目录存在
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)

    for folder_path, label in folders_with_labels:
        # 清理 label 中的换行符，用作文件夹名
        safe_label = label.replace("\n", "_").replace("+", "plus")
        output_subdir = os.path.join(base_output_dir, safe_label)
        
        # 查找视频文件 (支持 mp4, avi, mkv 等)
        video_file = None
        for f in os.listdir(folder_path):
            if f.startswith(prefix) and f.endswith(('.mp4', '.avi', '.mkv', '.mov')):
                video_file = os.path.join(folder_path, f)
                break
        
        if not video_file:
            print(f"![跳过] 在目录 {folder_path} 中未找到前缀为 {prefix} 的视频")
            continue

        print(f"正在处理 [{label}] -> {video_file}")
        
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        # 执行提取
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"无法打开视频: {video_file}")
            continue

        for idx in sorted(set(frame_indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # 文件名包含前缀和帧号，方便后期对齐
                save_name = f"{prefix}_frame_{idx:04d}.png"
                save_path = os.path.join(output_subdir, save_name)
                cv2.imwrite(save_path, frame)
            else:
                print(f"无法读取帧: {idx}")
        
        cap.release()

# --- 配置参数 ---
folders_with_labels = [
    ("/starmap/nas/dataset/processed/CarlaData30hr/videos", "Original"),
    ("/starmap/nas/dataset/processed/CarlaData30hr/fresco_result/", "Fresco"),
    ("/starmap/nas/dataset/processed/CarlaData30hr/tclight_result/", "TCLight"),
    ("/starmap/nas/dataset/processed/CarlaData30hr/cosmos_results/blur_result/", "Cosmos\nBlur"),
    ("/starmap/nas/dataset/processed/CarlaData30hr/cosmos_results/edge_result/", "Cosmos\nEdge"),
    ("/starmap/nas/dataset/processed/CarlaData30hr/cosmos_results/depth_result/", "Cosmos\nDepth"),
    ("/starmap/nas/dataset/processed/CarlaData30hr/cosmos_results/depth_edge_result/", "Cosmos\nDepth+Edge"),
    ("/starmap/nas/dataset/processed/CarlaData30hr/cosmos_results/seg_result/", "Cosmos\nSeg"),
    ("/starmap/nas/dataset/processed/CarlaData30hr/pca8_result/vid2vid_2B_control_720p_t24_dinocontrol_layer4_cr1_embedding_rectified_flow_train_keyframe/guidance3_iter000016000_seed2025/dino_cw1.0/", "DwD\n(Ours)"), 
    ("/starmap/nas/dataset/processed/CarlaData30hr/highresx4_pca8_conv_mergeframe/vid2vid_2B_control_720p_t24_dino_prep_control_layer4_cr1_embedding_rectified_flow_train_keyframe/guidance3_iter000022000_seed2025/dino_cw1.0/", "DwD\n(22000)"),
    ("/starmap/nas/dataset/processed/CarlaData30hr/highresx4_pca8_conv_mergeframe/vid2vid_2B_control_720p_t24_dino_prep_control_layer4_cr1_embedding_rectified_flow_train_keyframe/guidance3_iter000030000_seed2025/dino_cw1.0/", "DwD\n(30000)"),
]

prefix_str = "Town07_Route0036_4"
frames = [i for i in range(7,54)]  # 你想要抽取的索引
output_root = f"./{prefix_str}_comparison_results"

if __name__ == "__main__":
    extract_frames_multi_folders(folders_with_labels, prefix_str, frames, output_root)
    print("\n所有任务已完成！")