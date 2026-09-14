import torch
import os
from pathlib import Path

# 设置路径
base_path = "my_experiments/exp3_mid_filter/Town10HD_Route0044_1_mean_k13/features"

# 加载所有帧
frames = []
for i in range(1, 94):  # 从 00000001 到 00000093
    # frame_0001.pt ~ frame_0093.pt
    file_path = os.path.join(base_path, f"frame_{i:04d}.pt")
    frame = torch.load(file_path)  # 形状应该是 (C, H, W)
    frames.append(frame)

# 堆叠成 (C, T, H, W)
video_tensor = torch.stack(frames, dim=1)  # (C, T=93, H, W)

# 添加 batch 维度变成 (B, C, T, H, W)
video_tensor = video_tensor.unsqueeze(0)  # (1, C, T=93, H, W)
torch.save(video_tensor, os.path.join(base_path, "merged_dino_features_93.pt"))
print(f"最终形状: {video_tensor.shape}")