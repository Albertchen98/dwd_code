import os
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm import tqdm

# --- 配置 ---
MODEL_DIR = "/starmap/nas/model_hubs/models--openai--clip-vit-large-patch14"
Real_VIDEO_FOLDER = "/starmap/nas/dataset/processed/processed_nuplan/render_testing/videos/pinhole_front/"
CG_VIDEO_FOLDER = "/starmap/nas/dataset/processed/CarlaData30hr/videos/"
SAMPLE_INTERVAL = 15

# 候选池
# 专注于物理光学、传感器特性和环境随机性
POS_LIST = [
    "Authentic dashcam footage, street photo",                   # 总体描述
    "Natural sunlight with complex soft shadows",                 # 光照
    "Weathered road textures and organic grime",                 # 纹理
    "Realistic camera vibration and road bump",                  # 抖动
    "Sensor noise and natural motion smear"                      # 曝光/噪声
]

NEG_LIST = [
    "Video game, CGI render, virtual world",                     # 总体描述
    "Simplified lighting with static baked shadows",             # 光照
    "Perfectly smooth surfaces, repetitive patterns",            # 纹理
    "Stiff movement, rigid-body physics",                        # 抖动
    "Clean digital image, artificial sharpness"                  # 曝光/噪声
]

def get_features(model, processor, text_list, device):
    with torch.no_grad():
        inputs = processor(text=text_list, return_tensors="pt", padding=True).to(device)
        feat = model.get_text_features(**inputs)
        return feat / feat.norm(dim=-1, keepdim=True)

def analyze_pairwise_diffs():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL_DIR).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_DIR)
    
    pos_feat = get_features(model, processor, POS_LIST, device) # [N_pos, D]
    neg_feat = get_features(model, processor, NEG_LIST, device) # [N_neg, D]
    
    video_files = glob(os.path.join(VIDEO_FOLDER, "*.mp4"))[::10]
    
    # 存储所有帧的差值矩阵
    diff_accumulator = np.zeros((len(POS_LIST), len(NEG_LIST)))
    frame_total = 0

    for v_path in tqdm(video_files, desc="分析视频两两差异"):
        cap = cv2.VideoCapture(v_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if frame_total % SAMPLE_INTERVAL == 0:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                inputs = processor(images=img, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    img_f = model.get_image_features(**inputs)
                    img_f /= img_f.norm(dim=-1, keepdim=True)
                    
                    s_pos = (img_f @ pos_feat.T).cpu().numpy() # [1, N_pos]
                    s_neg = (img_f @ neg_feat.T).cpu().numpy() # [1, N_neg]
                    
                    # 两两计算差异 (Similarity Difference)
                    # 使用广播机制: [N_pos, 1] - [1, N_neg] -> [N_pos, N_neg]
                    current_diff = s_pos.T / s_neg 
                    diff_accumulator += current_diff
            frame_total += 1
        cap.release()

    avg_diff_matrix = diff_accumulator / (frame_total / SAMPLE_INTERVAL)

    # --- 可视化 ---
    plt.figure(figsize=(14, 10))
    # 使用 RdBu_r 调色板，红色表示正差异大（Pos显著好于Neg），蓝色表示负差异
    sns.heatmap(avg_diff_matrix, annot=True, fmt=".4f", cmap="RdBu_r", center=1,
                xticklabels=[n[:25] for n in NEG_LIST], 
                yticklabels=[p[:25] for p in POS_LIST])
    
    plt.title("Pairwise Score Difference Matrix: Sim(Video, Pos) - Sim(Video, Neg)")
    plt.xlabel("Negative Prompts")
    plt.ylabel("Positive Prompts")
    plt.tight_layout()
    plt.savefig("pairwise_diff_analysis_CG.png")
    
    # 找到最强差异对
    i, j = np.unravel_index(avg_diff_matrix.argmax(), avg_diff_matrix.shape)
    print(f"\n最佳判别组合 (最大距离):")
    print(f"Positive: {POS_LIST[i]}")
    print(f"Negative: {NEG_LIST[j]}")
    print(f"最大 Score Margin: {avg_diff_matrix[i, j]:.4f}")

def compute_ratio_matrix(video_folder, model, processor, pos_feat, neg_feat, device, label="Video"):
    video_files = glob(os.path.join(video_folder, "*.mp4"))[::10]
    accumulator = np.zeros((len(POS_LIST), len(NEG_LIST)))
    count = 0
    
    for v_path in tqdm(video_files, desc=f"Processing {label}"):
        cap = cv2.VideoCapture(v_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if count % SAMPLE_INTERVAL == 0:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                inputs = processor(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    img_f = model.get_image_features(**inputs)
                    img_f /= img_f.norm(dim=-1, keepdim=True)
                    s_pos = (img_f @ pos_feat.T).cpu().numpy() # [1, N_pos]
                    s_neg = (img_f @ neg_feat.T).cpu().numpy() # [1, N_neg]
                    # 计算比率矩阵 [N_pos, N_neg]
                    accumulator += (s_pos.T / np.clip(s_neg, 1e-6, None))
            count += 1
        cap.release()
    return accumulator / (count / SAMPLE_INTERVAL)

def select_best_prompts():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL_DIR).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_DIR)
    
    pos_feat = get_features(model, processor, POS_LIST, device)
    neg_feat = get_features(model, processor, NEG_LIST, device)

    # 1. 分别计算 Real 和 CG 的表现
    print("开始分析真实视频 (Real)...")
    m_real = compute_ratio_matrix(Real_VIDEO_FOLDER, model, processor, pos_feat, neg_feat, device, "Real")
    
    print("开始分析合成视频 (CG)...")
    m_cg = compute_ratio_matrix(CG_VIDEO_FOLDER, model, processor, pos_feat, neg_feat, device, "CG")

    # 2. 计算判别增益 (Discriminative Gain)
    # Gain 越大，说明这对 Prompt 在 Real 上的优势相对于 CG 越明显
    gain_matrix = m_real / np.clip(m_cg, 1e-6, None)

    # 3. 可视化增益矩阵
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(gain_matrix, annot=True, fmt=".3f", cmap="magma")
    plt.title("Prompt Pair Selection: Discriminative Gain (Real_Ratio / CG_Ratio)")
    plt.xlabel("Negative Prompts")
    plt.ylabel("Positive Prompts")
    plt.tight_layout()
    plt.savefig("prompt_selection_gain.png")

    # 4. 挑选最优 Top-K 组合
    # 扁平化索引并排序
    flat_idx = gain_matrix.flatten().argsort()[::-1]
    
    print("\n" + "="*80)
    print(f"{'Rank':<5} | {'Gain':<8} | {'Positive Prompt':<35} | {'Negative Prompt'}")
    print("-" * 80)
    for i in range(5):
        p_idx, n_idx = divmod(flat_idx[i], len(NEG_LIST))
        print(f"{i+1:<5} | {gain_matrix[p_idx, n_idx]:.4f} | {POS_LIST[p_idx][:33]:<35} | {NEG_LIST[n_idx]}")
    print("="*80)

if __name__ == "__main__":
    select_best_prompts()