import argparse
import os
import sys

import torch
import torch.distributed as dist
from tqdm import tqdm

from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.experiment.experiment_list import EXPERIMENTS
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.experiment_av.experiment_list import (
    EXPERIMENTS as EXPERIMENTS_AV,
)
from PIL import Image
import numpy as np
from cosmos_transfer2._src.transfer2.inference.arg_parser import parse_arguments
from cosmos_transfer2._src.transfer2.inference.inference_pipeline_img_dino import Control2ImageInference
from cosmos_transfer2._src.transfer2.inference.utils import (
    _IMAGE_EXTENSIONS,
    _VIDEO_EXTENSIONS,
    color_message,
    get_prompt_from_path,
    get_unique_seed,
    parse_control_input_file_paths,
    parse_control_input_single_file_paths,
    validate_image_context_path,
)

def generate_save_path(
    save_dir: str,
    guidance: int,
    iter_num: str,
    seed: int,
    hint_key: str,
    control_weight: float,
    sigma_max: float,
    preset_edge_threshold: str,
    preset_blur_strength: str,
    video_name: str,
    ref_image_name: str = None,
    num_conditional_frames: int = 1,
    context_frame_idx: int = 0,
) -> str:
    """Generate save path for output video. Without extension suffix like .mp4."""

    # Build the path components
    path_components = [save_dir]

    # Add reference image folder if applicable
    if ref_image_name:
        path_components.append(f"ref_img_{ref_image_name}")
    if context_frame_idx is not None:
        path_components.append(f"ref_img_frame_idx_{context_frame_idx}")

    # Add guidance and iteration info
    path_components.append(f"guidance{guidance}_iter{iter_num}_seed{seed}")

    # Add control settings
    control_info = f"{hint_key.replace(',', '+')}_cw{control_weight}"
    if sigma_max is not None:
        control_info += f"_smax{sigma_max}"
    if num_conditional_frames != 1:
        control_info += f"_overlap{num_conditional_frames}"
    if "edge" in hint_key:
        preset_str = preset_edge_threshold.replace("_", "")
        control_info += f"_edge-{preset_str}"
    if "vis" in hint_key:
        preset_str = preset_blur_strength.replace("_", "")
        control_info += f"_vis-{preset_str}"
    path_components.append(control_info)

    # Add base name
    path_components.append(video_name)

    return os.path.join(*path_components)


def process_single_video(
    image_path: str,
    prompt: str,
    neg_prompt: str | None,
    input_control_image_paths: dict,
    save_dir: str,
    guidance: int,
    inference_pipeline: "Control2ImageInference",
    args: argparse.Namespace,
    device_rank: int,
) -> None:
    """Process a single video with a single guidance value and single reference image."""
    # Generate save path
    image_name = os.path.basename(image_path).split(".")[0]
    save_path = os.path.join(save_dir, image_name)
    log.info(f"[Rank {device_rank}] save_path is defined as {save_path}")

    # Prepare inference arguments
    inference_kwargs = {
        "image_path": image_path,
        "prompt": prompt,
        "negative_prompt": neg_prompt,
        "guidance": guidance,
        "seed": args.seed,
        "resolution": args.resolution,
        "control_weight": args.control_weight,
        "sigma_max": args.sigma_max,
        "hint_key": args.hint_key.split(","),
        "num_steps": args.num_steps,
        "input_control_image_paths": input_control_image_paths,
        "shift": args.shift,
        "device_rank": device_rank,
    }
    # Run model inference
    output_img = inference_pipeline.generate_image(**inference_kwargs)
    
    # Only log "saving" messages, actual saving is done by each rank (assuming unique filenames per input)
    log.info(f"[Rank {device_rank}] saving image to {save_path}.jpg")
    
    # Remove batch dimension and normalize to [0, 1] range
    save_img(output_img, save_path)
    # save prompt
    # prompt_save_path = f"{save_path}.txt"
    # if not isinstance(prompt, list):  
    #     with open(prompt_save_path, "w") as f:
    #         f.write(prompt)
    # else:
    #     with open(prompt_save_path, "w") as f:
    #         for prompt_i in prompt:
    #             f.write(prompt_i + "\n")
    log.success(f"[Rank {device_rank}] Generated video saved to {save_path}.png")
    torch.cuda.empty_cache()


def save_img(output_img: torch.Tensor, save_path: str, default_ext: str = '.png'):
    """
    保存 BxCx1xHxW 尺寸的 PyTorch 张量图像。

    Args:
        output_img (torch.Tensor): 形状为 (B, C, 1, H, W) 的 PyTorch 张量。
        save_path (str): 保存输出文件的路径，可以是目录或带扩展名的文件路径。
        default_ext (str): 如果 save_path 中不包含扩展名，则使用的默认扩展名。
    """
    # --- 1. 形状处理 ---
    B, C, T, H, W = output_img.shape
    
    # 移除尺寸为 1 的时间维度 (索引 2)
    img_batch = output_img.squeeze(2) 
    
    # --- 2. 预处理数据类型 ---
    # 将张量移动到 CPU，并转换为 NumPy 数组
    img_np_batch = img_batch.cpu().numpy()
    
    # 转换为 uint8，假设浮点数范围在 [0, 1]
    if img_np_batch.dtype == np.float32 or img_np_batch.dtype == np.float64:
        img_np_batch = (img_np_batch * 255).astype(np.uint8)
    
    # --- 3. 确定保存目录和文件名前缀 ---
    
    base_dir = os.path.dirname(save_path)
    base_name_with_ext = os.path.basename(save_path)
    
    # 获取文件名和扩展名
    base_name, ext = os.path.splitext(base_name_with_ext)
    
    # 确定最终的目录和前缀
    if not ext:
        # 如果 save_path 没有扩展名 (例如: 'outputs' 或 'outputs/my_run')
        if base_dir and not base_name:
            # save_path 是纯目录，例如 'outputs/'
            final_dir = base_dir
            final_prefix = "output"
        elif not base_dir and not base_name:
            # save_path 是 '.' 或 ''
            final_dir = '.'
            final_prefix = "output"
        else:
            # save_path 是目录/前缀, 例如 'outputs/my_run'
            final_dir = base_dir if base_dir else '.'
            final_prefix = base_name
            
        final_ext = default_ext
        os.makedirs(final_dir, exist_ok=True)
        
    else:
        # 如果 save_path 包含扩展名 (例如: 'outputs/my_file.png')
        final_dir = base_dir if base_dir else '.'
        final_prefix = base_name
        final_ext = ext
        os.makedirs(final_dir, exist_ok=True)
        
    
    # --- 4. 循环处理批次中的每张图像 ---
    for i in range(B):
        single_img_np = img_np_batch[i]
        
        # --- 5. 维度转换 (C, H, W) -> (H, W, C) for PIL ---
        mode = None
        if C == 1:
            # 灰度图：(1, H, W) -> (H, W)
            single_img_np = single_img_np.squeeze(0)
            mode = 'L'
        elif C == 3:
            # RGB 图：(3, H, W) -> (H, W, 3)
            single_img_np = np.transpose(single_img_np, (1, 2, 0))
            mode = 'RGB'
        else:
            print(f"Skipping image {i}: Unsupported channel count C={C}")
            continue

        # --- 6. 构造文件名并保存 ---
        
        # 当 B > 1 或 save_path 没有扩展名时，需要添加索引
        if B > 1 or not ext:
            output_filename = os.path.join(final_dir, f"{final_prefix}_{i:03d}{final_ext}")
        else:
            # B=1 且 save_path 包含扩展名时，直接使用原始路径
            output_filename = save_path

        try:
            pil_img = Image.fromarray(single_img_np, mode=mode)
            pil_img.save(output_filename)
            print(f"Image {i} saved successfully to: {output_filename}")
        except Exception as e:
            print(f"Error saving image {i} to {output_filename}: {e}")

def main() -> None:
    args = parse_arguments()
    torch.manual_seed(args.seed)

    registered_exp_name=args.experiment
    exp_override_opts = args.exp_override_opts
    
    # --- Distributed Init for Data Parallelism ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    print(f"local rank: {local_rank}")

    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        print(f"Initialized Distributed Process Group: Rank {rank}/{world_size}, Local Rank {local_rank}")
    else:
        print("Running in Single GPU mode.")

    device_rank = local_rank
    
    # NOTE: For data parallelism (each GPU has a full model), we pass process_group=None
    # This tells the model NOT to try to split itself across GPUs.
    process_group = None

    # Initialize the inference class
    inference_pipeline = Control2ImageInference(
        registered_exp_name=registered_exp_name,
        s3_credential_path=args.s3_cred,
        exp_override_opts=exp_override_opts,
        process_group=process_group, 
        cache_dir=args.cache_dir,
        checkpoint_paths=args.ckpt_paths.split(",") if args.ckpt_paths else None,
        skip_load_model=args.skip_load_model,
        base_load_from=args.base_load_from,
    )

    # Create save directory structure (Only rank 0 creates directory to avoid race condition)
    save_dir = os.path.join(args.save_root, args.experiment)
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
    
    # Wait for Rank 0 to create directory
    if world_size > 1:
        dist.barrier()

    # Prepare reference image info if available
    # Process all videos from folder if specified
    if args.video_folder:
        prompt_folder = args.prompt_folder if args.prompt_folder != "" else args.video_folder

        video_files = [
            f
            for f in sorted(os.listdir(args.video_folder))
            if os.path.splitext(f)[1] in _VIDEO_EXTENSIONS + _IMAGE_EXTENSIONS
        ]
        
        # --- Data Splitting Logic ---
        # Each rank takes a subset of files: [0, 1, 2, 3, 4, 5, 6, 7] -> Rank 0 gets [0, 4], Rank 1 gets [1, 5]...
        all_files_count = len(video_files)
        video_files_to_process = video_files[rank::world_size]
        
        if args.limit_num_videos:
            # Apply limit AFTER splitting or BEFORE? 
            # Usually users want to limit total files. Let's limit total first for consistency.
            video_files = video_files[: args.limit_num_videos]
            video_files_to_process = video_files[rank::world_size]

        print(f"[Rank {rank}] Processing {len(video_files_to_process)}/{all_files_count} files.")

        # Use position to avoid progress bars overlapping in terminal
        for video_file in tqdm(video_files_to_process, desc=f"Rank {rank} Processing", position=rank):
            video_path = os.path.join(args.video_folder, video_file)

            # prompt, neg_prompt = get_prompt_from_path(prompt_path, args.prompt)
            prompt, neg_prompt = get_prompt_from_path(args.prompt_path, args.prompt)
            if not neg_prompt:
                neg_prompt = args.negative_prompt
            
            # Only log prompt on Rank 0 to reduce clutter
            if device_rank == 0:
                log.info(color_message(f"Prompt: {prompt}", "grey"))

            input_control_video_paths = parse_control_input_file_paths(
                args.input_control_folder_edge,
                args.input_control_folder_vis,
                args.input_control_folder_depth,
                args.input_control_folder_seg,
                args.input_control_folder_dino,
                args.input_control_folder_edge_mask,
                args.input_control_folder_vis_mask,
                args.input_control_folder_depth_mask,
                args.input_control_folder_seg_mask,
                args.input_control_folder_dino_mask,
                args.input_control_folder_inpaint_mask,
                video_file,
            )

            if args.seed is None:
                args.seed = get_unique_seed(
                    video_path, args.save_root, args.experiment, args.ckpt_iter, args.num_conditional_frames
                )

            # Process the video with all guidance values
            for guidance in args.guidance:
                process_single_video(
                    image_path=video_path,
                    prompt=prompt,
                    neg_prompt=neg_prompt,
                    input_control_image_paths=input_control_video_paths,
                    save_dir=save_dir,
                    guidance=guidance,
                    inference_pipeline=inference_pipeline,
                    args=args,
                    device_rank=device_rank,
                )

    # Process a single video if specified
    elif args.video_path:
        # In data parallel mode, single video path is tricky. 
        # Usually we only want Rank 0 to run it, OR we want all ranks to run it with different seeds.
        # Here we assume only Rank 0 should run it to avoid overwriting files.
        if rank == 0:
            prompt, neg_prompt = get_prompt_from_path(args.prompt_path, args.prompt)
            if not neg_prompt:
                neg_prompt = args.negative_prompt
            log.info(color_message(f"Prompt: {prompt}", "grey"))

            input_control_video_paths = parse_control_input_single_file_paths(
                input_control_video_path_edge=args.input_control_video_path_edge,
                input_control_video_path_vis=args.input_control_video_path_vis,
                input_control_video_path_depth=args.input_control_video_path_depth,
                input_control_video_path_seg=args.input_control_video_path_seg,
                input_control_video_path_dino=args.input_control_video_path_dino,
                input_control_video_path_edge_mask=args.input_control_video_path_edge_mask,
                input_control_video_path_vis_mask=args.input_control_video_path_vis_mask,
                input_control_video_path_depth_mask=args.input_control_video_path_depth_mask,
                input_control_video_path_seg_mask=args.input_control_video_path_seg_mask,
                input_control_video_path_dino_mask=args.input_control_video_path_dino_mask,
                input_control_video_path_inpaint_mask=args.input_control_video_path_inpaint_mask,
            )
            log.info(f"input_control_video_paths are {input_control_video_paths}")
            
            for guidance in args.guidance:
                process_single_video(
                    image_path=args.video_path,
                    prompt=prompt,
                    neg_prompt=neg_prompt,
                    input_control_image_paths=input_control_video_paths,
                    save_dir=save_dir,
                    guidance=guidance,
                    inference_pipeline=inference_pipeline,
                    args=args,
                    device_rank=device_rank,
                )
        else:
            # Other ranks just wait or exit
            pass

    else:
        raise ValueError("Either --video_folder or --video_path must be specified")

    # clean up properly
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()