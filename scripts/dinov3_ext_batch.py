import torch
from transformers import DINOv3ViTImageProcessorFast, AutoModel, DINOv3ViTModel
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling
from safetensors.torch import save_file
from einops import rearrange
import os
import ray
from decord import VideoReader, cpu
from tqdm import tqdm
import click
from pathlib import Path
import multiprocessing as mp
import numpy as np
import cv2
import torch.nn.functional as F

NUMGPU=8
HEIGHT=704
WIDTH=1280
BATCHSIZE=24
MODEL_PATH = r"/cache/checkpoints/dinov3-vitl16"
NORMALIZE = True

@ray.remote(num_gpus=1)
class DinoV3Processor:
    def __init__(self):
        self.device = torch.device("cuda")
        self.processor = DINOv3ViTImageProcessorFast(
                                resample = PILImageResampling.BILINEAR,
                                image_mean = IMAGENET_DEFAULT_MEAN,
                                image_std = IMAGENET_DEFAULT_STD,
                                do_resize = True,
                                size={"height": HEIGHT, "width": WIDTH},
                                do_rescale = True,
                                do_normalize = True)
        self.model = AutoModel.from_pretrained(
                                MODEL_PATH, 
                                device_map="auto",
                                attn_implementation="flash_attention_2",
                                dtype=torch.bfloat16,)
        
        if NORMALIZE:
            self.model.norm.elementwise_affine = False
            self.model.norm.weight = None
            self.model.norm.bias = None

        self.model.eval()
    
    def process_batch(self, frames_batch):
        
        # Process images
        inputs = self.processor(images=frames_batch, return_tensors="pt").to(self.model.device)    
        # Extract features
        with torch.inference_mode():
            outputs = self.model(**inputs)
            # last_hidden_states = outputs.last_hidden_state[:,5:]
            last_hidden_states = outputs.last_hidden_state
            # features = last_hidden_states.reshape(-1, int(HEIGHT/16), int(WIDTH/16), last_hidden_states.shape[-1]) # b h w c
            # features = rearrange(features, "t h w c -> c t h w")
            
        
        return last_hidden_states.cpu()
    
def process_video(video_path, output_dir, processors):
    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
    n_frames = len(vr)
    frame_ids = list(range(0, n_frames))
    
    # Load all frames
    frames = vr.get_batch(frame_ids).asnumpy()
    
    # Split frames into batches
    frame_batches = [
        frames[i:i+BATCHSIZE] 
        for i in range(0, n_frames, BATCHSIZE)
    ]
    
    # Process batches in parallel
    results = []
    for batch in tqdm(frame_batches, desc=f"Processing {video_path.stem}"):
        # Round-robin assignment to processors
        processor = processors[len(results) % NUMGPU]
        results.append(processor.process_batch.remote(batch))
    
    # Gather all results
    features = torch.cat(ray.get(results), dim=0) # b h w c
    

    # pca_components_128 = np.load('pca_components_128.npy')
    # pca_mean_128 = np.load('pca_mean_128.npy')
    # features_centred = features - torch.from_numpy(pca_mean_128)[None,None,None]
    # features_tranf = features_centred @ torch.from_numpy(pca_components_128.T) # t h w c
    # # Save results
    output_file_path = os.path.join(output_dir, f"{video_path.stem}.safetensors")
    # # np.savez_compressed(output_npz_path, features=features)
    # features = rearrange(features, "t h w c -> c t h w")
    features = features.contiguous()
    # features_tranf = features_tranf.bfloat16().contiguous()
    save_file({"dino":features}, output_file_path)


@click.command()
@click.option("--input_folder", '-i', type=str, help="the input folder containing videos")
@click.option("--output_dir", '-o', type=str, help="the root folder of the output data")
def main(input_folder, output_dir):
    # Initialize Ray
    ray.init(num_gpus=NUMGPU)
    
    # Create processors
    processors = [DinoV3Processor.remote() for _ in range(NUMGPU)]
    
    input_folder = Path(input_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each video
    for video_path in input_folder.glob("*.mp4"):
        process_video(video_path, output_dir, processors)

    # Clean up
    ray.shutdown()

if __name__ == "__main__":
    main()
    
    
#python scripts/dinov3_ext_batch.py -i /cache/waymo/videos/pinhole_front -o /cache/waymo/dinov3-vitl16