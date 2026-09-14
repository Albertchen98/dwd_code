"""Extract all video frames to unmasked x4 PCA-32 caches for DwD training."""
import argparse
import json
from pathlib import Path


def prepare_frames(frames):
    """Match nuplan Dataset._get_frames and its ImageNet input normalization."""
    import torch
    from torchvision.transforms import v2
    from cosmos_transfer2._src.predict2.datasets.local_datasets.dataset_utils import ResizePreprocess, ToTensorVideo
    frames = ResizePreprocess((704, 1280))(ToTensorVideo()(frames))
    frames = (frames * 255).clamp(0, 255).to(torch.uint8)
    transform = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.bfloat16, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(frames).permute(1, 0, 2, 3).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--pca-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1, help="Frames per encoder call")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Invalid batch size or shard selection")
    import torch
    from decord import VideoReader, cpu
    from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.defaults.dinov3_encoder import DINOV3Encoder
    encoder = DINOV3Encoder(
        checkpoint_dir=str(args.dino_checkpoint), height=2816, width=5120,
        use_processor=False, use_dino_pca=True, use_random_channel=False,
        pca_mean_path=str(args.pca_dir / "patch_pca_mean_32.npy"),
        pca_comp_path=str(args.pca_dir / "patch_pca_components_32.npy"),
        forward_chunk_size=args.batch_size,
    ).eval()
    if encoder.pca_comp.shape != (32, 1024):
        raise ValueError("Expected PCA components [32,1024] for DINOv3 ViT-L/16")
    videos = sorted(args.video_dir.glob("*.mp4"))[args.shard_index::args.num_shards]
    if not videos:
        raise ValueError("No videos in the selected shard")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for video in videos:
        destination = args.output_dir / f"{video.stem}.pt"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}; select a new output directory")
        reader = VideoReader(str(video), ctx=cpu(0), num_threads=2)
        output = torch.empty((1, 32, len(reader), 176, 320), dtype=torch.bfloat16)
        with torch.inference_mode():
            for start in range(0, len(reader), args.batch_size):
                end = min(start + args.batch_size, len(reader))
                frames = torch.from_numpy(reader.get_batch(list(range(start, end))).asnumpy()).permute(0, 3, 1, 2)
                features = encoder(prepare_frames(frames).cuda())
                output[:, :, start:end] = features.cpu()
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(output, temporary)
        temporary.replace(destination)
        import hashlib
        metadata = dict(video=video.name, shape=list(output.shape), fps=float(reader.get_avg_fps()),
                        dino_checkpoint=str(args.dino_checkpoint), layernorm="pretrained",
                        scale=4, tail_drop=False, temporal_downsample=False)
        for name in ("patch_pca_mean_32.npy", "patch_pca_components_32.npy"):
            metadata[name] = hashlib.sha256((args.pca_dir / name).read_bytes()).hexdigest()
        destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Saved {destination}: {tuple(output.shape)}", flush=True)


if __name__ == "__main__":
    main()
