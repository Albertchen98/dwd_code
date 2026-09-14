"""Fit a reproducible, unwhitened DINOv3 PCA basis on training videos only."""
import argparse
import json
from pathlib import Path


def merge_covariance(count, mean, scatter, x):
    """Merge a batch into a centered covariance accumulator without storing past samples."""
    import torch
    batch_mean = x.mean(0)
    centered = x - batch_mean
    delta = batch_mean - mean
    total = count + len(x)
    scatter = scatter + centered.T @ centered + torch.outer(delta, delta) * (count * len(x) / total)
    mean = mean + delta * (len(x) / total)
    return total, mean, scatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-videos", type=int, default=1000)
    parser.add_argument("--patches-per-video", type=int, default=4096)
    args = parser.parse_args()
    if not 1 <= args.components <= 1024 or min(args.max_videos, args.patches_per_video) < 1:
        parser.error("Invalid PCA sample settings")
    import numpy as np
    import torch
    from decord import VideoReader, cpu
    from scripts.extract_dino_features import prepare_frames
    from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.defaults.dinov3_encoder import DINOV3Encoder
    rng = np.random.default_rng(args.seed)
    videos = sorted(args.video_dir.glob("*.mp4"))
    if not videos:
        raise ValueError("No training videos found")
    selected = rng.permutation(len(videos))[:args.max_videos]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in (f"patch_pca_mean_{args.components}.npy", f"patch_pca_components_{args.components}.npy", "pca_metadata.json"):
        if (args.output_dir / name).exists():
            raise FileExistsError(args.output_dir / name)
    encoder = DINOV3Encoder(checkpoint_dir=str(args.dino_checkpoint), height=2816, width=5120,
                            use_processor=False, use_dino_pca=False, forward_chunk_size=1).eval()
    # Merge per-video centered scatter matrices; memory is O(D^2), independent of dataset size.
    count = 0
    mean = torch.zeros(1024, dtype=torch.float64)
    scatter = torch.zeros((1024, 1024), dtype=torch.float64)
    samples = []
    for index in selected:
        video = videos[int(index)]
        reader = VideoReader(str(video), ctx=cpu(0), num_threads=2)
        frame_id = int(rng.integers(len(reader)))
        frames = torch.from_numpy(reader.get_batch([frame_id]).asnumpy()).permute(0, 3, 1, 2)
        with torch.inference_mode():
            features = encoder(prepare_frames(frames).cuda())
            flat = features[0, :, 0].permute(1, 2, 0).reshape(-1, 1024)
            ids = rng.choice(len(flat), min(len(flat), args.patches_per_video), replace=False)
            x = flat[torch.as_tensor(ids, device=flat.device)].cpu().double()
        count, mean, scatter = merge_covariance(count, mean, scatter, x)
        samples.append(dict(video=video.name, frame=frame_id))
        del features, flat, x
        print(f"PCA samples: {count}", flush=True)
    if count <= args.components:
        raise ValueError("Not enough sampled patches for the requested PCA dimension")
    eigenvalues, eigenvectors = torch.linalg.eigh(scatter / (count - 1))
    components = eigenvectors[:, -args.components:].flip(1).T.contiguous()
    # Resolve arbitrary eigenvector signs for repeatability.
    pivots = components.abs().argmax(1)
    signs = components[torch.arange(args.components), pivots].sign()
    components *= signs[:, None]
    np.save(args.output_dir / f"patch_pca_mean_{args.components}.npy", mean.float().numpy())
    np.save(args.output_dir / f"patch_pca_components_{args.components}.npy", components.float().numpy())
    metadata = dict(seed=args.seed, samples=samples, count=count, scale=4, layernorm="pretrained",
                    whitening=False, dino_checkpoint=str(args.dino_checkpoint),
                    eigenvalues=eigenvalues[-args.components:].flip(0).tolist())
    (args.output_dir / "pca_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
