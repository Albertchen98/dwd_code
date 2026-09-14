"""Launch DwD training with x4 DINO inputs, PCA tail drop and temporal downsampling."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def path_value(value):
    return json.dumps(str(Path(value).resolve()))


def build_command(args):
    if args.gpus < 1 or 24 % args.gpus:
        raise ValueError("gpus must be a positive divisor of the 24 latent frames")
    if min(args.max_iter, args.checkpoint_every) < 1 or args.workers < 0:
        raise ValueError("iteration counts must be positive and workers must be nonnegative")
    if args.feature_mode == "online" and (args.dino_checkpoint is None or args.pca_dir is None):
        raise ValueError("online mode requires --dino-checkpoint and --pca-dir")
    hubs = args.model_hubs.resolve()
    checkpoint = args.resume if args.resume else args.base_checkpoint
    values = [
        f"experiment=dwd_{args.feature_mode}",
        f"dataloader_train.dataset.dataset_dir={path_value(args.dataset)}",
        f"dataloader_val.dataset.dataset_dir={path_value(args.val_dataset or args.dataset)}",
        f"dataloader_train.num_workers={args.workers}",
        f"dataloader_val.num_workers={args.workers}",
        f"model_parallel.context_parallel_size={args.gpus}",
        f"model.config.fsdp_shard_size={args.gpus}",
        f"trainer.max_iter={args.max_iter}",
        f"trainer.seed={args.seed}",
        f"trainer.run_validation={str(args.val_dataset is not None).lower()}",
        f"checkpoint.save_iter={args.checkpoint_every}",
        f"checkpoint.load_path={path_value(checkpoint)}",
        f"checkpoint.load_training_state={str(args.resume is not None).lower()}",
        f"checkpoint.strict_resume={str(args.resume is not None).lower()}",
        f"model.config.tokenizer.vae_pth={path_value(hubs / 'nvidia/Cosmos-Predict2.5-2B/tokenizer.pth')}",
        f"model.config.text_encoder_config.ckpt_path={path_value(hubs / 'nvidia/Cosmos-Reason1-7B')}",
    ]
    if args.run_name:
        values.append(f"job.name={json.dumps(args.run_name)}")
    if args.feature_mode == "online":
        values += [
            f"model.config.dinov3_encoder.checkpoint_dir={path_value(args.dino_checkpoint)}",
            f"model.config.dinov3_encoder.pca_mean_path={path_value(args.pca_dir / 'patch_pca_mean_32.npy')}",
            f"model.config.dinov3_encoder.pca_comp_path={path_value(args.pca_dir / 'patch_pca_components_32.npy')}",
        ]
    return [sys.executable, "-m", "torch.distributed.run", "--standalone",
            f"--nproc_per_node={args.gpus}", "-m", "scripts.train",
            "--config=cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py",
            "--", *values, *args.overrides]


def check_inputs(args):
    """Check clip/embedding/cache correspondence without importing GPU dependencies."""
    errors = []
    required = [args.model_hubs / 'nvidia/Cosmos-Predict2.5-2B/tokenizer.pth']
    if args.resume:
        if not args.resume.is_dir() or not (args.resume / 'model').is_dir():
            errors.append(f"Expected DCP iteration directory containing model/: {args.resume}")
    elif not args.base_checkpoint.is_file():
        errors.append(f"Missing base checkpoint: {args.base_checkpoint}")
    if args.feature_mode == 'online':
        required += [args.dino_checkpoint, args.pca_dir / 'patch_pca_mean_32.npy',
                     args.pca_dir / 'patch_pca_components_32.npy']
    for path in required:
        if not path.exists():
            errors.append(f"Missing model asset: {path}")
    datasets = [args.dataset] + ([args.val_dataset] if args.val_dataset else [])
    if args.val_dataset and args.dataset.resolve() == args.val_dataset.resolve():
        errors.append("Validation must use a separate held-out dataset directory")
    for root in datasets:
        videos = sorted((root / 'videos/pinhole_front').glob('*.mp4'))
        if not videos:
            errors.append(f"No videos in {root / 'videos/pinhole_front'}")
        for video in videos:
            stem = video.stem
            text_dir = root / 'cosmos_reason_xxl'
            if not any((text_dir / (stem + ext)).is_file() for ext in ('.safetensors', '.pkl')):
                errors.append(f"Missing text embedding for {video.name} in {text_dir}")
            if args.feature_mode == 'offline' and not (root / 'dinov3_x4_pca32' / f'{stem}.pt').is_file():
                errors.append(f"Missing DINO feature cache for {video.name} in {root}")
    if errors:
        remaining = f"\n... and {len(errors)-20} more" if len(errors) > 20 else ''
        raise ValueError('\n'.join(errors[:20]) + remaining)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--val-dataset', type=Path, help='Held-out data; enables validation')
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument('--base-checkpoint', type=Path, help='Base pretrained .pt for a new run')
    checkpoint.add_argument('--resume', type=Path, help='DCP iteration directory for full-state resume')
    parser.add_argument('--model-hubs', type=Path, required=True)
    parser.add_argument('--feature-mode', choices=('offline', 'online'), default='offline')
    parser.add_argument('--dino-checkpoint', type=Path)
    parser.add_argument('--pca-dir', type=Path)
    parser.add_argument('--gpus', type=int, default=8)
    parser.add_argument('--max-iter', type=int, default=10000)
    parser.add_argument('--checkpoint-every', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, help='Overrides IMAGINAIRE_OUTPUT_ROOT')
    parser.add_argument('--run-name')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Print command without checking assets or launching')
    mode.add_argument('--check-only', action='store_true', help='Check file correspondence without launching')
    parser.add_argument('overrides', nargs='*', help='Advanced Hydra overrides appended last')
    return parser, parser.parse_args()


def main():
    parser, args = parse_args()
    try:
        command = build_command(args)
        if args.check_only and args.overrides:
            raise ValueError('--check-only checks launcher arguments; use explicit arguments instead of Hydra overrides')
        if not args.dry_run:
            check_inputs(args)
    except ValueError as exc:
        parser.error(str(exc))
    env = os.environ.copy()
    if args.output_dir:
        env['IMAGINAIRE_OUTPUT_ROOT'] = str(args.output_dir.resolve())
        print(f"IMAGINAIRE_OUTPUT_ROOT={shlex.quote(env['IMAGINAIRE_OUTPUT_ROOT'])}", flush=True)
    print(shlex.join(command), flush=True)
    if args.check_only:
        print('Input paths and clip correspondence OK. Tensor contents and GPU environment were not checked.')
    elif not args.dry_run:
        subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=env, check=True)


if __name__ == '__main__':
    main()
