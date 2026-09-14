#!/usr/bin/env python3
"""
Simple utility to convert saved DINO feature tensors (.pt) to a 3-channel RGB video using PCA.

Example:
  python scripts/visualize_dino_features.py \
    --input test_videos/dino_anyupx8_pca8_norm/13849332693800388551_960_000_980_000_0_10fps.pt \
    --out out.mp4 --resize 1280 704 --fps 10
"""
import argparse
import torch
import numpy as np
from PIL import Image
import imageio


def features_to_frames(feats: torch.Tensor, resize: tuple | None = None, use_first3: bool = False, channels: list | None = None, missing_behavior: str = 'error'):
    """Convert features tensor to a list of frames (uint8 HxWx3).

    Accepts shapes:
      - (B, C, T, H, W)
      - (C, T, H, W) -> treated as (1, C, T, H, W)
      - (T, H, W, C) -> converted to (1, C, T, H, W)

    Behavior:
      - If `channels` is provided (list of 3 ints), those channel indices are
        used directly as RGB channels. Indices support negative indexing.
        If any index is out of range, handling is controlled by `missing_behavior`:
           * 'error' (default) => raise RuntimeError
           * 'zero' => fill missing channels with zeros
           * 'clip' => clamp indices to [0, C-1]
           * 'mod' => wrap indices with modulo C
      - Else if `use_first3` is True: use first 3 channels (with padding if C<3).
      - Otherwise PCA is used to reduce to 3 channels.

    The output is scaled to [0,255]. Returns list of lists: one list of frames per batch element.
    """
    # Normalize input shape to (B,C,T,H,W)
    if isinstance(feats, np.ndarray):
        feats = torch.from_numpy(feats)
    if feats.ndim == 4:
        # either (C,T,H,W) or (T,H,W,C)
        C, T, H, W = feats.shape
        # heuristic: if first dim is small -> (C,T,H,W)
        if C <= 64:
            feats = feats.unsqueeze(0)
        else:
            # (T,H,W,C) -> permute
            feats = feats.permute(3, 0, 1, 2).unsqueeze(0)
    elif feats.ndim == 3:
        # (T,H,W) -> add channel
        T, H, W = feats.shape
        feats = feats.unsqueeze(0).unsqueeze(0)
    elif feats.ndim == 5:
        pass
    else:
        raise ValueError(f"Unexpected tensor shape: {feats.shape}")

    B, C, T, H, W = feats.shape

    # If explicit channels requested, use them (with missing behavior handling)
    if channels is not None:
        if len(channels) != 3:
            raise ValueError("`channels` must be a list of 3 integers")
        # resolve indices and build extraction tensor
        resolved = []
        for idx in channels:
            if idx < -C or idx >= C:
                if missing_behavior == 'error':
                    raise RuntimeError(f"Requested channel index {idx} out of range for C={C}. Use --missing-behavior to change this.")
                elif missing_behavior == 'zero':
                    resolved.append(None)
                elif missing_behavior == 'clip':
                    resolved.append(min(max(idx, 0), C-1))
                elif missing_behavior == 'mod':
                    resolved.append(idx % C)
                else:
                    raise ValueError(f"Unknown missing_behavior: {missing_behavior}")
            else:
                resolved.append(idx % C)

        # Build proj with possible zeros for missing
        channels_list = []
        for r in resolved:
            if r is None:
                channels_list.append(torch.zeros((B, 1, T, H, W), dtype=torch.float32))
            else:
                feat = feats[:, r:r+1, :, :, :].to(torch.float32)
                feat = (feat - feat.min()) / max(feat.max() - feat.min(), 1e-9)  # per-channel normalization
                channels_list.append(feat)
        stacked = torch.cat(channels_list, dim=1)
        proj = stacked.permute(0, 2, 3, 4, 1).cpu().numpy()  # (B,T,H,W,3)
        

    elif use_first3:
        # Use the first three channels directly as RGB (with simple handling
        # for C<3 by repeating channels or padding zeros). Cast to float32 to
        # avoid unsupported dtype like bfloat16 when converting to numpy.
        if C >= 3:
            proj = feats[:, :3, :, :, :].to(torch.float32).permute(0, 2, 3, 4, 1).cpu().numpy()  # (B,T,H,W,3)
        elif C == 2:
            a = feats[:, :2, :, :, :].to(torch.float32)
            third = torch.zeros((B, 1, T, H, W), dtype=torch.float32, device=feats.device)
            stacked = torch.cat([a, third], dim=1)
            proj = stacked.permute(0, 2, 3, 4, 1).cpu().numpy()
        elif C == 1:
            a = feats[:, 0:1, :, :, :].to(torch.float32)
            stacked = torch.cat([a, a, a], dim=1)
            proj = stacked.permute(0, 2, 3, 4, 1).cpu().numpy()
        else:
            # Fallback to zeros if no channels (shouldn't happen)
            proj = np.zeros((B, T, H, W, 3), dtype=np.float32)

        # Normalize globally to [0,1]
        mn = float(proj.min())
        mx = float(proj.max())
        proj = (proj - mn) / max(mx - mn, 1e-9)
    else:
        # PCA reduction to top-3 components
        X = feats.permute(0, 2, 3, 4, 1).reshape(-1, C).cpu().float()
        Xc = X - X.mean(dim=0, keepdim=True)
        # covariance (C x C)
        cov = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
        evals, evecs = torch.linalg.eigh(cov)
        pcs = evecs[:, -3:]  # top-3 components
        proj = (Xc @ pcs).reshape(B, T, H, W, 3).numpy()

        # Normalize globally to [0,1]
        mn = float(proj.min())
        mx = float(proj.max())
        proj = (proj - mn) / max(mx - mn, 1e-9)

    # Convert to uint8 frames and optionally resize
    out_batches = []
    for b in range(B):
        frames = []
        for t in range(T):
            img = (proj[b, t] * 255.0).astype(np.uint8)  # H,W,3
            if resize is not None:
                pil = Image.fromarray(img)
                pil = pil.resize((resize[0], resize[1]), resample=Image.BILINEAR)
                img = np.array(pil)
            frames.append(img)
        out_batches.append(frames)
    return out_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to .pt file (torch.save tensor)")
    parser.add_argument("--out", required=True, help="Output mp4 path")
    parser.add_argument("--resize", nargs=2, type=int, metavar=("W", "H"), default=None,
                        help="Optional resize (width height) for output frames, e.g. 1280 704")
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS")
    parser.add_argument("--batch-index", type=int, default=0, help="Which batch element to save (if B>1)")
    parser.add_argument("--use-first3", action="store_true", help="Use first 3 channels directly as RGB (no PCA)")
    parser.add_argument("--channels", type=str, default=None, help="Comma-separated channel indices to use as RGB, e.g. '1024,371,245'")
    parser.add_argument("--missing-behavior", type=str, default='error', choices=['error','zero','clip','mod'], help="Behavior when requested channels are out of range: error, zero, clip, mod")
    args = parser.parse_args()

    data = torch.load(args.input, map_location="cpu")
    # If file contains a dict, try to find a tensor inside
    if isinstance(data, dict):
        # common key: 'features' or first tensor-like entry
        for k in ["features", "feat", "output", "tensor", "arr"]:
            if k in data and torch.is_tensor(data[k]):
                data = data[k]
                break
        else:
            # fallback: pick first tensor value
            for v in data.values():
                if torch.is_tensor(v):
                    data = v
                    break

    if not torch.is_tensor(data):
        raise RuntimeError("Loaded object is not a tensor. Please pass a .pt that is a saved tensor or dict containing a tensor.")

    # ensure shape (B,C,T,H,W)
    feats = data
    if feats.ndim == 5:
        pass
    elif feats.ndim == 4:
        # (C,T,H,W) or (T,H,W,C) handled in helper
        pass
    elif feats.ndim == 3:
        # (T,H,W)
        pass
    else:
        raise RuntimeError(f"Unsupported tensor dims: {feats.shape}")

    channels = None
    if args.channels:
        channels = [int(x) for x in args.channels.split(',')]

    out_batches = features_to_frames(
        feats,
        resize=tuple(args.resize) if args.resize else None,
        use_first3=args.use_first3,
        channels=channels,
        missing_behavior=args.missing_behavior,
    )
    # Save the chosen batch index
    batch_idx = args.batch_index
    if batch_idx >= len(out_batches):
        raise IndexError(f"batch-index {batch_idx} out of range for B={len(out_batches)}")
    frames = out_batches[batch_idx]

    # Write mp4
    print(f"Writing {len(frames)} frames to {args.out} (fps={args.fps}) ...")
    imageio.mimwrite(args.out, frames, fps=args.fps, macro_block_size=None)
    print("Done.")


if __name__ == "__main__":
    main()
