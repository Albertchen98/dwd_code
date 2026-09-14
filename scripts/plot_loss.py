#!/usr/bin/env python3
"""
Plot losses from logs like:
Iteration 5599: Hit counter: 5599/10000 | Loss: 0.0493 | Time: 10.15s

Usage:
  python scripts/plot_loss.py sh_scripts/train_anyup_video_anyupx8_pca8_upfactor_8.1.log -o loss.png --span 50
"""
import re
import argparse
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

LINE_RE = re.compile(
    r"Iteration\s+(\d+):\s+Hit counter:\s+(\d+)/(\d+)\s+\|\s+Loss:\s+([0-9]*\.?[0-9eE+-]+)\s+\|\s+Time:\s+([0-9]*\.?[0-9]+)s"
)

def parse_file(path):
    iters_losses = defaultdict(list)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            m = LINE_RE.search(ln)
            if m:
                it = int(m.group(1))
                loss = float(m.group(4))
                iters_losses[it].append(loss)
    if not iters_losses:
        raise ValueError("No matching lines found in file: " + path)
    iters = sorted(iters_losses.keys())
    losses = [np.mean(iters_losses[it]) for it in iters]
    return np.array(iters), np.array(losses)

def ema(values, alpha=None, span=None):
    if alpha is None:
        if span is None:
            span = 50
        alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i-1]
    return out

def plot(iters, losses, smoothed, outpath, title=None, ylog=False):
    plt.figure(figsize=(10,6))
    plt.plot(iters, losses, marker='.', linestyle='-', alpha=0.25, label='raw loss')
    plt.plot(iters, smoothed, color='C1', linewidth=2, label='EMA smoothed')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    if title:
        plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    if ylog:
        plt.yscale('log')
    plt.tight_layout()
    dirname = os.path.dirname(outpath)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()

def main():
    p = argparse.ArgumentParser(description="Plot loss + EMA from training log")
    p.add_argument("logfile", help="Path to log file")
    p.add_argument("-o", "--output", default="loss.png", help="Output PNG path")
    p.add_argument("--alpha", type=float, help="EMA alpha (0..1). If set, overrides --span")
    p.add_argument("--span", type=int, help="EMA span (converts to alpha=2/(span+1)). Default 50", default=50)
    p.add_argument("--title", help="Plot title")
    p.add_argument("--ylog", action="store_true", help="Use log scale for y axis")
    args = p.parse_args()

    try:
        iters, losses = parse_file(args.logfile)
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(2)

    if args.alpha is not None:
        alpha = args.alpha
        span = None
    else:
        alpha = None
        span = args.span

    sm = ema(losses, alpha=alpha, span=span)
    plot(iters, losses, sm, args.output, title=args.title, ylog=args.ylog)
    print(f"Saved plot to {args.output}")

if __name__ == "__main__":
    main()