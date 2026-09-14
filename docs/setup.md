# DwD environment setup

Run the following commands from the root of this DwD checkout. The training and
nuPlan preprocessing entry points are documented in [DwD training](training_dwd.md).

## Base environment

The inherited Cosmos GPU dependency set targets Linux x86-64, Python 3.10,
CUDA 12.8 and PyTorch 2.7.1. Its setup guide specifies Ampere or newer NVIDIA GPUs,
glibc >= 2.31 and NVIDIA driver >= 570.124.06. These are inherited environment
constraints, not measured memory requirements for DwD's x4 training configuration.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed,
then create and activate the environment:

```bash
uv sync --frozen
source .venv/bin/activate
uv pip install -r requirements-dwd.txt
```

The inherited `uv.lock` contains Transformers 4.51.3, which does not provide DINOv3.
The final command installs the DwD dependency overlay into the same environment.
Do not run another frozen sync after installing the overlay: it would restore the
old dependency versions. This combination still needs end-to-end GPU validation
before it can become the release's tested environment lock.

## Model assets

Prepare DINOv3, the Cosmos base checkpoint, the video tokenizer, Cosmos-Reason
weights and the PCA basis at explicit local paths. See the
[asset layout and training commands](training_dwd.md#environment-and-assets).
The DwD launcher expects those files to exist; it does not automatically download
them. Use the exact PCA basis associated with an existing trained checkpoint.

## Basic checks

Check that the driver and PyTorch can see the GPUs:

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
```

Check the DINOv3 imports:

```bash
python -c 'from transformers import DINOv3ViTModel, DINOv3ViTImageProcessorFast'
```

For feature extraction memory limits, start with `--batch-size 1`. The training
launcher defaults to offline features, so DINOv3 does not occupy GPU memory during
training. See [the training guide](training_dwd.md) for online extraction.
