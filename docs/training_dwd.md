# DwD training

This is the entry point for the DwD training release. Use `scripts.train_dwd`
for the defaults below. `sh_scripts/` contains only the x4 temporal-convolution
inference launcher. Training targets are real nuPlan videos. CarlaData30hr is the
rendered CARLA dataset planned for a separate data release; no download URL or data
archive is included in this code update.

## Default model

| Component | Default |
| --- | --- |
| Target RGB video | 93 consecutive frames, resized to 704 × 1280 |
| DINOv3 | ViT-L/16, final block, frozen pretrained weights and original LayerNorm |
| DINO input size | 2816 × 5120 (height and width each ×4) |
| Feature grid | 176 × 320 per frame |
| PCA | 32 components, global training-set mean, no whitening |
| Random Channel Tail Drop | Uniformly retain the first k channels, k ∈ {4,8,12,16,20,24,28,32}; zero the remainder |
| Temporal adapter | Prepend 3 zero feature frames; convolutional temporal downsampling ×4 (96 → 24 frames) |
| Spatial adapter | Convolutional downsampling ×4 (176×320 → 44×80) |
| Optimization | Existing Cosmos control-branch training, 10,000 iterations, checkpoint every 1,000 iterations |
| Parallelism | Default 8 GPUs, context parallelism and FSDP shard size 8 |

×4 changes the encoder input, not the output video resolution. Cache all 32 PCA
channels and all RGB frames. Tail drop is sampled during training, and temporal
compression is learned inside the model. Do not perform either operation when
creating the feature cache. The tensor channel count stays 32 even after dropping.

The temporal convolutions retained from the experimental implementation use symmetric
padding and GroupNorm; this implementation is not strictly causal. Changing that
behavior requires a separately validated model/weight revision.

There is no `NORMALIZE_DINO` switch in this training path. The old switch modified
LayerNorm affine parameters; the released path always retains pretrained LayerNorm.
This is separate from ImageNet RGB preprocessing (still required) and optional feature
L2 normalization (disabled here).

## Environment and assets

Start with [the repository environment setup](setup.md). The inherited `uv.lock`
contains Transformers 4.51.3 and is not a complete DwD environment: DINOv3 requires a
newer Transformers implementation. After installing the base environment, install the
DwD overlay below **in the same environment**, and do not run an old frozen sync over it:

```bash
uv pip install -r requirements-dwd.txt
```

The overlay is a candidate environment, not a GPU-validated release lock. The current
code update has CPU unit checks, but no end-to-end GPU training validation. A tested
CUDA/PyTorch/Transformers lock and measured GPU memory requirements remain release
checks. The inherited GPU wheels target Python 3.10/CUDA 12.8/PyTorch 2.7.1; keep those
constraints when following setup.md.

Prepare local assets (paths are explicit launcher arguments):

```text
model_hubs/
├── dinov3-vitl16/                  # Hugging Face DINOv3 ViT-L/16 checkpoint
└── nvidia/
    ├── Cosmos-Predict2.5-2B/
    │   └── tokenizer.pth
    └── Cosmos-Reason1-7B/          # caption embedding encoder
checkpoints/
├── base_model.pt                  # alias for Predict2.5-2B base/post-trained BF16 EMA weights
└── pca/
    ├── patch_pca_mean_32.npy
    └── patch_pca_components_32.npy
```

`base_model.pt` refers specifically to the [Cosmos-Predict2.5-2B post-trained BF16 EMA weights](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B/blob/15a82a2ec231bc318692aa0456a36537c806e7d4/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt), filename `base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt`. DwD is built on Cosmos-Transfer2.5 code and uses this Predict2.5 checkpoint to initialize the generative backbone. See the [README download instructions](../README.md#base-model-exact-checkpoint) for the pinned revision and local alias setup.

A published trained checkpoint must include its exact PCA basis and configuration.
Refitting PCA changes the basis; newly fitted PCA files cannot replace those belonging
to an existing checkpoint. Asset download locations and checksums for the project's
trained weights are still to be supplied.

## nuPlan preprocessing

1. Follow [Epona's data preparation instructions](https://github.com/Kevin-thu/Epona/blob/main/data_preparation/README.md)
   to download/reorganize nuPlan and generate sequence JSON metadata. Configure the
   roots and output paths in Epona's extraction script. Use the training split for
   training and PCA fitting; keep evaluation logs separate.
2. Convert Epona `CAM_F0` sequences into videos with the adapter below. It preserves
   frame order, exports sequences of at least 93 frames, handles duplicate accumulated
   metadata records, and keeps the original image dimensions during encoding. Supply
   the **actual source camera cadence** as `SOURCE_FPS`, verified against the timestamps;
   the adapter does not resample time. The training loader reports the resulting video FPS.

```bash
python -m scripts.prepare_nuplan_videos \
  --metadata-dir /path/to/epona/sequence_json \
  --sensor-root /path/to/nuplan/sensor_blobs \
  --output-root /path/to/nuplan_train \
  --fps "$SOURCE_FPS"
```

The adapter is for Epona sequence records with `CAM_F0`, `scene`, and `data_root` fields.
Review frame continuity/timestamps from the upstream extraction before converting.
This is an interface to Epona output, not a reproduction of any unpublished dataset filtering.

3. Supply a text caption per clip at `metas/<stem>.txt`. The captioning policy and
   finalized train/validation log lists still need to be supplied for exact paper
   reproduction. Compute Cosmos-Reason embeddings:

```bash
python -m scripts.get_cr1_embeddings \
  --dataset_path /path/to/nuplan_train \
  --pretrained_model_name_or_path /path/to/model_hubs/nvidia/Cosmos-Reason1-7B
```

The dataset accepts `cosmos_reason_xxl/<stem>.safetensors` with key `text_embedding`
and the historical `.pkl` format. RGB, captions, and features must have the same stem.

4. Fit PCA once on **training data only**, unless using the exact PCA files distributed
   with a checkpoint. The fitter samples one seeded random frame per selected video,
   samples patches, accumulates centered covariance on CPU, and saves the mean,
   components, eigenvalues and sampling manifest. It uses the same ×4 encoder and RGB
   preparation as cache extraction. It does not whiten features.

```bash
python -m scripts.dino_pca_fit \
  --video-dir /path/to/nuplan_train/videos/pinhole_front \
  --dino-checkpoint /path/to/model_hubs/dinov3-vitl16 \
  --output-dir /path/to/checkpoints/pca \
  --components 32 --seed 0
```

5. Extract offline feature maps (the default training mode):

```bash
python -m scripts.extract_dino_features \
  --video-dir /path/to/nuplan_train/videos/pinhole_front \
  --output-dir /path/to/nuplan_train/dinov3_x4_pca32 \
  --dino-checkpoint /path/to/model_hubs/dinov3-vitl16 \
  --pca-dir /path/to/checkpoints/pca --batch-size 1
```

Each cache is a CPU tensor `[1,32,T,176,320]` in a `.pt` file. Metadata records FPS,
shape and PCA checksums. Extraction writes via temporary files and refuses existing
outputs, so an incompatible cache is never silently reused. For multiple GPUs,
run one process per GPU with `CUDA_VISIBLE_DEVICES`, `--num-shards N` and distinct
`--shard-index 0..N-1`. A full-video CPU output buffer uses roughly 3.44 MiB per frame;
plan host RAM and disk space accordingly. Both online and offline paths resize and
normalize decoded RGB identically before DINO encoding.

Expected layout:

```text
nuplan_train/
├── videos/pinhole_front/<stem>.mp4
├── metas/<stem>.txt
├── cosmos_reason_xxl/<stem>.safetensors
└── dinov3_x4_pca32/<stem>.pt
```

## Training commands

Offline features (default):

```bash
IMAGINAIRE_OUTPUT_ROOT=./outputs python -m scripts.train_dwd \
  --dataset /path/to/nuplan_train \
  --model-hubs /path/to/model_hubs \
  --base-checkpoint /path/to/checkpoints/base_model.pt \
  --gpus 8
```

Online DINO features (requires additional GPU memory and encoder compute; skips only
step 5, still needs PCA files and text embeddings):

```bash
IMAGINAIRE_OUTPUT_ROOT=./outputs python -m scripts.train_dwd \
  --dataset /path/to/nuplan_train \
  --model-hubs /path/to/model_hubs \
  --base-checkpoint /path/to/checkpoints/base_model.pt \
  --feature-mode online \
  --dino-checkpoint /path/to/model_hubs/dinov3-vitl16 \
  --pca-dir /path/to/checkpoints/pca --gpus 8
```

Online extraction processes one frame at a time through the frozen backbone by default.
It keeps all frame features for the temporal adapter and does not enable gradients for
DINOv3. ImageNet normalization is performed in the dataset; `use_processor=False`
avoids normalizing again. Offline mode does not instantiate the DINO encoder.

Use `--dry-run` to inspect the launcher command without loading GPUs or model assets.
Additional Hydra overrides may follow the arguments. For a 1-step smoke run use
`--max-iter 1`; fewer GPUs require sufficient per-GPU memory and a context-parallel size
compatible with the 24 latent frames. The launcher supports single-node training;
for multi-node use the same experiment via `torchrun --nnodes ... --node_rank ...`
and the repository's `scripts.train` entry point.

Training-state resume (use a DCP checkpoint directory, not a base-model `.pt`):

```bash
python -m scripts.train_dwd \
  --dataset /path/to/nuplan_train --model-hubs /path/to/model_hubs \
  --resume /path/to/run/checkpoints/iter_000005000 --gpus 8
```

Validation is disabled by default. For a separate held-out dataset, override
`dataloader_val.dataset.dataset_dir=/path/to/nuplan_val` before enabling validation;
do not report metrics from the launcher's placeholder training-root validation loader.
The launcher also accepts `--val-dataset /path/to/nuplan_val` to set the validation
root and enable validation together. See the [README](../README.md) for the complete workflow.

## Inference launcher

For a checkpoint trained with the default configuration and its matching offline
PCA features, activate the same environment and run from the repository root:

```bash
bash sh_scripts/inference_hr_video.sh model.pt video.mp4 features.pt prompt.txt outputs/demo
```

For a folder of videos with matching feature and prompt stems, use the same launcher:

```bash
INPUT_MODE=folder bash sh_scripts/inference_hr_video.sh model.pt videos/ features/ prompts/ outputs/batch
```

The default is 8 GPUs and 32 PCA channels; use `NUM_GPUS` and `PCA_CHANNELS` to
configure them. `PCA_CHANNELS=8` retains only the first 8 channels from the same
32-component basis. Tail drop is a training augmentation, not a random inference
operation. The launcher processes up to 93 frames per video and uses the model
asset paths in `dwd_offline` (by default under `./model_hubs`). Set `DRY_RUN=1`
to inspect the command without loading models. Folder mode processes videos using
the selected GPU group; it does not create separate per-GPU workers. GPU execution
of this launcher has not yet been validated.

For checkpoint conversion use `python -m scripts.distcp_to_pt --help`; the old
machine-specific conversion shell wrapper has been removed.
