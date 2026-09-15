#!/usr/bin/env bash
# DwD x4 / PCA-32 tail-drop-trained model / temporal-convolution inference.
# Usage: bash sh_scripts/inference_hr_video.sh CHECKPOINT VIDEO FEATURES PROMPT OUTPUT
# Folder batch: INPUT_MODE=folder bash sh_scripts/inference_hr_video.sh CHECKPOINT VIDEOS FEATURES PROMPTS OUTPUT
# Paths are relative to the repository root. Activate the training environment first.
set -euo pipefail

if [[ $# != 5 ]]; then
    echo "Usage: $0 CHECKPOINT VIDEO_OR_FOLDER FEATURES_OR_FOLDER PROMPT_OR_FOLDER OUTPUT" >&2
    exit 2
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$script_dir/.."

python_bin=${PYTHON:-python}
num_gpus=${NUM_GPUS:-8}
channels=${PCA_CHANNELS:-32}
input_mode=${INPUT_MODE:-video}
if [[ ! "$num_gpus" =~ ^[1-9][0-9]*$ ]] || ((24 % num_gpus != 0)); then
    echo "NUM_GPUS must be a positive divisor of 24." >&2
    exit 2
fi
case "$channels" in 3|8|12|16|20|24|28|32) ;; *) echo "PCA_CHANNELS must be 3,8,12,16,20,24,28,32." >&2; exit 2 ;; esac
case "$input_mode" in
    video) inputs=(--video_path "$2" --input_control_video_path_dino "$3" --prompt_path "$4") ;;
    folder) inputs=(--video_folder "$2" --input_control_folder_dino "$3" --prompt_folder "$4") ;;
    *) echo "INPUT_MODE must be video or folder." >&2; exit 2 ;;
esac

command=("$python_bin" -m torch.distributed.run --standalone "--nproc_per_node=$num_gpus"
    -m cosmos_transfer2._src.transfer2.inference.inference_vid2vid_dinocontrol_batch
    --experiment dwd_offline --ckpt_iter iter_000000000 --num_gpus "$num_gpus"
    "${inputs[@]}" --save_root "$5" --hint_key dino
    --control_weight "${CONTROL_WEIGHT:-1.0}" --seed "${SEED:-2025}"
    --guidance "${GUIDANCE:-3}" --num_conditional_frames 1 --num_steps "${NUM_STEPS:-35}"
    --skip_load_model --ckpt_paths "$1"
    --max_frames 93 --start_frame 0 --end_frame 93 --shift 5
    --use_prep_anyup_feature --prep_anyup_feature_channel "$channels" --anyup_factor 4
    --exp_override_opts
    "model_parallel.context_parallel_size=$num_gpus"
    "model.config.fsdp_shard_size=$num_gpus"
    model.config.text_encoder_config.compute_online=True
    model.config.tokenizer.compile_encode=False)

if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
for input in "$1" "$2" "$3" "$4"; do
    if [[ ! -e "$input" ]]; then
        echo "Missing input: $input" >&2
        exit 2
    fi
done
exec "${command[@]}"
