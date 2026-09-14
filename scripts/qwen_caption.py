"""Generate video captions with Qwen3-VL and trim them before text encoding."""

import argparse
import os
from pathlib import Path
import re

DEFAULT_MODEL = "Qwen/Qwen3-VL-32B-Instruct"


def trim_caption(text: str, max_words: int = 200) -> str:
    """Normalize whitespace and trim at a nearby sentence boundary when possible."""
    if max_words < 1:
        raise ValueError("max_words must be positive")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    words = text.split()
    if not words:
        raise ValueError("The model returned an empty caption")
    if len(words) <= max_words:
        return " ".join(words)
    # Prefer a complete sentence within the last 10% of the word budget.
    for end in range(max_words, max(1, int(max_words * 0.9)) - 1, -1):
        if re.search(r'''[.!?]["')\]]*$''', words[end - 1]):
            return " ".join(words[:end])
    return " ".join(words[:max_words]).rstrip(",;:") + "."


def main(args):
    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Video directory does not exist: {input_dir}")
    videos = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4")
    if not videos:
        raise ValueError(f"No MP4 videos found in {input_dir}")
    if len({p.stem for p in videos}) != len(videos):
        raise ValueError("Video filename stems must be unique")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_output).expanduser().resolve() if args.raw_output else None
    if raw_dir is not None:
        if raw_dir == output_dir:
            raise ValueError("Raw and trimmed caption directories must differ")
        raw_dir.mkdir(parents=True, exist_ok=True)
    videos = [p for p in videos if not (output_dir / f"{p.stem}.txt").exists()]
    if not videos:
        print("All captions already exist; nothing to process.")
        return

    # Keep --help and text postprocessing usable without GPU dependencies.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info
    from tqdm import tqdm

    processor = AutoProcessor.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        limit_mm_per_prompt={"image": 0, "video": 1},
        tensor_parallel_size=args.tensor_parallel_size,
        seed=0,
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=1024)
    prompt = (
        f"Describe this driving video in one English paragraph of about {args.max_words} words. "
        "Cover the road layout, surrounding buildings and vegetation, visible traffic and pedestrians, "
        "ego-vehicle motion, and visible lighting and weather. Describe only details supported by "
        "the video; do not invent objects, clouds, signs, or events. Output only the caption, "
        "without headings, lists, or commentary."
    )
    failures = []
    for video in tqdm(videos, desc="Captioning videos"):
        try:
            messages = [{"role": "user", "content": [
                {"type": "video", "video": str(video), "max_pixels": 704 * 1280, "fps": args.fps},
                {"type": "text", "text": prompt},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            _, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            video_kwargs["do_resize"] = False
            inputs = {
                "prompt": text,
                "multi_modal_data": {"video": video_inputs},
                "mm_processor_kwargs": video_kwargs,
            }
            generated = llm.generate([inputs], sampling_params=sampling_params)[0].outputs[0].text
            caption = trim_caption(generated, args.max_words)
            if raw_dir is not None:
                with (raw_dir / f"{video.stem}.txt").open("x", encoding="utf-8") as handle:
                    handle.write(generated + "\n")
            destination = output_dir / f"{video.stem}.txt"
            temporary = destination.with_suffix(".txt.tmp")
            temporary.write_text(caption + "\n", encoding="utf-8")
            temporary.replace(destination)
            print(f"{video.name}: {len(generated.split())} → {len(caption.split())} words")
        except Exception as exc:
            failures.append(video.name)
            print(f"Failed {video.name}: {exc}")
    if failures:
        raise RuntimeError(f"Caption generation failed for {len(failures)} video(s): {', '.join(failures)}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True, help="Directory containing MP4 videos")
    parser.add_argument("-o", "--output", required=True, help="Directory for trimmed captions (dataset metas/)")
    parser.add_argument("-m", "--model", "--MODEL_PATH", default=DEFAULT_MODEL, help="Qwen3-VL model ID or local path")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--fps", type=float, default=1.0, help="Video sampling FPS for caption generation")
    parser.add_argument("--max-words", type=int, default=200)
    parser.add_argument("--raw-output", help="Optional separate directory for untrimmed model responses")
    args = parser.parse_args()
    if args.tensor_parallel_size < 1 or args.max_words < 1 or not 0 < args.fps < float("inf"):
        parser.error("--tensor-parallel-size, --max-words, and --fps must be positive and finite")
    return args


if __name__ == "__main__":
    main(parse_args())
