"""Convert Epona CAM_F0 sequence metadata into DwD front-camera videos.

Schema reference: https://github.com/Kevin-thu/Epona/blob/main/data_preparation/create_nuplan_json.py
No frame resampling: --fps must describe the source camera sequence.
"""
import argparse
import json
from pathlib import Path


def iter_sequences(metadata_dir, sensor_root=None):
    seen = {}
    for path in sorted(metadata_dir.glob("*.json")):
        records = json.loads(path.read_text())
        if not isinstance(records, list):
            raise ValueError(f"{path}: expected a list of Epona sequence records")
        for record in records:
            source_root = Path(record["data_root"])
            log = source_root.name
            source_root = sensor_root / log if sensor_root else source_root
            scene = str(record["scene"])
            if Path(scene).name != scene or scene in (".", ".."):
                raise ValueError(f"Invalid scene identifier: {scene}")
            stem = f"{scene}_{log}"
            names = tuple(record["CAM_F0"])
            if stem in seen:
                if seen[stem] != names:
                    raise ValueError(f"Conflicting sequences for {stem}")
                continue  # Epona per-log JSON files may contain accumulated records.
            seen[stem] = names
            frames = [source_root / "CAM_F0" / name for name in names]
            yield stem, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, help="Override sensor_blobs root")
    parser.add_argument("--fps", type=float, required=True, help="Actual camera cadence; not a resampling request")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("fps must be positive")
    import cv2
    target = args.output_root / "videos/pinhole_front"
    target.mkdir(parents=True, exist_ok=True)
    manifest = []
    for stem, frames in iter_sequences(args.metadata_dir, args.sensor_root):
        if len(frames) < 93:
            print(f"Skipping {stem}: only {len(frames)} frames")
            continue
        destination = target / f"{stem}.mp4"
        if destination.exists():
            raise FileExistsError(destination)
        for frame in frames:
            if not frame.is_file():
                raise FileNotFoundError(frame)
        first = cv2.imread(str(frames[0]))
        if first is None:
            raise ValueError(f"Unreadable image: {frames[0]}")
        h, w = first.shape[:2]
        temporary = target / f"{stem}.tmp.mp4"
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create video: {temporary}")
        try:
            for frame in frames:
                rgb = cv2.imread(str(frame))
                if rgb is None or rgb.shape[:2] != (h, w):
                    raise ValueError(f"Unreadable image or inconsistent resolution: {frame}")
                writer.write(rgb)
        finally:
            writer.release()
        temporary.replace(destination)
        manifest.append(dict(video=destination.name, frames=len(frames), fps=args.fps,
                             first_frame=str(frames[0]), last_frame=str(frames[-1])))
    if not manifest:
        raise ValueError("No sequences of at least 93 frames were exported")
    (args.output_root / "video_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
