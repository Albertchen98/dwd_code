import os
import sys
import numpy as np
# Attempt imports inside the function to provide better feedback on missing dependencies

def split_video_into_frames(video_path, output_fps=None):
    """
    Splits a video file into a sequence of image frames using the Decord library.

    The output directory is created with the same name as the video file (without extension).
    
    Args:
        video_path (str): The full path to the input video file.
        output_fps (int, optional): The frame rate for extraction. 
                                    If None (default), it extracts every frame.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return

    # --- Dependency Check and Import ---
    try:
        from decord import VideoReader
        from decord import cpu
        from PIL import Image
    except ImportError:
        print("\nFATAL ERROR: Decord or Pillow library not found.")
        print("Please install them using: pip install decord numpy pillow")
        print("Note: Decord may require additional steps for installation on some systems.")
        return

    # 1. Derive the output directory name
    base_name = os.path.basename(video_path)
    # Remove the extension (e.g., 'my_video.mp4' -> 'my_video')
    folder_name = os.path.join(os.path.dirname(video_path), os.path.splitext(base_name)[0])
    
    # Check if the folder name is empty
    if not folder_name:
        folder_name = "video_frames"

    # 2. Create the output directory if it doesn't exist
    try:
        os.makedirs(folder_name, exist_ok=True)
        print(f"Output directory created: '{folder_name}'")
    except OSError as e:
        print(f"Error creating directory {folder_name}: {e}")
        return

    # 3. Initialize Decord VideoReader
    try:
        # Use CPU context for general purpose extraction
        # Decord is highly optimized for this task
        vr = VideoReader(video_path, ctx=cpu(0))
    except Exception as e:
        print(f"\nFATAL ERROR: Decord could not open the video file.")
        print(f"Please ensure the video format is supported and readable.")
        print(f"Error details: {e}")
        return
        
    total_frames = len(vr)
    output_count = 0

    if output_fps is not None and output_fps > 0:
        # --- FPS-based Sampling ---
        video_fps = vr.get_avg_fps()
        if video_fps == 0:
             print("Error: Could not determine video FPS. Cannot perform FPS-based sampling.")
             return

        # Calculate the step size (frame interval) for the desired FPS
        step = max(1, int(round(video_fps / output_fps)))
        
        # Generate indices for sampling across the whole video
        frame_indices = np.arange(0, total_frames, step)
        
        print(f"Video FPS: {video_fps:.2f}. Sampling interval: 1 frame every {step} frames.")
        print(f"Extracting {len(frame_indices)} frames at approximately {output_fps} FPS...")
        
        # Load and save frames in small batches for memory efficiency
        batch_size = 100
        
        for i in range(0, len(frame_indices), batch_size):
            batch_indices = frame_indices[i:i + batch_size].tolist()
            
            # get_batch loads multiple frames much faster than looping and loading one-by-one
            frame_batch = vr.get_batch(batch_indices).asnumpy()
            
            for j, frame_array in enumerate(frame_batch):
                output_count += 1
                output_filename = os.path.join(folder_name, f"frame_{output_count:05d}.jpg")
                
                # Convert NumPy array (H, W, C) to PIL Image
                img = Image.fromarray(frame_array)
                # Save as high-quality JPEG (quality=95)
                img.save(output_filename, format='JPEG', quality=95)
                
                if output_count % 100 == 0:
                    print(f"Progress: {output_count}/{len(frame_indices)} frames saved.", end='\r')

    else:
        # --- Extract All Frames ---
        print(f"Extracting all {total_frames} frames (original video FPS)...")
        
        for i, frame_nd_array in enumerate(vr):
            frame_array = frame_nd_array.asnumpy() # Convert Decord's NDArray to NumPy array
            output_count += 1
            output_filename = os.path.join(folder_name, f"frame_{output_count:05d}.jpg")
            
            # Convert NumPy array (H, W, C) to PIL Image
            img = Image.fromarray(frame_array)
            img.save(output_filename, format='JPEG', quality=95)
            
            if output_count % 100 == 0:
                # Use i + 1 as we are iterating sequentially
                print(f"Progress: {output_count}/{total_frames} frames saved.", end='\r')

    print(f"\nExtraction complete! Total {output_count} frames saved in '{folder_name}'.")

if __name__ == "__main__":
    # Check for command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python video_splitter.py <path/to/your/video.mp4> [optional_fps_rate]")
        print("\nExample 1 (All frames): python video_splitter.py my_vacation.avi")
        print("Example 2 (1 frame per second): python video_splitter.py my_vacation.avi 1")
        sys.exit(1)

    video_file = sys.argv[1]
    
    # Check for optional FPS argument
    fps = None
    if len(sys.argv) >= 3:
        try:
            fps = int(sys.argv[2])
            if fps <= 0:
                print("Warning: FPS rate must be a positive integer. Extracting all frames instead.")
                fps = None
        except ValueError:
            print(f"Warning: '{sys.argv[2]}' is not a valid number for FPS. Extracting all frames instead.")
            fps = None

    split_video_into_frames(video_file, output_fps=fps)