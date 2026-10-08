#!/usr/bin/env python3

import sys
import os
import cv2
import numpy as np
import pydicom
from PIL import Image

if len(sys.argv) < 2:
    print("Usage: python convert_ultrasound.py <path_to_ultrasound.dcm>")
    sys.exit(1)

dcm_path = sys.argv[1]
base_name = os.path.splitext(os.path.basename(dcm_path))[0]

print(f"Reading ultrasound: {dcm_path}")
ds = pydicom.dcmread(dcm_path)
pixel_array = ds.pixel_array

num_frames = int(ds.get("NumberOfFrames", 1))
samples_per_pixel = int(ds.get("SamplesPerPixel", 1)) # 1=Gray, 3=Color

processed_frames = []

# Unpack frames safely along the correct axes
if num_frames == 1:
    frames = [pixel_array]
else:
    frames = pixel_array

for frame in frames:
    # Normalize brightness to standard 8-bit scale
    norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Standardize to RGB space
    if samples_per_pixel == 1 and len(norm.shape) == 2:
        norm = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
    processed_frames.append(norm)

height, width, _ = processed_frames[0].shape

if num_frames == 1:
    # Output a single clean image
    out_name = f"{base_name}_output.png"
    Image.fromarray(processed_frames[0]).save(out_name)
    print(f"SUCCESS: Saved static frame to {out_name} ({width}x{height})")
else:
    # Output video loops (GIF and MP4)
    frame_time = int(ds.get("CineRate", ds.get("FrameTime", 33)))
    fps = max(1, int(1000 / frame_time))
    
    # Save GIF
    gif_name = f"{base_name}_output.gif"
    pil_frames = [Image.fromarray(f) for f in processed_frames]
    pil_frames[0].save(gif_name, save_all=True, append_images=pil_frames[1:], duration=frame_time, loop=0)
    
    # Save MP4
    mp4_name = f"{base_name}_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(mp4_name, fourcc, fps, (width, height))
    for frame in processed_frames:
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video_writer.release()
    
    print(f"SUCCESS: Saved video loop to {gif_name} and {mp4_name}")
