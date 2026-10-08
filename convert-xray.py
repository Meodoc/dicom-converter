#!/usr/bin/env python3

import sys
import os
import cv2
import numpy as np
import pydicom
from PIL import Image

if len(sys.argv) < 2:
    print("Usage: python convert_xray.py <path_to_xray.dcm>")
    sys.exit(1)

dcm_path = sys.argv[1]
base_name = os.path.splitext(os.path.basename(dcm_path))[0]
out_name = f"{base_name}_output.png"

print(f"Reading X-Ray: {dcm_path}")
ds = pydicom.dcmread(dcm_path)

# Automatically apply VOI LUT if available to preserve clinical contrast curves
if hasattr(ds, 'pixel_array'):
    try:
        from pydicom.pixel_data_handlers.util import apply_voi_lut
        pixel_array = apply_voi_lut(ds.pixel_array, ds)
    except:
        pixel_array = ds.pixel_array
else:
    print("Error: DICOM file contains no pixel data mapping matrix.")
    sys.exit(1)

# X-Rays can sometimes contain inverse monochrome mapping headers
if ds.get("PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
    print("Inverting photometric pixel mapping (Monochrome1 detected)...")
    pixel_array = np.amax(pixel_array) - pixel_array

# Normalize raw 12-bit or 16-bit X-ray data to standard 8-bit intensity scale
norm_img = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Save to file
Image.fromarray(norm_img).save(out_name)
print(f"SUCCESS: Saved pristine X-ray image to {out_name} ({norm_img.shape[1]}x{norm_img.shape[0]})")
