"""
Build Pose Library from Images

Batch process images to extract pose vectors and save to a library file.
This file is used during training for varied initial positions.

Usage:
    python build_pose_library.py --folder images/
    python build_pose_library.py --folder dataset/poses/ --output models/my_library.npy
"""

import argparse
import os
import numpy as np
from glob import glob
import warnings
warnings.filterwarnings('ignore')

# Import pose converter
from humanoid_walk.perception.pose_converter import PoseConverter


# Supported image extensions
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']


def find_images(folder):
    """Find all image files in folder."""
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(glob(os.path.join(folder, ext)))
        images.extend(glob(os.path.join(folder, ext.upper())))
    return sorted(images)


def build_library(args):
    """Build pose library from all images in folder."""
    print("\n" + "=" * 60)
    print(" BUILDING POSE LIBRARY")
    print("=" * 60)
    
    # Check folder exists
    if not os.path.exists(args.folder):
        print(f"❌ Folder not found: {args.folder}")
        return
    
    # Find images
    images = find_images(args.folder)
    print(f"\n📁 Folder: {args.folder}")
    print(f"📷 Found {len(images)} images")
    
    if len(images) == 0:
        print("❌ No images found!")
        return
    
    # Check detector model
    detector_model = "models/efficientdet_lite2.tflite"
    if not os.path.exists(detector_model):
        print(f"❌ Detector model not found: {detector_model}")
        return
    
    # Initialize converter
    print("\n🔄 Initializing PoseConverter...")
    converter = PoseConverter(detector_model, min_vis_threshold=0.5, min_detect_conf=0.3)
    
    # Process all images
    print(f"\n📊 Processing {len(images)} images...")
    print("-" * 40)
    
    pose_library = []
    successful = 0
    failed = 0
    
    for i, img_path in enumerate(images):
        img_name = os.path.basename(img_path)
        try:
            theta_init, _, _ = converter.run_pipeline(img_path)
            
            if theta_init is not None and len(theta_init) == 12:
                pose_library.append(theta_init)
                successful += 1
                status = "✅"
            else:
                failed += 1
                status = "⚠️ No pose"
        except Exception as e:
            failed += 1
            status = f"❌ Error"
        
        # Progress bar
        progress = (i + 1) / len(images) * 100
        print(f"  [{progress:5.1f}%] {status} {img_name}")
    
    print("-" * 40)
    print(f"\n📊 Results:")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    
    if len(pose_library) == 0:
        print("\n❌ No poses extracted! Check your images.")
        return
    
    # Convert to numpy array
    pose_library = np.array(pose_library, dtype=np.float32)
    print(f"\n📦 Library shape: {pose_library.shape}")
    print(f"   {pose_library.shape[0]} poses × {pose_library.shape[1]} joint angles")
    
    # Save library
    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, pose_library)
    print(f"\n💾 Saved to: {output_path}")
    
    # Show sample
    print(f"\n🔍 Sample pose (first):")
    print(f"   {pose_library[0][:6]}...")
    
    print("\n" + "=" * 60)
    print(" LIBRARY BUILT SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nTo train with this library:")
    print(f"  python train.py --use-sb3 --pose-library {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build pose library from images")
    parser.add_argument("--folder", type=str, required=True,
                        help="Folder containing images")
    parser.add_argument("--output", type=str, default="models/pose_library.npy",
                        help="Output file path (default: models/pose_library.npy)")
    args = parser.parse_args()
    
    build_library(args)


if __name__ == "__main__":
    main()
