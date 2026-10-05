#!/usr/bin/env python3
"""
Quick start example for the multi-stage mapping pipeline.

This script demonstrates basic usage and can be run with sample images.
"""

import sys
from pathlib import Path

# Add mapping package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mapping.pipeline.pipeline import run_mapping_pipeline
from mapping.pipeline.config import PipelineConfig, MatcherConfig, GPSPairSelectorConfig


def example_basic():
    """Basic usage with defaults."""
    print("=" * 70)
    print("EXAMPLE 1: Basic usage with defaults")
    print("=" * 70)
    
    image_dir = "./sample_images"  # Change to your image directory
    output_dir = "./mapping_output"
    
    results = run_mapping_pipeline(
        image_dir=image_dir,
        output_dir=output_dir,
        device="cuda"  # or "cpu"
    )
    
    print("\nResults:")
    print(f"  Status: {results['status']}")
    print(f"  Pairs selected: {results['pairs_selected']}")
    print(f"  Pairs matched: {results['pairs_matched']}")
    print(f"  Total verified inliers: {results['verified_matches']}")
    print(f"  SP+LG matches: {results['sp_lg_matches']}")
    print(f"  LoFTR fallbacks: {results['loftr_fallbacks']}")
    print(f"  Features: {results.get('features_h5', 'N/A')}")
    print(f"  Matches: {results.get('matches_h5', 'N/A')}")
    print(f"  COLMAP DB: {results.get('colmap_db', 'N/A')}")


def example_custom_config():
    """Usage with custom configuration."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Custom configuration (for grassy fields)")
    print("=" * 70)
    
    # Create custom config optimized for grassy/texture-poor areas
    config = PipelineConfig()
    
    # Make matching more aggressive (fallback to LoFTR sooner)
        config.matcher.min_inliers_sp_lg = 60  # Lower threshold for inliers
        config.matcher.min_keypoints_per_image = 400  # Lower threshold for keypoints
        config.matcher.max_reprojection_error_px = 4.0  # Accept higher reprojection error
    
    # Increase temporal neighbors (compensate for GPS jitter)
        config.gps_selector.temporal_neighbors_radius = 10
        config.gps_selector.k_neighbors = 25
    
    image_dir = "./sample_images"
    output_dir = "./mapping_output_grassy"
    
    results = run_mapping_pipeline(
        image_dir=image_dir,
        output_dir=output_dir,
        device="cuda",
        config=config
    )
    
    print("\nResults:")
    print(f"  Status: {results['status']}")
    print(f"  Pairs matched: {results['pairs_matched']}/{results['pairs_selected']}")
    print(f"  LoFTR fallbacks: {results['loftr_fallbacks']} (more for grassy areas)")


def example_cpu_fallback():
    """Usage on CPU (slower but works everywhere)."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: CPU-only mode (no GPU)")
    print("=" * 70)
    
    image_dir = "./sample_images"
    output_dir = "./mapping_output_cpu"
    
    results = run_mapping_pipeline(
        image_dir=image_dir,
        output_dir=output_dir,
        device="cpu"  # CPU mode
    )
    
    print("\nResults (CPU mode - slower but portable):")
    print(f"  Status: {results['status']}")
    print(f"  Pairs matched: {results['pairs_matched']}")


def example_read_results():
    """Read and inspect the output files."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Reading output files")
    print("=" * 70)
    
    try:
        import h5py
        
        features_h5 = "./mapping_output/features.h5"
        matches_h5 = "./mapping_output/matches.h5"
        
        # Read features
        print("\nFeatures:")
        with h5py.File(features_h5, 'r') as f:
            for img_name in list(f.keys())[:3]:  # First 3 images
                kpts = f[img_name]["keypoints"][:]
                desc = f[img_name].get("descriptors", None)
                desc_str = f"(shape {desc.shape})" if desc is not None else "(not stored)"
                print(f"  {img_name}: {len(kpts)} keypoints {desc_str}")
        
        # Read matches
        print("\nMatches:")
        with h5py.File(matches_h5, 'r') as f:
            for pair_key in list(f.keys())[:3]:  # First 3 pairs
                matches = f[pair_key][:]
                print(f"  {pair_key}: {len(matches)} matches")
    
    except ImportError:
        print("h5py not installed, skipping h5 file inspection")
    except FileNotFoundError as e:
        print(f"Output files not found: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-stage mapping pipeline examples")
    parser.add_argument("--example", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Which example to run (1-4)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda",
                        help="Device to use for matching")
    parser.add_argument("--images", type=str, default="./sample_images",
                        help="Path to image directory")
    parser.add_argument("--output", type=str, default="./mapping_output",
                        help="Path to output directory")
    
    args = parser.parse_args()
    
    print("\nMulti-Stage Mapping Pipeline - Quick Start Examples")
    print("=" * 70)
    
    if args.example == 1:
        example_basic()
    elif args.example == 2:
        example_custom_config()
    elif args.example == 3:
        example_cpu_fallback()
    elif args.example == 4:
        example_read_results()
    
    print("\n" + "=" * 70)
    print("Done!")
