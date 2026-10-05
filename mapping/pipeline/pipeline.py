"""
Multi-stage mapping pipeline: GPS-based pair selection → SuperPoint+LightGlue/LoFTR 
cascade matching → RANSAC verification → COLMAP SfM → Export.

This is Architecture B for grassy field mapping:
  1. GPS k-NN + temporal neighbor pairs
  2. SuperPoint + LightGlue first (fast)
  3. Fallback to LoFTR if low inliers/keypoints
  4. RANSAC geometric verification
  5. COLMAP incremental SfM
  6. Export features.h5 + matches.h5 + point cloud
"""

import numpy as np
import cv2
import os
import json
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time
import traceback

from .config import PipelineConfig, MatcherConfig, GPSPairSelectorConfig
from .gps_pair_selector import GPSPairSelector, ImageMetadata
from .feature_matcher import MultiStageFeatureMatcher
from .match_writer import VerifiedMatchAggregator
from .colmap_mapper import SfMPipeline


def _detect_optimal_device() -> str:
    """Auto-detect optimal device: MPS (Mac with Apple Silicon) > CPU."""
    try:
        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except:
        pass
    return "cpu"


def _get_num_workers_default(num_workers: int = 0) -> int:
    """
    Auto-detect optimal number of workers for multiprocessing.
    
    Args:
        num_workers: 0 = auto-detect, otherwise use specified value
    
    Returns:
        Number of worker processes
    """
    if num_workers > 0:
        return num_workers
    
    # Auto-detect: use half of available CPUs (leave some for OS)
    # On M4: 10 cores total, use 6-8 workers
    cpu_count = mp.cpu_count()
    return max(2, min(8, cpu_count // 2 + 2))


def _save_progress_image(output_dir: str, pairs_matched: int, total_pairs: int,
                        sp_lg_matches: int, loftr_fallbacks: int, verified_matches: int):
    """Save a progress image showing current matching statistics."""
    # Create progress visualization
    img_width, img_height = 800, 600
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 240
    
    # Draw progress info
    progress_pct = (pairs_matched / total_pairs * 100) if total_pairs > 0 else 0
    
    # Title
    cv2.putText(img, f"Pipeline Progress", (50, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    
    # Progress bar
    bar_y = 120
    bar_width = 700
    bar_height = 40
    filled_width = int(bar_width * progress_pct / 100)
    cv2.rectangle(img, (50, bar_y), (50 + bar_width, bar_y + bar_height), (200, 200, 200), -1)
    cv2.rectangle(img, (50, bar_y), (50 + filled_width, bar_y + bar_height), (0, 200, 0), -1)
    cv2.rectangle(img, (50, bar_y), (50 + bar_width, bar_y + bar_height), (0, 0, 0), 2)
    cv2.putText(img, f"{progress_pct:.1f}%", (400, bar_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Statistics
    y_start = 200
    stats = [
        f"Pairs Matched: {pairs_matched}/{total_pairs}",
        f"SuperPoint+LightGlue: {sp_lg_matches}",
        f"LoFTR Fallbacks: {loftr_fallbacks}",
        f"Total Inliers: {verified_matches}",
    ]
    
    for i, stat in enumerate(stats):
        cv2.putText(img, stat, (50, y_start + i * 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (40, 40, 40), 2)
    
    # Save
    progress_path = os.path.join(output_dir, "stitch_progress.jpg")
    cv2.imwrite(progress_path, img)
    return progress_path


def _match_pair_worker(args_tuple: Tuple) -> Dict:
    """
    Worker function for multiprocessing pair matching.
    Must be at module level to be picklable.
    
    Args:
        args_tuple: (image1, image2, image1_name, image2_name, image1_id, image2_id, 
                    matcher_config_dict, device)
    
    Returns:
        Dict with match results
    """
    (image1, image2, image1_name, image2_name, image1_id, image2_id, 
     matcher_config, device) = args_tuple
    
    try:
        # Debug: Check inputs
        if image1 is None or image2 is None:
            raise ValueError("Image data is None")
        if image1.size == 0 or image2.size == 0:
            raise ValueError(f"Empty images: {image1.shape} and {image2.shape}")
        
        # Recreate matcher for this worker
        matcher = MultiStageFeatureMatcher(
            min_inliers=matcher_config["min_inliers"],
            min_keypoints=matcher_config["min_keypoints"],
            max_reproj_error_px=matcher_config["max_reproj_error"],
            use_efficient_loftr=matcher_config["use_efficient_loftr"],
            device=device,
            verbose=False
        )
        
        match_result = matcher.match_pair(image1, image2)
        
        return {
            "image1_id": image1_id,
            "image2_id": image2_id,
            "image1_name": image1_name,
            "image2_name": image2_name,
            "success": True,
            "match_result": match_result
        }
    except Exception as e:
        import sys
        error_tb = traceback.format_exc()
        # Also capture stderr in case there are print statements
        return {
            "image1_id": image1_id,
            "image2_id": image2_id,
            "image1_name": image1_name,
            "image2_name": image2_name,
            "success": False,
            "error": str(e) if str(e) else repr(e),
            "traceback": error_tb,
            "exception_type": type(e).__name__
        }


class MultiStageMappingPipeline:
    """Orchestrates the entire multi-stage mapping pipeline."""
    
    def __init__(self, config: Optional[PipelineConfig] = None, verbose: bool = True):
        """
        Args:
            config: PipelineConfig with all settings (uses defaults if None)
            verbose: Print debug info
        """
        self.config = config or PipelineConfig()
        self.verbose = verbose
    
    def load_images_from_dir(self, image_dir: str, metadata_dir: Optional[str] = None) \
            -> Tuple[List[np.ndarray], List[ImageMetadata], Dict[str, int]]:
        """
        Load images and metadata from directory.
        
        Expects:
          - Images: *.jpg or *.png
          - Metadata: either *.json (per-image) or single metadata.csv
        
        Args:
            image_dir: Directory containing images
            metadata_dir: Directory containing metadata (defaults to image_dir)
        
        Returns:
            Tuple of:
              - images: List of (H, W, 3) RGB images
              - metadata: List of ImageMetadata objects (same order as images)
              - image_name_to_id: Dict mapping filename -> image_id
        """
        if metadata_dir is None:
            metadata_dir = image_dir
        
        # Find all image files
        image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) +
                            glob.glob(os.path.join(image_dir, "*.JPG")) +
                            glob.glob(os.path.join(image_dir, "*.png")) +
                            glob.glob(os.path.join(image_dir, "*.PNG")))
        
        if not image_files:
            raise ValueError(f"No images found in {image_dir}")
        
        images = []
        metadata_list = []
        image_name_to_id = {}
        
        for image_id, image_path in enumerate(image_files):
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not load {image_path}, skipping")
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            images.append(img)
            
            # Load metadata
            image_name = os.path.basename(image_path)
            image_name_to_id[image_name] = image_id
            
            # Try to load JSON metadata (per-image)
            metadata_path = os.path.splitext(image_path)[0] + ".json"
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    data = json.load(f)
                    telem = data.get("image", {}).get("telemetry", {})
                    gps = telem.get("gps", {})
                    
                    meta = ImageMetadata(
                        image_id=image_id,
                        filename=image_name,
                        latitude=float(gps.get("latitude", 0)),
                        longitude=float(gps.get("longitude", 0)),
                        altitude=float(telem.get("altitude", 0)),
                        timestamp=float(telem.get("timestamp", image_id)),
                        yaw=float(telem.get("planeYaw", 0))
                    )
                    metadata_list.append(meta)
            else:
                # Fallback: use dummy metadata
                meta = ImageMetadata(
                    image_id=image_id,
                    filename=image_name,
                    latitude=0.0,
                    longitude=0.0,
                    altitude=0.0,
                    timestamp=float(image_id)
                )
                metadata_list.append(meta)
        
        if self.verbose:
            print(f"[Pipeline] Loaded {len(images)} images from {image_dir}")
        
        return images, metadata_list, image_name_to_id
    
    def run(self, image_dir: str, output_dir: str, 
            device: str = "auto", metadata_dir: Optional[str] = None) -> Dict:
        """
        Run complete multi-stage pipeline.
        
        Args:
            image_dir: Input image directory
            output_dir: Output directory for results
            device: "auto" (detect MPS on Mac or CPU), "cpu", "mps", or "cuda"
            metadata_dir: Metadata directory (defaults to image_dir)
        
        Returns:
            Dict with pipeline results and status
        """
        # Auto-detect device if requested
        if device == "auto":
            device = _detect_optimal_device()
            if self.verbose:
                print(f"[Pipeline] Auto-detected device: {device}")
        
        # Validate device
        if device not in ["cpu", "cuda", "mps"]:
            if self.verbose:
                print(f"[Pipeline] Warning: Unknown device '{device}', falling back to cpu")
            device = "cpu"
        results = {
            "status": "failed",
            "pairs_selected": 0,
            "pairs_matched": 0,
            "verified_matches": 0,
            "sp_lg_matches": 0,
            "loftr_fallbacks": 0,
            "sfm_status": "not_run",
            "output_dir": output_dir,
        }
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Load images and metadata
        if self.verbose:
            print("\n" + "="*70)
            print("STEP 1: Loading images and metadata")
            print("="*70)
        
        try:
            images, metadata_list, image_name_to_id = self.load_images_from_dir(image_dir, metadata_dir)
        except Exception as e:
            if self.verbose:
                print(f"[Pipeline] Error loading images: {e}")
            return results
        
        # Step 2: Select candidate pairs (GPS + temporal)
        if self.verbose:
            print("\n" + "="*70)
            print("STEP 2: Selecting candidate pairs (GPS + temporal)")
            print("="*70)
        
        selector = GPSPairSelector(
            k_neighbors=self.config.gps_selector.k_neighbors,
            temporal_radius=self.config.gps_selector.temporal_neighbors_radius,
            max_gps_distance_deg=self.config.gps_selector.gps_max_distance_deg,
            use_temporal=self.config.gps_selector.use_temporal_neighbors,
            verbose=self.verbose
        )
        
        candidate_pairs = selector.select_pairs(metadata_list)
        results["pairs_selected"] = len(candidate_pairs)
        
        if not candidate_pairs:
            if self.verbose:
                print("[Pipeline] No candidate pairs selected!")
            return results
        
        # Step 3: Multi-stage matching on each pair
        if self.verbose:
            print("\n" + "="*70)
            print("STEP 3: Multi-stage matching (SuperPoint+LightGlue → LoFTR cascade)")
            print("="*70)
        
        matcher = MultiStageFeatureMatcher(
            min_inliers=self.config.matcher.min_inliers_sp_lg,
            min_keypoints=self.config.matcher.min_keypoints_per_image,
            max_reproj_error_px=self.config.matcher.max_reprojection_error_px,
            use_efficient_loftr=self.config.matcher.use_efficient_loftr,
            device=device,
            verbose=self.verbose
        )
        
        aggregator = VerifiedMatchAggregator(verbose=self.verbose)
        
        # Determine number of workers
        num_workers = _get_num_workers_default(self.config.num_workers)
        
        # Disable multiprocessing for MPS (torch models don't pickle well on MPS device)
        # MPS GPU already provides parallelism, so single-process is fine
        if device == "mps":
            num_workers = 1
            if self.verbose:
                print("[Pipeline] MPS GPU detected: using single-process mode (MPS models don't pickle for multiprocessing)")
        
        use_multiprocessing = num_workers > 1
        
        # On CPU, multiprocessing can cause issues with PyTorch model pickling
        # Disable for now and use single-process mode
        if device == "cpu" and use_multiprocessing:
            if self.verbose:
                print(f"[Pipeline] NOTE: Disabling multiprocessing on CPU device (PyTorch model pickling issues)")
                print(f"[Pipeline] Use 'mps' or 'cuda' device for multiprocessing acceleration")
            use_multiprocessing = False
        
        if self.verbose:
            if use_multiprocessing:
                print(f"[Pipeline] Using multiprocessing with {num_workers} workers")
            else:
                print(f"[Pipeline] Running in single-process mode")
        
        # Prepare matcher config dict (must be JSON-serializable for multiprocessing)
        matcher_config = {
            "min_inliers": self.config.matcher.min_inliers_sp_lg,
            "min_keypoints": self.config.matcher.min_keypoints_per_image,
            "max_reproj_error": self.config.matcher.max_reprojection_error_px,
            "use_efficient_loftr": self.config.matcher.use_efficient_loftr,
        }
        
        if use_multiprocessing:
            # Multiprocessing: prepare worker tasks
            worker_tasks = []
            for i, j in candidate_pairs:
                worker_tasks.append((
                    images[i], images[j],
                    metadata_list[i].filename, metadata_list[j].filename,
                    i, j,
                    matcher_config,
                    device
                ))
            
            # Run with ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_match_pair_worker, task): idx 
                          for idx, task in enumerate(worker_tasks)}
                
                last_progress_save = time.time()
                progress_save_interval = 10  # seconds
                
                for pair_idx, future in enumerate(as_completed(futures)):
                    try:
                        result = future.result()
                        
                        if result["success"]:
                            match_result = result["match_result"]
                            n_inliers = match_result["n_inliers"]
                            method = match_result["method"]
                            
                            if n_inliers > 0:
                                results["pairs_matched"] += 1
                                results["verified_matches"] += n_inliers
                                
                                if method == "sp_lg":
                                    results["sp_lg_matches"] += 1
                                else:
                                    results["loftr_fallbacks"] += 1
                                
                                aggregator.add_match_result(
                                    image1_name=result["image1_name"],
                                    image2_name=result["image2_name"],
                                    image1_id=result["image1_id"],
                                    image2_id=result["image2_id"],
                                    match_result=match_result
                                )
                            
                            if self.verbose and (pair_idx + 1) % max(1, len(worker_tasks) // 10) == 0:
                                print(f"  [{pair_idx + 1}/{len(worker_tasks)}] "
                                      f"Progress: {results['pairs_matched']} pairs matched")
                            
                            # Save progress image periodically
                            current_time = time.time()
                            if current_time - last_progress_save > progress_save_interval:
                                _save_progress_image(str(output_dir), results["pairs_matched"], 
                                                    len(worker_tasks), results["sp_lg_matches"],
                                                    results["loftr_fallbacks"], results["verified_matches"])
                                last_progress_save = current_time
                        else:
                            if self.verbose:
                                error_msg = result.get('error', 'Unknown error')
                                error_type = result.get('exception_type', 'Unknown')
                                error_tb = result.get('traceback', '')
                                print(f"  [Pair {pair_idx+1}] Error: [{error_type}] {error_msg}")
                                if error_tb and len(error_tb) > 0:
                                    # Print first few lines of traceback
                                    tb_lines = error_tb.split('\n')[:10]
                                    for tb_line in tb_lines:
                                        if tb_line.strip():
                                            print(f"    {tb_line}")
                    
                    except Exception as e:
                        if self.verbose:
                            print(f"[Pipeline] Worker task failed: {e}")
                        continue
        
        else:
            # Single-process mode (original sequential loop)
            matcher = MultiStageFeatureMatcher(
                min_inliers=self.config.matcher.min_inliers_sp_lg,
                min_keypoints=self.config.matcher.min_keypoints_per_image,
                max_reproj_error_px=self.config.matcher.max_reprojection_error_px,
                use_efficient_loftr=self.config.matcher.use_efficient_loftr,
                device=device,
                verbose=self.verbose
            )
            
            last_progress_save = time.time()
            progress_save_interval = 10  # seconds
            
            for pair_idx, (i, j) in enumerate(candidate_pairs):
                if self.verbose:
                    print(f"\n[Pipeline] Pair {pair_idx+1}/{len(candidate_pairs)}: "
                          f"({metadata_list[i].filename}, {metadata_list[j].filename})")
                
                try:
                    match_result = matcher.match_pair(images[i], images[j])
                    
                    n_inliers = match_result["n_inliers"]
                    method = match_result["method"]
                    
                    if n_inliers > 0:
                        results["pairs_matched"] += 1
                        results["verified_matches"] += n_inliers
                        
                        if method == "sp_lg":
                            results["sp_lg_matches"] += 1
                        else:
                            results["loftr_fallbacks"] += 1
                        
                        aggregator.add_match_result(
                            image1_name=metadata_list[i].filename,
                            image2_name=metadata_list[j].filename,
                            image1_id=i,
                            image2_id=j,
                            match_result=match_result
                        )
                    
                    # Save progress image periodically
                    current_time = time.time()
                    if current_time - last_progress_save > progress_save_interval:
                        _save_progress_image(str(output_dir), results["pairs_matched"], 
                                            len(candidate_pairs), results["sp_lg_matches"],
                                            results["loftr_fallbacks"], results["verified_matches"])
                        last_progress_save = current_time
                
                except Exception as e:
                    if self.verbose:
                        print(f"[Pipeline] Error matching pair: {e}")
                    continue
        
        if self.verbose:
            print(f"\n[Pipeline] Matching summary:")
            print(f"  Total verified inliers: {results['verified_matches']}")
            print(f"  Pairs with matches: {results['pairs_matched']}/{results['pairs_selected']}")
            print(f"  SuperPoint+LightGlue: {results['sp_lg_matches']}")
            print(f"  LoFTR fallbacks: {results['loftr_fallbacks']}")
        
        if results["pairs_matched"] == 0:
            if self.verbose:
                print("[Pipeline] No successful matches found!")
            return results
        
        # Step 4: Export matches to h5
        if self.verbose:
            print("\n" + "="*70)
            print("STEP 4: Exporting matches to h5 (hloc format)")
            print("="*70)
        
        features_h5 = str(output_dir / "features.h5")
        matches_h5 = str(output_dir / "matches.h5")
        
        try:
            aggregator.export_h5(features_h5, matches_h5)
            results["features_h5"] = features_h5
            results["matches_h5"] = matches_h5
        except Exception as e:
            if self.verbose:
                print(f"[Pipeline] Error exporting h5: {e}")
        
        # Step 5: Export to COLMAP database
        if self.verbose:
            print("\n" + "="*70)
            print("STEP 5: Exporting to COLMAP database")
            print("="*70)
        
        colmap_db = str(output_dir / "database.db")
        
        try:
            aggregator.export_colmap_db(
                db_path=colmap_db,
                image_names_list=[m.filename for m in metadata_list]
            )
            results["colmap_db"] = colmap_db
        except Exception as e:
            if self.verbose:
                print(f"[Pipeline] Error exporting COLMAP DB: {e}")
        
        # Step 6: Run COLMAP SfM (optional)
        if self.verbose:
            print("\n" + "="*70)
            print("STEP 6: Running COLMAP SfM (if COLMAP is installed)")
            print("="*70)
        
        try:
            sfm_pipeline = SfMPipeline(verbose=self.verbose)
            sfm_results = sfm_pipeline.run_full_pipeline(
                database_path=colmap_db,
                image_dir=image_dir,
                output_dir=str(output_dir / "colmap_model"),
                image_names_list=[m.filename for m in metadata_list],
                gps_coords=[(m.latitude, m.longitude, m.altitude) for m in metadata_list],
                export_as_ply=True,
                num_threads=mp.cpu_count()  # Use all CPU cores for COLMAP
            )
            
            results["sfm_status"] = sfm_results["status"]
            results["sfm_results"] = sfm_results
        
        except Exception as e:
            if self.verbose:
                print(f"[Pipeline] Warning: COLMAP SfM not available: {e}")
            results["sfm_status"] = "skipped"
        
        # Final summary
        results["status"] = "success"
        
        if self.verbose:
            print("\n" + "="*70)
            print("PIPELINE COMPLETE")
            print("="*70)
            print(f"Status: {results['status']}")
            print(f"Output directory: {output_dir}")
            print(f"Features: {features_h5}")
            print(f"Matches: {matches_h5}")
            print(f"COLMAP DB: {colmap_db}")
            if "point_cloud_path" in sfm_results and sfm_results["point_cloud_path"]:
                print(f"Point cloud: {sfm_results['point_cloud_path']}")
        
        return results


def run_mapping_pipeline(image_dir: str, output_dir: str, device: str = "auto",
                        config: Optional[PipelineConfig] = None,
                        num_workers: int = 0) -> Dict:
    """
    Convenience function to run the full pipeline.
    
    Args:
        image_dir: Input image directory
        output_dir: Output directory
        device: "auto" (detect MPS on Mac or CPU), "cpu", "mps", or "cuda"
        config: PipelineConfig (uses defaults if None)
        num_workers: Number of worker processes (0 = auto-detect)
    
    Returns:
        Dict with results
    """
    if config is None:
        config = PipelineConfig()
    
    if num_workers > 0:
        config.num_workers = num_workers
    
    pipeline = MultiStageMappingPipeline(config=config, verbose=True)
    return pipeline.run(image_dir, output_dir, device=device)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python pipeline.py <image_dir> <output_dir> [device] [num_workers]")
        print("  device: auto (detect), cpu, mps, or cuda (default: auto)")
        print("  num_workers: number of worker processes (0=auto-detect, default: 0)")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    output_dir = sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else "auto"
    num_workers = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    
    results = run_mapping_pipeline(image_dir, output_dir, device=device, num_workers=num_workers)
    print(f"\nFinal status: {results['status']}")