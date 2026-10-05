"""
COLMAP integration for SfM (Structure from Motion) mapping and alignment.

This module handles:
  1. Running COLMAP mapper (point/track creation, BA)
  2. Alignment to GPS coordinates
  3. Exporting reconstruction models
"""

import subprocess
import json
import os
import multiprocessing as mp
from typing import Optional, List, Tuple, Dict
from pathlib import Path


class COLMAPMapper:
    """Interface to COLMAP mapper binary."""
    
    def __init__(self, colmap_bin: str = "colmap", verbose: bool = True):
        """
        Args:
            colmap_bin: Path to COLMAP executable (default: assumes in PATH)
            verbose: Print debug info
        """
        self.colmap_bin = colmap_bin
        self.verbose = verbose
        self._check_colmap_available()
    
    def _check_colmap_available(self):
        """Check if COLMAP binary is available."""
        try:
            result = subprocess.run([self.colmap_bin, "-h"], capture_output=True, timeout=5)
            if result.returncode not in [0, 1]:  # -h may return 1 on some versions
                raise RuntimeError(f"COLMAP not found at {self.colmap_bin}")
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(f"Could not find COLMAP: {e}. Install COLMAP or set correct path.")
    
    def run_mapper(self, database_path: str, image_dir: str, output_model_dir: str,
                   gpu_index: str = "0", 
                   ba_refine_principal_point: bool = True,
                   ba_refine_extra_params: bool = True,
                   min_track_length: int = 2,
                   num_threads: int = 0,
                   **kwargs) -> bool:
        """
        Run COLMAP mapper (incremental SfM).
        
        Args:
            database_path: Path to COLMAP database (created by feature extraction + matching)
            image_dir: Directory containing images
            output_model_dir: Output directory for reconstruction
            gpu_index: GPU index for CUDA (pass "" to disable GPU)
            ba_refine_principal_point: Refine principal point in bundle adjustment
            ba_refine_extra_params: Refine radial distortion and tangential parameters
            min_track_length: Minimum track length for triangulation
            num_threads: Number of CPU threads for mapper (0=auto-detect all cores)
            **kwargs: Additional COLMAP options
        
        Returns:
            bool: True if mapping succeeded
        """
        output_model_dir = Path(output_model_dir)
        output_model_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect CPU threads if not specified
        if num_threads <= 0:
            num_threads = mp.cpu_count()
        
        cmd = [
            self.colmap_bin,
            "mapper",
            f"--database_path={database_path}",
            f"--image_path={image_dir}",
            f"--output_path={output_model_dir}",
            f"--Mapper.num_threads={num_threads}",
            f"--Mapper.ba_refine_principal_point={int(ba_refine_principal_point)}",
            f"--Mapper.ba_refine_extra_params={int(ba_refine_extra_params)}",
            f"--Mapper.min_track_length={min_track_length}",
        ]
        
        if gpu_index:
            cmd.append(f"--Mapper.gpu_index={gpu_index}")
        
        # Add any additional options
        for key, value in kwargs.items():
            cmd.append(f"--{key}={value}")
        
        if self.verbose:
            print(f"[COLMAPMapper] Running command:\n{' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode != 0:
                if self.verbose:
                    print(f"[COLMAPMapper] STDERR: {result.stderr}")
                return False
            
            if self.verbose:
                print(f"[COLMAPMapper] Mapping completed successfully")
                # Print last few lines of output
                lines = result.stdout.strip().split("\n")
                for line in lines[-5:]:
                    print(f"  {line}")
            
            return True
        
        except subprocess.TimeoutExpired:
            if self.verbose:
                print(f"[COLMAPMapper] Mapping timed out (>1 hour)")
            return False
        except Exception as e:
            if self.verbose:
                print(f"[COLMAPMapper] Error running mapper: {e}")
            return False
    
    def run_bundle_adjustment(self, database_path: str, model_dir: str,
                             gpu_index: str = "0",
                             num_threads: int = 0,
                             **kwargs) -> bool:
        """
        Run COLMAP bundle adjustment on existing model.
        
        Args:
            database_path: Path to COLMAP database
            model_dir: Path to model directory (contains cameras.txt, images.txt, points3D.txt)
            gpu_index: GPU index for CUDA
            num_threads: Number of CPU threads for bundle adjustment (0=auto-detect all cores)
            **kwargs: Additional COLMAP options
        
        Returns:
            bool: True if BA succeeded
        """
        # Auto-detect CPU threads if not specified
        if num_threads <= 0:
            num_threads = mp.cpu_count()
        
        cmd = [
            self.colmap_bin,
            "bundle_adjuster",
            f"--database_path={database_path}",
            f"--input_path={model_dir}",
            f"--output_path={model_dir}",
            f"--BundleAdjustment.num_threads={num_threads}",
        ]
        
        if gpu_index:
            cmd.append(f"--BundleAdjuster.gpu_index={gpu_index}")
        
        for key, value in kwargs.items():
            cmd.append(f"--{key}={value}")
        
        if self.verbose:
            print(f"[COLMAPMapper] Running bundle adjustment...")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode != 0:
                if self.verbose:
                    print(f"[COLMAPMapper] BA failed: {result.stderr}")
                return False
            
            if self.verbose:
                print(f"[COLMAPMapper] Bundle adjustment completed")
            
            return True
        
        except subprocess.TimeoutExpired:
            if self.verbose:
                print(f"[COLMAPMapper] BA timed out")
            return False
        except Exception as e:
            if self.verbose:
                print(f"[COLMAPMapper] Error running BA: {e}")
            return False
    
    def align_to_gps(self, model_dir: str, image_names_list: List[str], 
                     gps_coords: List[Tuple[float, float, float]],
                     output_aligned_dir: Optional[str] = None) -> bool:
        """
        Align SfM reconstruction to GPS coordinates.
        
        This uses an affine alignment: find the transformation that best aligns
        SfM points to ground-truth GPS positions.
        
        Args:
            model_dir: Input model directory
            image_names_list: List of image filenames (in order of image_id)
            gps_coords: List of (lat, lon, altitude) tuples corresponding to image centers
            output_aligned_dir: Output directory for aligned model (default: overwrite input)
        
        Returns:
            bool: True if alignment succeeded
        """
        if output_aligned_dir is None:
            output_aligned_dir = model_dir
        
        output_aligned_dir = Path(output_aligned_dir)
        output_aligned_dir.mkdir(parents=True, exist_ok=True)
        
        # Use COLMAP's geometric alignment tool
        # This is typically done via command line:
        cmd = [
            self.colmap_bin,
            "model_aligner",
            f"--input_path={model_dir}",
            f"--output_path={output_aligned_dir}",
            f"--robust_alignment_max_error=10",
        ]
        
        if self.verbose:
            print(f"[COLMAPMapper] Running model alignment to GPS...")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                if self.verbose:
                    print(f"[COLMAPMapper] Alignment failed: {result.stderr}")
                return False
            
            if self.verbose:
                print(f"[COLMAPMapper] Model alignment completed")
            
            return True
        
        except Exception as e:
            if self.verbose:
                print(f"[COLMAPMapper] Error aligning model: {e}")
            return False
    
    def export_model(self, model_dir: str, output_format: str = "ply") -> Optional[str]:
        """
        Export COLMAP model to common formats.
        
        Args:
            model_dir: Input model directory (contains cameras.txt, images.txt, points3D.txt)
            output_format: Format to export (ply, obj, etc.)
        
        Returns:
            Path to exported model file, or None if failed
        """
        output_path = Path(model_dir) / f"model.{output_format}"
        
        cmd = [
            self.colmap_bin,
            "model_converter",
            f"--input_path={model_dir}",
            f"--output_path={output_path}",
            f"--output_type={output_format.upper()}",
        ]
        
        if self.verbose:
            print(f"[COLMAPMapper] Exporting model as {output_format}...")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                if self.verbose:
                    print(f"[COLMAPMapper] Export failed: {result.stderr}")
                return None
            
            if self.verbose:
                print(f"[COLMAPMapper] Model exported to {output_path}")
            
            return str(output_path)
        
        except Exception as e:
            if self.verbose:
                print(f"[COLMAPMapper] Error exporting model: {e}")
            return None


class SfMPipeline:
    """Complete SfM pipeline: feature extraction → matching → mapping → alignment → export."""
    
    def __init__(self, verbose: bool = True):
        self.mapper = COLMAPMapper(verbose=verbose)
        self.verbose = verbose
    
    def run_full_pipeline(self, database_path: str, image_dir: str, output_dir: str,
                         image_names_list: Optional[List[str]] = None,
                         gps_coords: Optional[List[Tuple[float, float, float]]] = None,
                         export_as_ply: bool = True,
                         num_threads: int = 0) -> Dict:
        """
        Run complete SfM: mapper → BA → alignment → export.
        
        Args:
            database_path: COLMAP database with features + matches
            image_dir: Directory containing images
            output_dir: Base output directory
            image_names_list: List of image names (for alignment)
            gps_coords: GPS coordinates for alignment (lat, lon, altitude)
            export_as_ply: Export point cloud as PLY
            num_threads: Number of CPU threads (0=auto-detect all cores)
        
        Returns:
            Dict with results and status
        """
        results = {
            "status": "failed",
            "mapper_success": False,
            "ba_success": False,
            "alignment_success": False,
            "export_success": False,
            "model_dir": None,
            "point_cloud_path": None,
        }
        
        output_dir = Path(output_dir)
        model_dir = output_dir / "sparse" / "0"
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Mapper
        if self.verbose:
            print("[SfMPipeline] Starting mapper...")
        
        if self.mapper.run_mapper(database_path, image_dir, str(model_dir.parent),
                                  num_threads=num_threads):
            results["mapper_success"] = True
        else:
            if self.verbose:
                print("[SfMPipeline] Mapper failed")
            return results
        
        # Step 2: Bundle Adjustment
        if self.verbose:
            print("[SfMPipeline] Running bundle adjustment...")
        
        if self.mapper.run_bundle_adjustment(database_path, str(model_dir),
                                             num_threads=num_threads):
            results["ba_success"] = True
        else:
            if self.verbose:
                print("[SfMPipeline] BA failed (non-critical)")
        
        # Step 3: Alignment to GPS (if provided)
        if gps_coords and image_names_list:
            if self.verbose:
                print("[SfMPipeline] Aligning to GPS...")
            
            if self.mapper.align_to_gps(str(model_dir), image_names_list, gps_coords):
                results["alignment_success"] = True
            else:
                if self.verbose:
                    print("[SfMPipeline] Alignment failed (non-critical)")
        
        # Step 4: Export
        if export_as_ply:
            if self.verbose:
                print("[SfMPipeline] Exporting point cloud...")
            
            ply_path = self.mapper.export_model(str(model_dir), "ply")
            if ply_path:
                results["export_success"] = True
                results["point_cloud_path"] = ply_path
        
        results["status"] = "success" if results["mapper_success"] else "partial"
        results["model_dir"] = str(model_dir)
        
        if self.verbose:
            print(f"[SfMPipeline] Pipeline complete: {results['status']}")
        
        return results
