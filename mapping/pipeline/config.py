"""
Configuration and defaults for the multi-stage mapping pipeline.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MatcherConfig:
    """Configuration for the SuperPoint + LightGlue / LoFTR cascade matcher."""
    
    # SuperPoint + LightGlue settings
    min_inliers_sp_lg: int = 120  # Fallback to LoFTR if fewer inliers
    min_keypoints_per_image: int = 800  # Fallback to LoFTR if fewer keypoints
    
    # RANSAC / Geometric verification
    max_reprojection_error_px: float = 2.0  # Median reprojection error threshold
    ransac_confidence: float = 0.999
    ransac_max_iters: int = 10000
    
    # Degeneracy check: reject if homography inliers >> essential matrix inliers
    # (indicates planar scene, but we still accept it with a warning)
    homography_to_essential_ratio_threshold: float = 2.5
    
    # LoFTR settings
    use_efficient_loftr: bool = False  # Set to True to use EfficientLoFTR
    loftr_outdoor_checkpoint: str = "outdoor"  # or "indoor"
    loftr_max_keypoints: int = 8000
    
    # Verbosity
    verbose: bool = True


@dataclass
class GPSPairSelectorConfig:
    """Configuration for GPS-based candidate pair selection."""
    
    # k-NN in GPS space
    k_neighbors: int = 15
    
    # Temporal neighborhood (±n frames)
    temporal_neighbors_radius: int = 4
    
    # GPS distance threshold to exclude already-close-in-GPS pairs
    gps_max_distance_deg: float = 0.01  # ~1 km at equator
    
    # Enable temporal neighbors (if timestamps available)
    use_temporal_neighbors: bool = True
    
    # Verbosity
    verbose: bool = True


@dataclass
class MatchWriterConfig:
    """Configuration for exporting matches to h5 and COLMAP formats."""
    
    # Output paths (will be set by pipeline)
    features_h5_path: Optional[str] = None
    matches_h5_path: Optional[str] = None
    colmap_db_path: Optional[str] = None
    
    # Compression
    h5_compression: str = "gzip"
    h5_compression_level: int = 4


@dataclass
class COLMAPMapperConfig:
    """Configuration for COLMAP mapper integration."""
    
    # COLMAP binary paths
    mapper_bin: str = "colmap"  # Assumes colmap is in PATH
    
    # Mapper options
    ba_refine_principal_point: bool = True
    ba_refine_extra_params: bool = True
    min_track_length: int = 2
    min_num_matches: int = 120  # Should align with matcher
    
    # Threading (CPU optimization for multi-core Macs)
    # 0 = auto-detect (uses all CPU cores)
    # Set to specific number to override (e.g., 10 for M4)
    num_threads: int = 0
    
    # Feature extraction (if needed)
    gpu_index: str = "0"
    
    # Output
    model_output_path: Optional[str] = None
    
    # Verbosity
    verbose: bool = True


@dataclass
class PipelineConfig:
    """Top-level configuration for the multi-stage matching pipeline."""
    
    matcher: MatcherConfig
    gps_selector: GPSPairSelectorConfig
    match_writer: MatchWriterConfig
    colmap_mapper: COLMAPMapperConfig
    
    # I/O paths
    image_list_txt_path: Optional[str] = None  # COLMAP format
    cameras_txt_path: Optional[str] = None
    output_dir: Optional[str] = None
    
    # Parallelization
    num_workers: int = 0  # 0 = auto-detect based on CPU cores. For single-process, use 1.
    
    # Verbosity
    verbose: bool = True
    
    def __init__(self):
        """Initialize with sensible defaults."""
        self.matcher = MatcherConfig()
        self.gps_selector = GPSPairSelectorConfig()
        self.match_writer = MatchWriterConfig()
        self.colmap_mapper = COLMAPMapperConfig()
        self.num_workers = 0  # Auto-detect
        self.verbose = True
