"""
GPS-based candidate pair selection for image matching.

This module selects image pairs for matching based on:
  1. GPS proximity (k-NN in lat/lon space)
  2. Temporal neighborhood (consecutive frames in time/sequence)
"""

import numpy as np
from typing import List, Tuple, Set, Optional
from dataclasses import dataclass


@dataclass
class ImageMetadata:
    """Metadata for a single image."""
    image_id: int
    filename: str
    latitude: float
    longitude: float
    altitude: float
    timestamp: Optional[float] = None  # Unix timestamp or sequence index
    yaw: Optional[float] = None


class GPSPairSelector:
    """Selects candidate image pairs for matching based on GPS and temporal cues."""
    
    def __init__(self, k_neighbors: int = 15, temporal_radius: int = 4,
                 max_gps_distance_deg: float = 0.01, use_temporal: bool = True,
                 verbose: bool = True):
        """
        Args:
            k_neighbors: Number of GPS nearest neighbors to consider per image
            temporal_radius: Number of frames before/after to include as temporal neighbors
            max_gps_distance_deg: Maximum GPS distance in degrees (filters distant pairs)
            use_temporal: Whether to include temporal neighbors
            verbose: Print debug info
        """
        self.k_neighbors = k_neighbors
        self.temporal_radius = temporal_radius
        self.max_gps_distance_deg = max_gps_distance_deg
        self.use_temporal = use_temporal
        self.verbose = verbose
    
    def select_pairs(self, image_metadata: List[ImageMetadata]) -> List[Tuple[int, int]]:
        """
        Select candidate image pairs for matching.
        
        Args:
            image_metadata: List of ImageMetadata for all images
        
        Returns:
            List of (image_id_i, image_id_j) pairs with i < j
        """
        n = len(image_metadata)
        if n < 2:
            return []
        
        pairs = set()
        
        # Extract GPS coordinates
        gps_coords = np.array([[m.latitude, m.longitude] for m in image_metadata])
        
        # 1. GPS k-NN pairs
        pairs.update(self._select_gps_neighbors(gps_coords, image_metadata))
        
        # 2. Temporal neighbors (if enabled)
        if self.use_temporal:
            pairs.update(self._select_temporal_neighbors(image_metadata))
        
        # Convert to sorted list of (i, j) with i < j
        pairs_list = sorted(list(pairs))
        
        if self.verbose:
            print(f"[GPSPairSelector] Selected {len(pairs_list)} candidate pairs from {n} images")
        
        return pairs_list
    
    def _select_gps_neighbors(self, gps_coords: np.ndarray,
                              image_metadata: List[ImageMetadata]) -> Set[Tuple[int, int]]:
        """
        Select pairs based on GPS k-NN.
        
        Args:
            gps_coords: (n, 2) array of [lat, lon]
            image_metadata: List of ImageMetadata
        
        Returns:
            Set of (i, j) pairs
        """
        n = len(gps_coords)
        pairs = set()
        
        # Compute pairwise distances (simple Euclidean in lat/lon space)
        # For better accuracy, use Haversine, but lat/lon Euclidean is fine for local areas
        for i in range(n):
            distances = np.sqrt(np.sum((gps_coords - gps_coords[i]) ** 2, axis=1))
            
            # Find k nearest neighbors (excluding self)
            k_nearest_indices = np.argsort(distances)[1:self.k_neighbors + 1]
            
            for j in k_nearest_indices:
                # Apply distance threshold
                if distances[j] <= self.max_gps_distance_deg:
                    pair = (min(i, j), max(i, j))
                    pairs.add(pair)
        
        if self.verbose:
            print(f"[GPSPairSelector] GPS k-NN selected {len(pairs)} pairs (k={self.k_neighbors})")
        
        return pairs
    
    def _select_temporal_neighbors(self, image_metadata: List[ImageMetadata]) -> Set[Tuple[int, int]]:
        """
        Select pairs based on temporal proximity.
        
        Assumes image_id is ordered by time/sequence.
        
        Args:
            image_metadata: List of ImageMetadata (assume sorted by time)
        
        Returns:
            Set of (i, j) pairs
        """
        n = len(image_metadata)
        pairs = set()
        
        for i in range(n):
            # Include ±temporal_radius neighbors
            for offset in range(1, self.temporal_radius + 1):
                j = i + offset
                if j < n:
                    pair = (i, j)
                    pairs.add(pair)
        
        if self.verbose:
            print(f"[GPSPairSelector] Temporal neighbors selected {len(pairs)} pairs "
                  f"(radius=±{self.temporal_radius})")
        
        return pairs


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute great-circle distance between two points on Earth.
    
    Args:
        lat1, lon1: First point in degrees
        lat2, lon2: Second point in degrees
    
    Returns:
        Distance in kilometers
    """
    R_earth_km = 6371.0
    
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance_km = R_earth_km * c
    
    return distance_km
