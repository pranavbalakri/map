"""
Match writer: Export verified matches to h5 (hloc format) and COLMAP database.

Supports both intermediate h5 export (for debugging/reuse) and direct COLMAP DB writing.
"""

import h5py
import numpy as np
import sqlite3
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class H5MatchWriter:
    """Export features and matches to h5 format (hloc-compatible)."""
    
    def __init__(self, features_h5_path: str, matches_h5_path: str, 
                 compression: str = "gzip", compression_level: int = 4, verbose: bool = True):
        """
        Args:
            features_h5_path: Path to write features.h5
            matches_h5_path: Path to write matches.h5
            compression: Compression method for h5
            compression_level: Compression level
            verbose: Print debug info
        """
        self.features_h5_path = features_h5_path
        self.matches_h5_path = matches_h5_path
        self.compression = compression
        self.compression_level = compression_level
        self.verbose = verbose
        
        # Create parent directories if needed
        Path(features_h5_path).parent.mkdir(parents=True, exist_ok=True)
        Path(matches_h5_path).parent.mkdir(parents=True, exist_ok=True)
    
    def write_features(self, image_names: List[str], 
                      all_keypoints: Dict[str, np.ndarray],
                      all_descriptors: Dict[str, np.ndarray]):
        """
        Write keypoints and descriptors to features.h5.
        
        Args:
            image_names: List of image filenames
            all_keypoints: Dict mapping image_name -> (N, 2) keypoints
            all_descriptors: Dict mapping image_name -> (N, D) descriptors
        """
        with h5py.File(self.features_h5_path, "w") as f:
            for img_name in image_names:
                if img_name not in all_keypoints:
                    if self.verbose:
                        print(f"  Warning: {img_name} not in keypoints dict, skipping")
                    continue
                
                kpts = all_keypoints[img_name]
                desc = all_descriptors.get(img_name, np.empty((len(kpts), 0), dtype=np.float32))
                
                # Create group for this image
                grp = f.create_group(img_name)
                grp.create_dataset("keypoints", data=kpts, compression=self.compression,
                                   compression_opts=self.compression_level)
                if desc.size > 0:
                    grp.create_dataset("descriptors", data=desc, compression=self.compression,
                                       compression_opts=self.compression_level)
        
        if self.verbose:
            print(f"[H5MatchWriter] Wrote features to {self.features_h5_path}")
    
    def write_matches(self, image_pair_matches: Dict[Tuple[str, str], np.ndarray]):
        """
        Write matches to matches.h5.
        
        Args:
            image_pair_matches: Dict mapping (img_name_0, img_name_1) -> (M, 2) match indices
        """
        with h5py.File(self.matches_h5_path, "w") as f:
            for (img0, img1), matches in image_pair_matches.items():
                # Use a consistent key format (pair of names)
                pair_key = f"{img0}_{img1}"
                f.create_dataset(pair_key, data=matches, compression=self.compression,
                                 compression_opts=self.compression_level)
        
        if self.verbose:
            print(f"[H5MatchWriter] Wrote {len(image_pair_matches)} pairs to {self.matches_h5_path}")


class COLMAPDatabaseWriter:
    """Write matches and camera info directly to COLMAP database."""
    
    # Schema borrowed from COLMAP pycolmap
    
    def __init__(self, db_path: str, verbose: bool = True):
        """
        Args:
            db_path: Path to COLMAP database (SQLite)
            verbose: Print debug info
        """
        self.db_path = db_path
        self.verbose = verbose
        
        # Create parent directory if needed
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def create_database(self):
        """Create COLMAP database schema."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create tables
        c.execute("""
            CREATE TABLE IF NOT EXISTS cameras (
                camera_id INTEGER PRIMARY KEY,
                model TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                params BLOB
            )
        """)
        
        c.execute("""
            CREATE TABLE IF NOT EXISTS images (
                image_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                camera_id INTEGER NOT NULL,
                prior_qw REAL,
                prior_qx REAL,
                prior_qy REAL,
                prior_qz REAL,
                prior_tx REAL,
                prior_ty REAL,
                prior_tz REAL,
                FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
            )
        """)
        
        c.execute("""
            CREATE TABLE IF NOT EXISTS keypoints (
                image_id INTEGER PRIMARY KEY,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB NOT NULL,
                FOREIGN KEY(image_id) REFERENCES images(image_id)
            )
        """)
        
        c.execute("""
            CREATE TABLE IF NOT EXISTS descriptors (
                image_id INTEGER PRIMARY KEY,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB NOT NULL,
                FOREIGN KEY(image_id) REFERENCES images(image_id)
            )
        """)
        
        c.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                pair_id INTEGER PRIMARY KEY,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB NOT NULL,
                image_id1 INTEGER NOT NULL,
                image_id2 INTEGER NOT NULL,
                FOREIGN KEY(image_id1) REFERENCES images(image_id),
                FOREIGN KEY(image_id2) REFERENCES images(image_id)
            )
        """)
        
        c.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_image_id_1_2 
            ON matches(image_id1, image_id2)
        """)
        
        c.execute("""
            CREATE TABLE IF NOT EXISTS two_view_geometries (
                pair_id INTEGER PRIMARY KEY,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB NOT NULL,
                config INTEGER NOT NULL,
                F BLOB,
                E BLOB,
                H BLOB,
                image_id1 INTEGER NOT NULL,
                image_id2 INTEGER NOT NULL,
                FOREIGN KEY(image_id1) REFERENCES images(image_id),
                FOREIGN KEY(image_id2) REFERENCES images(image_id)
            )
        """)
        
        conn.commit()
        conn.close()
        
        if self.verbose:
            print(f"[COLMAPDatabaseWriter] Created database at {self.db_path}")
    
    def add_cameras(self, cameras: Dict[int, Dict]):
        """
        Add cameras to database.
        
        Args:
            cameras: Dict mapping camera_id -> {model, width, height, params}
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        for cam_id, cam_info in cameras.items():
            params = np.array(cam_info.get("params", []), dtype=np.float64)
            params_blob = params.tobytes()
            
            c.execute("""
                INSERT OR REPLACE INTO cameras (camera_id, model, width, height, params)
                VALUES (?, ?, ?, ?, ?)
            """, (cam_id, cam_info["model"], cam_info["width"], cam_info["height"], params_blob))
        
        conn.commit()
        conn.close()
        
        if self.verbose:
            print(f"[COLMAPDatabaseWriter] Added {len(cameras)} cameras")
    
    def add_images(self, images: Dict[int, Dict]):
        """
        Add images to database.
        
        Args:
            images: Dict mapping image_id -> {name, camera_id, (optional) prior_pose}
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        for img_id, img_info in images.items():
            prior_pose = img_info.get("prior_pose", [None] * 7)
            
            c.execute("""
                INSERT OR REPLACE INTO images 
                (image_id, name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, 
                 prior_tx, prior_ty, prior_tz)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (img_id, img_info["name"], img_info["camera_id"], 
                  prior_pose[0], prior_pose[1], prior_pose[2], prior_pose[3],
                  prior_pose[4], prior_pose[5], prior_pose[6]))
        
        conn.commit()
        conn.close()
        
        if self.verbose:
            print(f"[COLMAPDatabaseWriter] Added {len(images)} images")
    
    def add_keypoints(self, keypoints_dict: Dict[int, np.ndarray]):
        """
        Add keypoints for images.
        
        Args:
            keypoints_dict: Dict mapping image_id -> (N, 2) keypoints
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        for img_id, kpts in keypoints_dict.items():
            if kpts.size == 0:
                continue
            
            kpts_data = kpts.astype(np.float32)
            kpts_blob = kpts_data.tobytes()
            
            c.execute("""
                INSERT OR REPLACE INTO keypoints (image_id, rows, cols, data)
                VALUES (?, ?, ?, ?)
            """, (img_id, kpts.shape[0], kpts.shape[1], kpts_blob))
        
        conn.commit()
        conn.close()
        
        if self.verbose:
            print(f"[COLMAPDatabaseWriter] Added keypoints for {len(keypoints_dict)} images")
    
    def add_matches(self, matches_dict: Dict[Tuple[int, int], np.ndarray]):
        """
        Add matches between image pairs.
        
        Args:
            matches_dict: Dict mapping (img_id1, img_id2) -> (M, 2) match indices
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        pair_id = 1
        for (img_id1, img_id2), matches in matches_dict.items():
            if matches.size == 0:
                continue
            
            matches_data = matches.astype(np.uint32)
            matches_blob = matches_data.tobytes()
            
            c.execute("""
                INSERT OR REPLACE INTO matches 
                (pair_id, rows, cols, data, image_id1, image_id2)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (pair_id, matches.shape[0], matches.shape[1], matches_blob, img_id1, img_id2))
            
            pair_id += 1
        
        conn.commit()
        conn.close()
        
        if self.verbose:
            print(f"[COLMAPDatabaseWriter] Added matches for {len(matches_dict)} pairs")


class VerifiedMatchAggregator:
    """Aggregate verified matches from multiple pairs into export-ready structures."""
    
    def __init__(self, verbose: bool = True):
        """
        Args:
            verbose: Print debug info
        """
        self.verbose = verbose
        self.image_keypoints = {}
        self.image_descriptors = {}
        self.pair_matches = {}
        self.image_name_to_id = {}
    
    def add_match_result(self, image1_name: str, image2_name: str, image1_id: int, image2_id: int,
                        match_result: Dict, kpts1: np.ndarray = None, kpts2: np.ndarray = None,
                        desc1: np.ndarray = None, desc2: np.ndarray = None):
        """
        Add a match result (from MultiStageFeatureMatcher) to aggregator.
        
        Args:
            image1_name, image2_name: Image filenames
            image1_id, image2_id: Image IDs in COLMAP
            match_result: Dict with keys keypoints1, keypoints2, match_indices, etc.
            kpts1, kpts2: Full keypoint arrays (optional)
            desc1, desc2: Full descriptor arrays (optional)
        """
        # Store mapping
        self.image_name_to_id[image1_name] = image1_id
        self.image_name_to_id[image2_name] = image2_id
        
        # Store keypoints (use match result keypoints if full ones not provided)
        if image1_name not in self.image_keypoints:
            self.image_keypoints[image1_name] = kpts1 if kpts1 is not None else match_result["keypoints1"]
        
        if image2_name not in self.image_keypoints:
            self.image_keypoints[image2_name] = kpts2 if kpts2 is not None else match_result["keypoints2"]
        
        # Store descriptors if provided
        if desc1 is not None:
            self.image_descriptors[image1_name] = desc1
        if desc2 is not None:
            self.image_descriptors[image2_name] = desc2
        
        # Store matches
        pair_key = (min(image1_id, image2_id), max(image1_id, image2_id))
        self.pair_matches[pair_key] = match_result["match_indices"]
        
        if self.verbose:
            print(f"[VerifiedMatchAggregator] Added {match_result['n_inliers']} verified matches "
                  f"for pair ({image1_name}, {image2_name})")
    
    def export_h5(self, features_h5_path: str, matches_h5_path: str):
        """Export to h5 format."""
        image_names = sorted(self.image_name_to_id.keys())
        
        writer = H5MatchWriter(features_h5_path, matches_h5_path, verbose=self.verbose)
        writer.write_features(image_names, self.image_keypoints, self.image_descriptors)
        writer.write_matches({(name1, name2): self.pair_matches[(self.image_name_to_id[name1], 
                                                                  self.image_name_to_id[name2])]
                             for name1 in image_names 
                             for name2 in image_names 
                             if name1 < name2 and 
                                (min(self.image_name_to_id[name1], self.image_name_to_id[name2]),
                                 max(self.image_name_to_id[name1], self.image_name_to_id[name2])) 
                                in self.pair_matches})
    
    def export_colmap_db(self, db_path: str, camera_intrinsics: Dict[int, Dict] = None,
                         image_names_list: List[str] = None):
        """
        Export to COLMAP database format.
        
        Args:
            db_path: Output database path
            camera_intrinsics: Dict mapping image_id -> {model, width, height, params}
            image_names_list: List of image names (in order of image_id)
        """
        writer = COLMAPDatabaseWriter(db_path, verbose=self.verbose)
        writer.create_database()
        
        # Add cameras
        if camera_intrinsics:
            # Group by unique camera model
            cameras = {0: camera_intrinsics.get(0, 
                      {"model": "PINHOLE", "width": 1920, "height": 1440, "params": [1920, 1440, 1920, 1440]})}
            writer.add_cameras(cameras)
        else:
            # Default camera
            cameras = {0: {"model": "PINHOLE", "width": 1920, "height": 1440, "params": [1920, 1440, 1920, 1440]}}
            writer.add_cameras(cameras)
        
        # Add images
        images = {}
        for img_id, img_name in enumerate(image_names_list or sorted(self.image_keypoints.keys())):
            images[img_id] = {"name": img_name, "camera_id": 0}
        writer.add_images(images)
        
        # Add keypoints
        keypoints_dict = {}
        for img_name, kpts in self.image_keypoints.items():
            img_id = self.image_name_to_id.get(img_name, len(keypoints_dict))
            keypoints_dict[img_id] = kpts
        writer.add_keypoints(keypoints_dict)
        
        # Add matches
        writer.add_matches(self.pair_matches)
