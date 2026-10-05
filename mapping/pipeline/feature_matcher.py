"""
Multi-stage feature matching: SuperPoint + LightGlue with fallback to LoFTR.

Pipeline:
  1. Run SuperPoint + LightGlue on each pair (fast)
  2. Geometric verification (RANSAC E/F matrix + inlier count)
  3. If failures or low inlier count → fallback to LoFTR on that pair
  4. Return verified matches (inlier keypoints and indices)
""" 

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
import warnings


class RANSACVerifier:
    """Geometric verification of matches via RANSAC (Essential or Fundamental matrix)."""
    
    def __init__(self, max_reproj_error_px: float = 2.0, confidence: float = 0.999,
                 max_iters: int = 10000, verbose: bool = True):
        """
        Args:
            max_reproj_error_px: Maximum reprojection error for inliers (pixels)
            confidence: RANSAC confidence level
            max_iters: Maximum RANSAC iterations
            verbose: Print debug info
        """
        self.max_reproj_error_px = max_reproj_error_px
        self.confidence = confidence
        self.max_iters = max_iters
        self.verbose = verbose
    
    def verify(self, kpts1: np.ndarray, kpts2: np.ndarray, match_indices: np.ndarray,
               intrinsics1: Optional[np.ndarray] = None,
               intrinsics2: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, str]:
        """
        Verify matches using RANSAC on Essential matrix (if intrinsics provided) or Fundamental.
        
        Args:
            kpts1: (N1, 2) keypoints from image 1
            kpts2: (N2, 2) keypoints from image 2
            match_indices: (M, 2) match indices [[idx1, idx2], ...]
            intrinsics1: (3, 3) camera intrinsic matrix for image 1 (optional)
            intrinsics2: (3, 3) camera intrinsic matrix for image 2 (optional)
        
        Returns:
            Tuple of:
              - inlier_mask: (M,) boolean mask of inliers
              - median_reproj_error: median reprojection error of inliers
              - method_used: "E" for Essential, "F" for Fundamental
        """
        if len(match_indices) < 8:
            if self.verbose:
                print(f"  [RANSAC] Too few matches ({len(match_indices)}) to estimate geometry")
            return np.zeros(len(match_indices), dtype=bool), np.inf, "none"
        
        try:
            # Extract matched points
            pts1 = kpts1[match_indices[:, 0]]
            pts2 = kpts2[match_indices[:, 1]]
            
            # Try Essential matrix (more principled) if intrinsics available
            if intrinsics1 is not None and intrinsics2 is not None:
                # Normalize points using intrinsics
                pts1_norm = (pts1 - intrinsics1[:2, 2]) / np.array([[intrinsics1[0, 0], intrinsics1[1, 1]]])
                pts2_norm = (pts2 - intrinsics2[:2, 2]) / np.array([[intrinsics2[0, 0], intrinsics2[1, 1]]])
                
                E, inlier_mask = cv2.findEssentialMat(
                    pts1_norm, pts2_norm,
                    cameraMatrix=np.eye(3),  # Already normalized
                    method=cv2.RANSAC,
                    prob=self.confidence,
                    threshold=self.max_reproj_error_px / max(intrinsics1[0, 0], intrinsics1[1, 1]),
                    maxIters=self.max_iters
                )
                method_used = "E"
            else:
                # Fall back to Fundamental matrix
                F, inlier_mask = cv2.findFundamentalMat(
                    pts1, pts2,
                    method=cv2.FM_RANSAC,
                    ransacReprojThreshold=self.max_reproj_error_px,
                    confidence=self.confidence,
                    maxIters=self.max_iters
                )
                method_used = "F"
            
            if inlier_mask is None:
                if self.verbose:
                    print(f"  [RANSAC] Geometry estimation failed")
                return np.zeros(len(match_indices), dtype=bool), np.inf, method_used
            
            inlier_mask = inlier_mask.flatten().astype(bool)
            
            # Compute reprojection error for inliers
            if method_used == "E" and E is not None:
                median_reproj_error = self._compute_essential_reproj_error(
                    pts1_norm[inlier_mask], pts2_norm[inlier_mask], E
                )
            elif method_used == "F" and F is not None:
                median_reproj_error = self._compute_fundamental_reproj_error(
                    pts1[inlier_mask], pts2[inlier_mask], F
                )
            else:
                median_reproj_error = np.inf
            
            if self.verbose:
                n_inliers = np.sum(inlier_mask)
                print(f"  [RANSAC] {method_used}-matrix: {n_inliers}/{len(match_indices)} inliers, "
                      f"median reproj error = {median_reproj_error:.2f} px")
            
            return inlier_mask, median_reproj_error, method_used
        
        except Exception as e:
            if self.verbose:
                print(f"  [RANSAC] Exception during verification: {e}")
            return np.zeros(len(match_indices), dtype=bool), np.inf, "error"
    
    @staticmethod
    def _compute_fundamental_reproj_error(pts1: np.ndarray, pts2: np.ndarray,
                                          F: np.ndarray) -> float:
        """Compute symmetric Fundamental matrix reprojection error."""
        pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
        pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])
        
        # Compute epipolar lines
        line2 = F @ pts1_h.T  # lines in img2
        line1 = F.T @ pts2_h.T  # lines in img1
        
        # Distance from point to line
        dist1 = np.abs(np.sum(pts2_h * line2.T, axis=1)) / np.sqrt(line2[0] ** 2 + line2[1] ** 2 + 1e-8)
        dist2 = np.abs(np.sum(pts1_h * line1.T, axis=1)) / np.sqrt(line1[0] ** 2 + line1[1] ** 2 + 1e-8)
        
        return np.median(dist1 + dist2)
    
    @staticmethod
    def _compute_essential_reproj_error(pts1_norm: np.ndarray, pts2_norm: np.ndarray,
                                         E: np.ndarray) -> float:
        """Compute symmetric Essential matrix reprojection error (normalized coords)."""
        # Similar to Fundamental but for normalized coords
        pts1_h = np.hstack([pts1_norm, np.ones((pts1_norm.shape[0], 1))])
        pts2_h = np.hstack([pts2_norm, np.ones((pts2_norm.shape[0], 1))])
        
        line2 = E @ pts1_h.T
        line1 = E.T @ pts2_h.T
        
        dist1 = np.abs(np.sum(pts2_h * line2.T, axis=1)) / np.sqrt(line2[0] ** 2 + line2[1] ** 2 + 1e-8)
        dist2 = np.abs(np.sum(pts1_h * line1.T, axis=1)) / np.sqrt(line1[0] ** 2 + line1[1] ** 2 + 1e-8)
        
        return np.median(dist1 + dist2)


class SuperPointLightGlueMatcher:
    def __init__(self, device="cpu", verbose=True):
        from lightglue import SuperPoint, LightGlue

        self.device = device
        self.verbose = verbose

        self.extractor = SuperPoint(
            max_num_keypoints=4096
        ).eval().to(device)

        self.matcher = LightGlue(
            features="superpoint"
        ).eval().to(device)

        if self.verbose:
            print("[SuperPointLightGlue] Initialized SuperPoint + LightGlue")

    def extract_keypoints(self, image: np.ndarray):
        import torch

        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        image = image.astype(np.float32) / 255.0
        image_t = torch.from_numpy(image)[None, None].to(self.device)

        with torch.no_grad():
            feats = self.extractor.extract(image_t)

        kpts = feats["keypoints"][0].cpu().numpy()
        desc = feats["descriptors"][0].cpu().numpy()

        return kpts, desc
    
    def match(self, kpts1, desc1, kpts2, desc2, shape1, shape2):
        import torch

        # Validate input shapes
        if len(kpts1) == 0 or len(kpts2) == 0:
            raise ValueError(f"Empty keypoints: kpts1={len(kpts1)}, kpts2={len(kpts2)}")
        if desc1.size == 0 or desc2.size == 0:
            raise ValueError(f"Empty descriptors: desc1={desc1.shape}, desc2={desc2.shape}")
        
        # LightGlue matcher expects feature dicts in this format
        # Keypoints: (1, N, 2), Descriptors: (1, N, D)
        # SuperPoint returns kpts (N, 2) and desc (N, D)
        try:
            data = {
                "image0": {
                    "keypoints": torch.from_numpy(kpts1)[None].float().to(self.device),  # (1, N, 2)
                    "descriptors": torch.from_numpy(desc1)[None].float().to(self.device),  # (1, N, D)
                },
                "image1": {
                    "keypoints": torch.from_numpy(kpts2)[None].float().to(self.device),  # (1, N, 2)
                    "descriptors": torch.from_numpy(desc2)[None].float().to(self.device),  # (1, N, D)
                }
            }
            
            # Debug: Check shapes and types
            try:
                img0_kpts_shape = data["image0"]["keypoints"].shape
                img1_kpts_shape = data["image1"]["keypoints"].shape
                img0_desc_shape = data["image0"]["descriptors"].shape
                img1_desc_shape = data["image1"]["descriptors"].shape
            except Exception as e:
                raise RuntimeError(f"Failed to verify tensor shapes: {e}")
            
            with torch.no_grad():
                out = self.matcher(data)
        except AssertionError as e:
            # LightGlue assertion failed - print diagnostic info
            error_msg = str(e) if str(e) else "Empty assertion message"
            diagnostic = (
                f"\n=== LightGlue AssertionError ===\n"
                f"Input keypoints: kpts1 shape={kpts1.shape} dtype={kpts1.dtype}, "
                f"kpts2 shape={kpts2.shape} dtype={kpts2.dtype}\n"
                f"Input descriptors: desc1 shape={desc1.shape} dtype={desc1.dtype}, "
                f"desc2 shape={desc2.shape} dtype={desc2.dtype}\n"
                f"Tensor keypoints: image0={img0_kpts_shape}, image1={img1_kpts_shape}\n"
                f"Tensor descriptors: image0={img0_desc_shape}, image1={img1_desc_shape}\n"
                f"Device: {self.device}\n"
                f"Assertion message: {error_msg}\n"
                f"=================================\n"
            )
            raise RuntimeError(diagnostic) from e

        matches0 = out["matches0"][0].cpu().numpy()
        valid = matches0 > -1
        return np.stack([np.where(valid)[0], matches0[valid]], axis=1)

    def _initialize(self):
        """Lazy initialization of LoFTR."""
        try:
            from kornia.feature import LoFTR
            
            checkpoint = "outdoor" if self.outdoor else "indoor"
            self.matcher = LoFTR(pretrained=checkpoint)
            self.matcher = self.matcher.to(self.device).eval()
            
            if self.verbose:
                print(f"[LoFTR] Initialized LoFTR ({checkpoint})")
        except ImportError as e:
            raise ImportError(
                "LoFTR not installed. Install with: pip install git+https://github.com/zju3dv/LoFTR.git"
            ) from e
    

class LoFTRMatcher:
    def __init__(self, device="cpu", outdoor=True, verbose=True):
        import torch
        from kornia.feature import LoFTR

        self.device = device
        self.verbose = verbose

        checkpoint = "outdoor" if outdoor else "indoor"
        self.matcher = LoFTR(pretrained=checkpoint).to(device).eval()

        if verbose:
            print(f"[LoFTR] Initialized LoFTR ({checkpoint})")

    def match(self, image1: np.ndarray, image2: np.ndarray):
        import torch
        import cv2
        # Convert to grayscale if needed
        if image1.ndim == 3:
            image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        if image2.ndim == 3:
            image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

        # Normalize and convert to 1x1xHxW tensor
        image1_t = torch.from_numpy(image1.astype("float32") / 255.0)[None, None].to(self.device)
        image2_t = torch.from_numpy(image2.astype("float32") / 255.0)[None, None].to(self.device)

        with torch.no_grad():
            out = self.matcher({"image0": image1_t, "image1": image2_t})

        # LoFTR returns keypoints and matching indices
        kpts1 = out["keypoints0"][0].cpu().numpy()
        kpts2 = out["keypoints1"][0].cpu().numpy()

        # LoFTR keypoints are semi-dense; matches are one-to-one
        match_indices = np.arange(len(kpts1)).reshape(-1, 1)
        match_indices = np.hstack([match_indices, match_indices])

        return kpts1, kpts2, match_indices



class MultiStageFeatureMatcher:
    """
    Multi-stage matcher: SuperPoint + LightGlue → LoFTR fallback.
    
    Pipeline:
      1. Try SuperPoint + LightGlue (fast)
      2. Verify with RANSAC
      3. If fails or too few inliers → fallback to LoFTR
      4. Return verified matches
    """
    
    def __init__(self, min_inliers: int = 120, min_keypoints: int = 800,
                 max_reproj_error_px: float = 2.0, use_efficient_loftr: bool = False,
                 device: str = "cpu", verbose: bool = True):
        """
        Args:
            min_inliers: Minimum RANSAC inliers to accept SuperPoint+LightGlue
            min_keypoints: Minimum SuperPoint keypoints to not fallback to LoFTR
            max_reproj_error_px: Maximum reprojection error threshold
            use_efficient_loftr: Use EfficientLoFTR instead of standard LoFTR
            device: "cpu" or "cuda"
            verbose: Print debug info
        """
        self.min_inliers = min_inliers
        self.min_keypoints = min_keypoints
        self.max_reproj_error_px = max_reproj_error_px
        self.use_efficient_loftr = use_efficient_loftr
        self.device = device
        self.verbose = verbose
        
        self.sp_lg_matcher = SuperPointLightGlueMatcher(device=device, verbose=verbose)
        self.loftr_matcher = LoFTRMatcher(device=device, outdoor=True, verbose=verbose)
        self.verifier = RANSACVerifier(max_reproj_error_px=max_reproj_error_px, verbose=verbose)
    
    def match_pair(self, image1: np.ndarray, image2: np.ndarray,
                   intrinsics1: Optional[np.ndarray] = None,
                   intrinsics2: Optional[np.ndarray] = None) -> Dict:
        """
        Match a pair of images using multi-stage cascade.
        
        Args:
            image1, image2: (H, W, 3) RGB images
            intrinsics1, intrinsics2: (3, 3) camera matrices (optional)
        
        Returns:
            Dict with keys:
              - "keypoints1": (N, 2) inlier keypoints in image 1
              - "keypoints2": (N, 2) inlier keypoints in image 2
              - "match_indices": (N, 2) match indices (for reference)
              - "n_inliers": int count
              - "median_reproj_error": float
              - "method": "sp_lg" or "loftr"
              - "status": "success" or "fallback_to_loftr" or "failed"
        """
        
        if self.verbose:
            print(f"\n[MultiStageFeatureMatcher] Matching pair (img1: {image1.shape}, img2: {image2.shape})")
        
        # Stage 1: Try SuperPoint + LightGlue
        if self.verbose:
            print("  Stage 1: SuperPoint + LightGlue")
        
        kpts1, desc1 = self.sp_lg_matcher.extract_keypoints(image1)
        kpts2, desc2 = self.sp_lg_matcher.extract_keypoints(image2)
        
        if self.verbose:
            print(f"    Extracted {len(kpts1)} keypoints from image 1, {len(kpts2)} from image 2")
        
        # Check if enough keypoints on both images
        if len(kpts1) < self.min_keypoints or len(kpts2) < self.min_keypoints:
            if self.verbose:
                print(f"    → Too few keypoints, falling back to LoFTR")
            # Skip LoFTR on MPS to avoid OOM, just return empty
            if self.device == "mps":
                if self.verbose:
                    print(f"    → MPS device: skipping LoFTR fallback (memory constraints)")
                return {
                    "keypoints1": np.empty((0, 2)),
                    "keypoints2": np.empty((0, 2)),
                    "match_indices": np.empty((0, 2), dtype=np.int32),
                    "n_inliers": 0,
                    "median_reproj_error": np.inf,
                    "method": "skipped_loftr_mps",
                    "status": "skipped"
                }
            return self._fallback_to_loftr(image1, image2)
        
        # Match with LightGlue

        match_indices = self.sp_lg_matcher.match(
            kpts1, desc1,
            kpts2, desc2,
            image1.shape[:2],
            image2.shape[:2],
        )

        
        if len(match_indices) < 8:
            if self.verbose:
                print(f"    → Too few matches ({len(match_indices)}), falling back to LoFTR")
            # Skip LoFTR on MPS to avoid OOM
            if self.device == "mps":
                if self.verbose:
                    print(f"    → MPS device: skipping LoFTR fallback (memory constraints)")
                return {
                    "keypoints1": np.empty((0, 2)),
                    "keypoints2": np.empty((0, 2)),
                    "match_indices": np.empty((0, 2), dtype=np.int32),
                    "n_inliers": 0,
                    "median_reproj_error": np.inf,
                    "method": "skipped_loftr_mps",
                    "status": "skipped"
                }
            return self._fallback_to_loftr(image1, image2)
        
        # Geometric verification
        inlier_mask, median_reproj_error, geom_method = self.verifier.verify(
            kpts1, kpts2, match_indices, intrinsics1, intrinsics2
        )
        
        n_inliers = np.sum(inlier_mask)
        
        if n_inliers < self.min_inliers or median_reproj_error > self.max_reproj_error_px:
            if self.verbose:
                print(f"    → Failed verification ({n_inliers} inliers, error={median_reproj_error:.2f}), "
                      f"falling back to LoFTR")
            # Skip LoFTR on MPS to avoid OOM
            if self.device == "mps":
                if self.verbose:
                    print(f"    → MPS device: skipping LoFTR fallback (memory constraints)")
                return {
                    "keypoints1": np.empty((0, 2)),
                    "keypoints2": np.empty((0, 2)),
                    "match_indices": np.empty((0, 2), dtype=np.int32),
                    "n_inliers": 0,
                    "median_reproj_error": np.inf,
                    "method": "skipped_loftr_mps",
                    "status": "skipped"
                }
            return self._fallback_to_loftr(image1, image2)
        
        # Success!
        inlier_matches = match_indices[inlier_mask]
        if self.verbose:
            print(f"    ✓ SuperPoint+LightGlue success: {n_inliers} inliers")
        
        return {
            "keypoints1": kpts1[inlier_matches[:, 0]],
            "keypoints2": kpts2[inlier_matches[:, 1]],
            "match_indices": inlier_matches,
            "n_inliers": n_inliers,
            "median_reproj_error": median_reproj_error,
            "method": "sp_lg",
            "status": "success"
        }
    
    def _fallback_to_loftr(self, image1: np.ndarray, image2: np.ndarray) -> Dict:
        """Fall back to LoFTR semi-dense matching with OOM recovery."""
        if self.verbose:
            print("  Stage 2 (Fallback): LoFTR")
        
        try:
            kpts1, kpts2, match_indices = self.loftr_matcher.match(image1, image2)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and self.device == "mps":
                # MPS ran out of memory, fall back to CPU
                if self.verbose:
                    print(f"    → MPS out of memory, falling back to CPU")
                try:
                    loftr_cpu = LoFTRMatcher(device="cpu", outdoor=True, verbose=False)
                    kpts1, kpts2, match_indices = loftr_cpu.match(image1, image2)
                except Exception as e2:
                    if self.verbose:
                        print(f"    ✗ LoFTR CPU also failed: {e2}")
                    return {
                        "keypoints1": np.empty((0, 2)),
                        "keypoints2": np.empty((0, 2)),
                        "match_indices": np.empty((0, 2), dtype=np.int32),
                        "n_inliers": 0,
                        "median_reproj_error": np.inf,
                        "method": "loftr",
                        "status": "failed"
                    }
            else:
                raise
        
        if len(kpts1) == 0:
            if self.verbose:
                print("    ✗ LoFTR failed to find matches")
            return {
                "keypoints1": np.empty((0, 2)),
                "keypoints2": np.empty((0, 2)),
                "match_indices": np.empty((0, 2), dtype=np.int32),
                "n_inliers": 0,
                "median_reproj_error": np.inf,
                "method": "loftr",
                "status": "failed"
            }
        
        # Optional: verify LoFTR matches with RANSAC
        inlier_mask, median_reproj_error, _ = self.verifier.verify(
            kpts1, kpts2, match_indices
        )
        
        n_inliers = np.sum(inlier_mask)
        if self.verbose:
            print(f"    ✓ LoFTR found {n_inliers} matches (after RANSAC verification)")
        
        return {
            "keypoints1": kpts1[inlier_mask],
            "keypoints2": kpts2[inlier_mask],
            "match_indices": match_indices[inlier_mask],
            "n_inliers": n_inliers,
            "median_reproj_error": median_reproj_error,
            "method": "loftr",
            "status": "fallback_to_loftr"
        }
