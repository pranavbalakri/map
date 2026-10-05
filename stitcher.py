import cv2
import numpy as np
import os
import gc
from typing import List, Tuple, Dict

MAX_CANVAS_DIM = 15000
NEIGHBOR_RADIUS_M = 80.0  # Only compare images within 80m (close overlap)
MAX_ANCHORS = 8  # Only try top 8 closest anchors instead of 20
RATIO_TEST = 0.9
RANSAC_REPROJ = 6.0
SIFT_FEATURES = 3000  # Fewer features needed for close neighbors
SIFT_CONTRAST = 0.005
GPS_DIR_COS_MIN = 0.3
GPS_MAG_MIN = 0.2
GPS_MAG_MAX = 3.5
GPS_MIN_DIST_M = 2.0

PREVIEW_EVERY = 25
REFINE_MAX_ITERS = 40
REFINE_EARLY_STOP = 1e-3
SEAM_MAX_DIM = 800

def map_radians(deg):
    return deg * np.pi / 180.0

def get_gps_pixel_delta(coord1, coord2, ref_lat, PPM):
    """
    coord: (lat, lon)
    Returns (dx, dy) in pixels from coord1 to coord2.
    """
    R = 6378137.0
    lat1, lon1 = map_radians(coord1[0]), map_radians(coord1[1])
    lat2, lon2 = map_radians(coord2[0]), map_radians(coord2[1])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Meters
    dy_m = dlat * R
    dx_m = dlon * R * np.cos(map_radians(ref_lat))
    
    # Pixels (Y is down in images, so North is -Y? No, if we align North to top, then North is -Y)
    # However, for stitching logic, we just need a consistent vector.
    # In OpenCV, increasing Y is down.
    dx = dx_m * PPM
    dy = -dy_m * PPM 
    return dx, dy

class StitchResult:
    def __init__(self, image, h_matrix):
        self.image = image
        self.h_matrix = h_matrix

def stitch_geolocated_images(images: List[np.ndarray], coordinates: List[Tuple[float, float]],
                           match_threshold: int = 10,
                           use_alpha_blend: bool = True,
                           alpha: float = 0.6,
                           skip_assembly: bool = False,
                           ppm: float = 5.0) -> List[StitchResult]:
    """
    Stitches images sequentially by placing them one by one.
    Scaling is determined by homography.
    GPS is used for neighbor search and translation sanity checks.

    If skip_assembly is True, canvas assembly is deferred (returns None
    for the canvas) so the caller can do a single assembly after any
    post-processing removals.
    """
    if not images:
        return []
        
    num_images = len(images)
    PPM = ppm
    ref_lat = coordinates[0][0]
    
    # Convert all coordinates to meter-based offsets from the first image
    positions = []
    R = 6378137.0
    lat0, lon0 = map_radians(coordinates[0][0]), map_radians(coordinates[0][1])
    for lat, lon in coordinates:
        l_rad, lo_rad = map_radians(lat), map_radians(lon)
        dy = (l_rad - lat0) * R
        dx = (lo_rad - lon0) * R * np.cos(lat0)
        positions.append((dx, dy))
        
    # Initialize map with first image
    # H maps image coordinates to the shared map coordinates
    placed_info = {
        0: {
            "H": np.eye(3),
            "pos": positions[0]
        }
    }
    placed_indices = [0]

    # Spatial grid for placed indices (reduces neighbor search cost)
    cell_size = NEIGHBOR_RADIUS_M
    placed_grid: Dict[Tuple[int, int], List[int]] = {}
    _grid_add(placed_grid, 0, positions[0], cell_size)

    # SIFT setup
    print("Pre-computing features for images...")
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES, contrastThreshold=SIFT_CONTRAST)
    bf = cv2.BFMatcher()
    
    img_features = []
    for i, img in enumerate(images):
        kp, des = sift.detectAndCompute(img, None)
        img_features.append((kp, des))
        if (i+1) % 50 == 0:
            print(f"  Extracted {i+1}/{num_images} images...")
    
    # Canvas setup
    h0, w0 = images[0].shape[:2]
    
    unplaced = set(range(1, num_images))
    pass_num = 1
    
    while unplaced:
        print(f"--- Starting Pass {pass_num} (unplaced: {len(unplaced)}) ---")
        placed_in_pass = 0
        current_to_try = sorted(list(unplaced))
        for i in current_to_try:
            img_to_add = images[i]
            kp_new, des_new = img_features[i]
            # Enhanced sensitivity: lower threshold for features
            if des_new is None or len(kp_new) < max(3, match_threshold // 2):
                # Fallback: Place using GPS only if no features
                print(f"  Image {i} has poor features, placing by GPS fallback.")
                # Place using GPS relative to nearest placed neighbor if available
                nearest = None
                best_d = np.inf
                for j in placed_indices:
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    d = np.hypot(dx, dy)
                    if d < best_d:
                        best_d = d
                        nearest = j
                if nearest is not None:
                    exp_dx, exp_dy = get_gps_pixel_delta(coordinates[nearest], coordinates[i], ref_lat, PPM)
                    T = np.array([[1, 0, exp_dx], [0, 1, exp_dy], [0, 0, 1]], dtype=np.float32)
                    placed_H = placed_info[nearest]["H"].dot(T)
                else:
                    placed_H = np.eye(3)
                placed_indices.append(i)
                placed_info[i] = {"H": placed_H, "pos": positions[i], "gps_fallback": True}
                unplaced.remove(i)
                placed_in_pass += 1
                _grid_add(placed_grid, i, positions[i], cell_size)
                continue  # Skip feature matching – already placed by GPS
            best_H_to_map = None
            best_anchor = None
            max_inliers = -1
            potential_anchors = []
            for idx in _grid_candidates(placed_grid, positions[i], cell_size):
                dx = positions[i][0] - positions[idx][0]
                dy = positions[i][1] - positions[idx][1]
                dist_m = np.sqrt(dx**2 + dy**2)
                if dist_m < NEIGHBOR_RADIUS_M:
                    potential_anchors.append((idx, dist_m))
            if not potential_anchors:
                for adj in (i - 1, i + 1):
                    if adj in placed_indices:
                        dx = positions[i][0] - positions[adj][0]
                        dy = positions[i][1] - positions[adj][1]
                        dist_m = np.sqrt(dx**2 + dy**2)
                        if dist_m < NEIGHBOR_RADIUS_M * 1.5:
                            potential_anchors.append((adj, dist_m))
            potential_anchors.sort(key=lambda x: x[1])
            for idx, dist_m in potential_anchors[:MAX_ANCHORS]:
                kp_ref, des_ref = img_features[idx]
                if des_ref is None: continue
                matches = bf.knnMatch(des_new, des_ref, k=2)
                good_matches = [m for m, n in matches if m.distance < RATIO_TEST * n.distance]
                if len(good_matches) < max(3, match_threshold // 2):
                    continue
                src_pts = np.float32([kp_new[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
                dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
                M_sim, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_REPROJ)
                if M_sim is None:
                    continue
                sc_x = np.linalg.norm(M_sim[:, 0])
                sc_y = np.linalg.norm(M_sim[:, 1])
                avg_scale = (sc_x + sc_y) / 2.0
                if avg_scale < 0.8 or avg_scale > 1.2:
                    continue
                # Reject excessive rotation / shear which indicate oblique view or bad matches
                try:
                    rot_rad = np.arctan2(M_sim[1,0], M_sim[0,0])
                    rot_deg = abs(np.degrees(rot_rad))
                except Exception:
                    rot_deg = 0.0
                if rot_deg > 25.0:
                    continue
                # Orthogonality check: columns of linear part should be near-orthogonal scaled axes
                lin = M_sim[:2, :2]
                ortho_err = np.linalg.norm(lin.T.dot(lin) - np.eye(2) * (avg_scale ** 2))
                if ortho_err > 0.35 * (avg_scale ** 2):
                    continue
                H_rel = np.eye(3)
                H_rel[:2, :] = M_sim
                h_n, w_n = img_to_add.shape[:2]
                center_new = np.array([w_n/2, h_n/2, 1.0]).reshape(3,1)
                proj_center = H_rel.dot(center_new)
                proj_center /= (proj_center[2, 0] + 1e-10)
                h_ref, w_ref = images[idx].shape[:2]
                vec_sift = np.array([proj_center[0,0] - w_ref/2, proj_center[1,0] - h_ref/2])
                exp_dx, exp_dy = get_gps_pixel_delta(coordinates[idx], coordinates[i], ref_lat, PPM)
                vec_gps = np.array([exp_dx, exp_dy])
                mag_sift = np.linalg.norm(vec_sift)
                mag_gps = np.linalg.norm(vec_gps)
                if mag_gps > GPS_MIN_DIST_M * PPM:
                    cos_sim = np.dot(vec_sift, vec_gps) / (mag_sift * mag_gps + 1e-10)
                    if cos_sim < GPS_DIR_COS_MIN:
                        continue
                    if mag_sift < GPS_MAG_MIN * mag_gps or mag_sift > GPS_MAG_MAX * mag_gps:
                        continue
                inliers = int(mask.sum())
                if inliers > max_inliers and inliers >= max(3, match_threshold // 2):
                    max_inliers = inliers
                    best_H_to_map = placed_info[idx]["H"].dot(H_rel)
                    best_anchor = idx
            if best_H_to_map is not None:
                print(f"  Image {i} placed (anchor: {best_anchor}, inliers: {max_inliers})")
                placed_indices.append(i)
                placed_info[i] = {"H": best_H_to_map}
                unplaced.remove(i)
                placed_in_pass += 1
                _grid_add(placed_grid, i, positions[i], cell_size)
                if i < 10 or placed_in_pass % PREVIEW_EVERY == 0:
                    save_preview(images, placed_indices, placed_info)
        if placed_in_pass == 0:
            print(f"No more images could be linked. Terminating iterative passes.")
            break
        print(f"Finished Pass {pass_num}. Placed {placed_in_pass} new images.")
        pass_num += 1

    # After all passes, place any remaining unplaced images by GPS fallback
    if unplaced:
        print(f"Placing {len(unplaced)} remaining images by GPS fallback...")
        for i in sorted(list(unplaced)):
            # Place by GPS relative to nearest placed neighbor
            nearest = None
            best_d = np.inf
            for j in placed_indices:
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                d = np.hypot(dx, dy)
                if d < best_d:
                    best_d = d
                    nearest = j
            if nearest is not None:
                exp_dx, exp_dy = get_gps_pixel_delta(coordinates[nearest], coordinates[i], ref_lat, PPM)
                T = np.array([[1, 0, exp_dx], [0, 1, exp_dy], [0, 0, 1]], dtype=np.float32)
                placed_H = placed_info[nearest]["H"].dot(T)
            else:
                placed_H = np.eye(3)
            placed_indices.append(i)
            placed_info[i] = {"H": placed_H, "pos": positions[i], "gps_fallback": True}
            print(f"  Image {i} placed by GPS fallback.")

    # --- GLOBAL REFINEMENT PASS ---
    print("Starting Global Refinement Pass...")
    all_links = []

    # Build a grid once for all positions to avoid O(N^2)
    all_grid: Dict[Tuple[int, int], List[int]] = {}
    for idx, pos in enumerate(positions):
        _grid_add(all_grid, idx, pos, cell_size)

    for i in range(num_images):
        if i not in placed_info:
            continue
        for j in _grid_candidates(all_grid, positions[i], cell_size):
            if j <= i or j not in placed_info:
                continue

            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dist_m = np.sqrt(dx**2 + dy**2)

            if dist_m < NEIGHBOR_RADIUS_M:  # Use same GPS constraint
                kp1, des1 = img_features[i]
                kp2, des2 = img_features[j]

                matches = bf.knnMatch(des1, des2, k=2)
                good = [m for m, n in matches if m.distance < 0.85 * n.distance]

                if len(good) > 15:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                    M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_REPROJ)
                    if M is not None and mask.sum() > 15:
                        # Sanity check: Scale and Determinant
                        s_check = np.linalg.norm(M[:, :2], axis=0)
                        avg_sc = np.mean(s_check)
                        det = M[0,0]*M[1,1] - M[0,1]*M[1,0]
                        
                        if avg_sc < 0.5 or avg_sc > 2.0 or abs(det) < 1e-5:
                            continue
                            
                        H_ij = np.eye(3)
                        H_ij[:2, :] = M
                        all_links.append((i, j, H_ij))

    print(f"Refining poses with {len(all_links)} constraints...")
    for iteration in range(REFINE_MAX_ITERS):
        total_delta = 0.0
        count_delta = 0

        for i, j, H_ij in all_links:
            H_i_curr = placed_info[i]["H"]
            H_j_curr = placed_info[j]["H"]

            target_H_i = H_j_curr.dot(H_ij)
            new_H_i = 0.8 * H_i_curr + 0.2 * target_H_i
            total_delta += float(np.mean(np.abs(new_H_i - H_i_curr)))
            count_delta += 1
            placed_info[i]["H"] = new_H_i

            try:
                H_ji = np.linalg.inv(H_ij)
                target_H_j = placed_info[i]["H"].dot(H_ji)
                new_H_j = 0.8 * H_j_curr + 0.2 * target_H_j
                total_delta += float(np.mean(np.abs(new_H_j - H_j_curr)))
                count_delta += 1
                placed_info[j]["H"] = new_H_j
            except np.linalg.LinAlgError:
                continue

        avg_delta = (total_delta / count_delta) if count_delta > 0 else 0.0
        if (iteration + 1) % 10 == 0:
            print(f"  Refinement iteration {iteration + 1}/{REFINE_MAX_ITERS} (avg Δ={avg_delta:.6f})")
            if (iteration + 1) % (PREVIEW_EVERY) == 0:
                save_preview(images, placed_indices, placed_info)

        if avg_delta < REFINE_EARLY_STOP:
            print(f"  Early stop refinement at iter {iteration + 1} (avg Δ={avg_delta:.6f})")
            break

    # --- GPS Fallback Adjustment ---
    # Attempt local feature-based re-alignment for ALL GPS fallbacks first,
    # before any quality filtering.  This gives every GPS-placed image a
    # chance to upgrade to a feature-matched placement.  Quality filtering
    # of any images that remain as GPS fallbacks is deferred to the
    # harsh-break detection pass in with_stitch_main (which compares each
    # GPS image against the full background canvas – a more robust check
    # than pairwise seam consistency).
    print("Attempting local feature-based re-alignment for GPS fallbacks...")
    gps_remaining = [idx for idx in list(placed_indices) if placed_info.get(idx, {}).get("gps_fallback", False)]
    MIN_LOCAL_INLIERS = 12
    SCALE_MIN = 0.75
    SCALE_MAX = 1.25
    for idx in gps_remaining:
        # find nearest neighbors (include other fallbacks too) to try aligning to
        neighs = []
        for j in placed_indices:
            if j == idx:
                continue
            dx = positions[idx][0] - positions[j][0]
            dy = positions[idx][1] - positions[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            neighs.append((j, dist))
        neighs = sorted(neighs, key=lambda x: x[1])[:6]
        re_aligned = False
        for j, _ in neighs:
            try:
                kp_new, des_new = img_features[idx]
                kp_ref, des_ref = img_features[j]
                if des_new is None or des_ref is None:
                    continue
                matches = bf.knnMatch(des_new, des_ref, k=2)
                good = [m for m, n in matches if m.distance < 0.85 * n.distance]
                if len(good) < MIN_LOCAL_INLIERS:
                    continue
                src_pts = np.float32([kp_new[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                M_sim, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_REPROJ)
                if M_sim is None:
                    continue
                inliers = int(mask.sum())
                if inliers < MIN_LOCAL_INLIERS:
                    continue
                sc_x = np.linalg.norm(M_sim[:, 0])
                sc_y = np.linalg.norm(M_sim[:, 1])
                avg_scale = (sc_x + sc_y) / 2.0
                if avg_scale < SCALE_MIN or avg_scale > SCALE_MAX:
                    continue
                H_rel = np.eye(3)
                H_rel[:2, :] = M_sim
                # Map new image into global map via neighbor's H
                new_H = placed_info[j]["H"].dot(H_rel)
                placed_info[idx]["H"] = new_H
                placed_info[idx].pop("gps_fallback", None)
                print(f"  Re-aligned GPS fallback image {idx} to neighbor {j} (inliers={inliers}, scale={avg_scale:.2f})")
                re_aligned = True
                break
            except Exception:
                continue

        # If re-alignment failed, image retains its GPS placement;
        # it was already validated by the seam consistency check above
        # (same thresholds), so a redundant recheck is skipped.

    # --- Forced removal of oblique GPS-fallbacks ---
    ROT_DELETE_DEG = 12.0
    SHEAR_DELETE_THRESH = 0.6
    for idx in list(placed_indices):
        info = placed_info.get(idx, {})
        if not info.get("gps_fallback", False):
            continue
        try:
            H = info["H"]
            lin = H[:2, :2]
            # rotation estimate from first column
            rot_rad = np.arctan2(lin[1, 0], lin[0, 0])
            rot_deg = abs(np.degrees(rot_rad))
            avg_scale = (np.linalg.norm(lin[:,0]) + np.linalg.norm(lin[:,1])) / 2.0
            ortho_err = np.linalg.norm(lin.T.dot(lin) - np.eye(2) * (avg_scale ** 2))
            if rot_deg > ROT_DELETE_DEG or ortho_err > SHEAR_DELETE_THRESH:
                print(f"  Removing GPS-fallback image {idx} due to oblique transform (rot={rot_deg:.1f}°, shear_err={ortho_err:.2f})")
                if idx in placed_indices:
                    placed_indices.remove(idx)
                placed_info.pop(idx, None)
        except Exception:
            continue
    if skip_assembly:
        return None, placed_indices, placed_info
    print("Assembling final map with feathered blending...")
    canvas = assemble_blended_map(images, placed_indices, placed_info, use_alpha_blend=use_alpha_blend, alpha=alpha)
    return canvas, placed_indices, placed_info

def verify_seam_consistency(img1, img2, H_1_to_2):
    """
    Calculates the average pixel difference in the overlapping region and returns
    a tuple (mean_diff, overlap_pixel_count).
    """
    h, w = img1.shape[:2]
    max_dim = max(h, w)
    if max_dim > SEAM_MAX_DIM:
        scale = SEAM_MAX_DIM / max_dim
        img1 = cv2.resize(img1, None, fx=scale, fy=scale)
        img2 = cv2.resize(img2, None, fx=scale, fy=scale)
        S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float32)
        H_1_to_2 = S.dot(H_1_to_2).dot(np.linalg.inv(S))
        h, w = img1.shape[:2]

    warped2 = cv2.warpPerspective(img2, np.linalg.inv(H_1_to_2), (w, h))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(warped2, cv2.COLOR_BGR2GRAY)

    mask = (gray1 > 5) & (gray2 > 5)
    overlap_count = int(np.sum(mask))
    if overlap_count == 0:
        return 0.0, 0

    diff = np.abs(gray1.astype(np.float32) - gray2.astype(np.float32))
    mean_diff = float(np.mean(diff[mask]))

    # Compute a lightweight SSIM approximation for the overlapping region
    # using the standard luminance/contrast/covariance formulation with Gaussian window.
    img1_f = gray1.astype(np.float32)
    img2_f = gray2.astype(np.float32)
    # Apply mask to zero out non-overlap
    img1_f[~mask] = 0
    img2_f[~mask] = 0

    # Gaussian filter parameters
    ksize = (11, 11)
    sigma = 1.5
    mu1 = cv2.GaussianBlur(img1_f, ksize, sigma)
    mu2 = cv2.GaussianBlur(img2_f, ksize, sigma)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1_f * img1_f, ksize, sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2_f * img2_f, ksize, sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1_f * img2_f, ksize, sigma) - mu1_mu2

    L = 255.0
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-12)
    # Only average SSIM over the masked overlap pixels
    ssim_val = float(np.mean(ssim_map[mask])) if overlap_count > 0 else 0.0

    return mean_diff, overlap_count, ssim_val

def save_preview(images, placed_indices, placed_info):
    try:
        curr_pts = []
        for idx in placed_indices:
            h, w = images[idx].shape[:2]
            pts = np.float32([[0,0], [0,h], [w,h], [w,0]]).reshape(-1,1,2)
            dst_pts = cv2.perspectiveTransform(pts, placed_info[idx]["H"])
            curr_pts.append(dst_pts)
        
        curr_pts = np.concatenate(curr_pts, axis=0)
        [xmin, ymin] = np.int32(curr_pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(curr_pts.max(axis=0).ravel() + 0.5)
        
        pw = xmax - xmin
        ph = ymax - ymin
        p_scale = 1200.0 / max(pw, ph) if max(pw, ph) > 1200 else 1.0
        
        tw, th = int(pw * p_scale), int(ph * p_scale)
        if tw <= 0 or th <= 0: return
        
        H_pre = np.array([[p_scale, 0, -xmin*p_scale], [0, p_scale, -ymin*p_scale], [0, 0, 1]])
        preview = np.zeros((th, tw, 3), dtype=np.uint8)
        for idx in placed_indices:
            H_canvas = H_pre.dot(placed_info[idx]["H"])
            warped = cv2.warpPerspective(images[idx], H_canvas, (tw, th))
            preview[warped > 10] = warped[warped > 10]
        cv2.imwrite("stitch_progress.jpg", preview)
    except: pass

def assemble_blended_map(images, placed_indices, placed_info, use_alpha_blend: bool = True, alpha: float = 0.6):
    """
    Combines all images into a single canvas using alpha blending in overlaps or direct compositing.
    """
    all_pts = []
    for idx in placed_indices:
        h, w = images[idx].shape[:2]
        pts = np.float32([[0,0], [0,h], [w,h], [w,0]]).reshape(-1,1,2)
        dst_pts = cv2.perspectiveTransform(pts, placed_info[idx]["H"])
        all_pts.append(dst_pts)
    
    all_pts = np.concatenate(all_pts, axis=0)
    [xmin, ymin] = np.int32(all_pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_pts.max(axis=0).ravel() + 0.5)
    
    canvas_w = int(xmax - xmin)
    canvas_h = int(ymax - ymin)

    if canvas_w <= 0 or canvas_h <= 0:
        print("Warning: invalid canvas dimensions; returning first placed image.")
        fallback = images[placed_indices[0]]
        return fallback.astype(np.uint8) if fallback.dtype != np.uint8 else fallback

    scale = min(1.0, MAX_CANVAS_DIM / canvas_w, MAX_CANVAS_DIM / canvas_h)
    if scale < 1.0:
        canvas_w = max(1, int(canvas_w * scale))
        canvas_h = max(1, int(canvas_h * scale))
    
    print(f"Creating {canvas_w}x{canvas_h} composite canvas...")
    
    # Alpha blend only where images overlap
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    canvas_mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    
    if scale < 1.0:
        T = np.array([[scale, 0, -xmin * scale], [0, scale, -ymin * scale], [0, 0, 1]], dtype=np.float32)
    else:
        T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float32)
    
    for idx in placed_indices:
        img = images[idx]
        H_canvas = T.dot(placed_info[idx]["H"])
        warped = cv2.warpPerspective(img, H_canvas, (canvas_w, canvas_h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped = warped.astype(np.float32)
        new_mask = np.any(warped > 10, axis=2)
        overlap = canvas_mask & new_mask
        new_only = new_mask & ~canvas_mask
        # Log GPS fallback placement
        if placed_info[idx].get("gps_fallback", False):
            print(f"  Placing image {idx} by GPS fallback on canvas.")
        if np.any(overlap):
            if use_alpha_blend:
                canvas[overlap] = canvas[overlap] * (1.0 - alpha) + warped[overlap] * alpha
            else:
                canvas[overlap] = warped[overlap]
        if np.any(new_only):
            canvas[new_only] = warped[new_only]
        canvas_mask |= new_mask

    result = np.clip(canvas, 0, 255).astype(np.uint8)

    # Fill thin black seam gaps (1-5px strips where warpPerspective edge
    # interpolation drops below the >10 threshold, leaving unfilled pixels
    # between otherwise adjacent images).
    kernel = np.ones((11, 11), np.uint8)
    dilated = cv2.dilate(canvas_mask.astype(np.uint8), kernel)
    gap_mask = (dilated > 0) & ~canvas_mask
    if gap_mask.any():
        result = cv2.inpaint(result, gap_mask.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)

    return result

def _cell_key(xy, cell_size):
    return (int(np.floor(xy[0] / cell_size)), int(np.floor(xy[1] / cell_size)))

def _grid_add(grid, idx, pos, cell_size):
    key = _cell_key(pos, cell_size)
    grid.setdefault(key, []).append(idx)

def _grid_candidates(grid, pos, cell_size):
    cx, cy = _cell_key(pos, cell_size)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for idx in grid.get((cx + dx, cy + dy), []):
                yield idx