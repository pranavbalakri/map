import cv2
import os
import numpy as np
import csv
from typing import List, Dict, Tuple
from multiprocessing import Pool
cv2.ocl.setUseOpenCL(False)
from stitcher import stitch_geolocated_images, assemble_blended_map, MAX_CANVAS_DIM

REFERENCE_ALTITUDE = 100
NORMALIZE_MAX_DIM = 1200
DEFAULT_WORKERS = os.cpu_count() or 4
# Horizontal field of view of the drone camera in degrees.
# 76° is typical for DJI Mavic/Phantom. Adjust for your specific model.
CAMERA_HFOV_DEG = 76.0
# Max dimension images are resized to in resize_image()
IMAGE_MAX_DIM = 800

# Harsh break detection thresholds
HARSH_BREAK_MIN_BOUNDARY_PX = 50     # ignore images with very few boundary pixels
HARSH_BREAK_MAX_PASSES = 5           # max iterative removal passes

def read_csv(csv_path: str) -> Tuple[Dict[str, Tuple[float, float, float, float]], float, float]:
    """
    Reads the center locations and directions from the CSV file.
    Also calculates mean latitude and longitude as reference points.
    
    Args:
    csv_path (str): Path to the CSV file containing center locations and directions.
    
    Returns:
    Tuple[Dict[str, Tuple[float, float, float, float]], float, float]: Dictionary mapping image filenames to their coordinates,
    altitude, and direction, plus mean lat/lon reference points.
    """
    metadata = {}
    
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            image = row['Image']
            lat = float(row['Latitude'])
            lon = float(row['Longitude'])
            altitude = float(row['Altitude'])
            direction = float(row['Degrees_Clockwise_from_North'])
            metadata[image] = (lat, lon, altitude, direction)
    
    lats = [v[0] for v in metadata.values()]
    lons = [v[1] for v in metadata.values()]
    ref_lat = sum(lats) / len(lats)
    ref_lon = sum(lons) / len(lons)
    return metadata, ref_lat, ref_lon

def rotate_image(image: np.ndarray, rotation_angle: float) -> np.ndarray:
    """
    Rotates the image based on the given angle to align north to the top.
    
    Args:
    image (np.ndarray): The input image.
    rotation_angle (int): The rotation angle in degrees from north clockwise.
    
    Returns:
    np.ndarray: The rotated image.
    """
    # Convert clockwise angle to counter-clockwise for rotation
    rotation_angle = -rotation_angle
    
    if rotation_angle == 0:
        return image
    elif rotation_angle == 180 or rotation_angle == -180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif rotation_angle == -90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        # Calculate the dimensions of the new image
        rows, cols = image.shape[:2]
        # Get the rotation matrix
        M = cv2.getRotationMatrix2D((cols/2, rows/2), rotation_angle, 1)
        
        # Calculate new image dimensions
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_cols = int((rows * sin) + (cols * cos))
        new_rows = int((rows * cos) + (cols * sin))
        
        # Adjust the rotation matrix
        M[0, 2] += (new_cols / 2) - cols/2
        M[1, 2] += (new_rows / 2) - rows/2
        
        # Perform the rotation with the new dimensions
        return cv2.warpAffine(image, M, (new_cols, new_rows))

def resize_image(image: np.ndarray, altitude: float) -> np.ndarray:
    """
    Zooms the image based on the given zoom factor.
    
    Args:
    image (np.ndarray): The input image.
    altitude (float): The altitude of the drone in meters.
    
    Returns:
    np.ndarray: The zoomed image.
    """
    if altitude == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    
    # Calculate scale to normalize altitude
    alt_scale = altitude / REFERENCE_ALTITUDE
    
    # Check dimensions and enforce max size to prevent OOM/Size limits during stitching
    h, w = image.shape[:2]
    max_dim = 800 # Limit max dimension
    
    current_scale = max_dim / max(h, w)
    
    # Use the smaller of the two scales (altitude normalization vs max size cap)
    # Actually, we should probably just normalize to specific size for mapping to work reasonably
    # Let's combine them: First normalize to altitude, THEN cap size?
    # Or just cap the reference altitude size.
    
    final_scale = min(alt_scale, current_scale)
    
    # Ensure we don't upscale too much if altitude is crazy high (unlikely)
    # But usually we downscale.
    
    return cv2.resize(image, None, fx=final_scale, fy=final_scale)

def coordinates_to_pixels(lat: float, lon: float, ref_lat: float, ref_lon: float) -> Tuple[int, int]:
    """
    Converts the latitude/longitude coordinates to pixel coordinates with improved accuracy.
    """
    # Earth's circumference in meters approx 40,075,000 meters
    # We need to define a scale. Let's assume the previous scale was based on roughly 111,111 meters per degree lat.
    # Previous: 3,200,000 pixels / degree lat.
    # So Scale factor ~ 28.8 pixels per meter.
    
    METERS_PER_DEGREE_LAT = 111132.954 - 559.822 * np.cos(2 * lat * np.pi / 180) + 1.175 * np.cos(4 * lat * np.pi / 180)
    METERS_PER_DEGREE_LON = 111132.954 * np.cos(lat * np.pi / 180)
    
    # Use a fixed scale factor (pixels per meter) to maintain aspect ratio
    # Derived from original constants: 3200000 / 111000 ~ 28.8
    PIXELS_PER_METER = 28.8 
    
    pixels_per_degree_lat = METERS_PER_DEGREE_LAT * PIXELS_PER_METER
    pixels_per_degree_lon = METERS_PER_DEGREE_LON * PIXELS_PER_METER
    
    pixel_lat = (lat - ref_lat) * pixels_per_degree_lat
    pixel_lon = (lon - ref_lon) * pixels_per_degree_lon
    
    pixel_x = int(pixel_lon)
    pixel_y = int(-pixel_lat)
    return (pixel_x, pixel_y)

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalizes the exposure and white balance of an image.
    """
    h, w = image.shape[:2]
    scale = 1.0
    if max(h, w) > NORMALIZE_MAX_DIM:
        scale = NORMALIZE_MAX_DIM / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=6, tileGridSize=(4,4))
    l = clahe.apply(l)
    l = cv2.multiply(l, 0.75)

    lab = cv2.merge([l, a, b])
    normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if scale != 1.0:
        normalized = cv2.resize(normalized, (w, h))
    return normalized

def _process_one_image(args: Tuple[str, str, Dict, bool]) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Module-level function for multiprocessing (must be picklable).
    Processes a single image: normalize, resize, rotate.
    
    Args:
        args: Tuple of (image_file, folder_path, metadata, normalize_flag)
    
    Returns:
        Tuple of (processed_image, (lat, lon)) or None if processing fails
    """
    image_file, folder_path, metadata, normalize_flag = args
    
    if image_file not in metadata:
        return None

    img_path = os.path.join(folder_path, image_file)
    img = cv2.imread(img_path)
    if img is None:
        return None

    lat, lon, altitude, rotation_angle = metadata[image_file]

    if normalize_flag:
        img = normalize_image(img)
    resized_img = resize_image(img, altitude)
    rotated_img = rotate_image(resized_img, float(rotation_angle))

    return rotated_img, (lat, lon)

def _compute_canvas_transform(images, placed_indices, placed_info):
    """Recompute the translation matrix used by assemble_blended_map."""
    all_pts = []
    for idx in placed_indices:
        h, w = images[idx].shape[:2]
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst_pts = cv2.perspectiveTransform(pts, placed_info[idx]["H"])
        all_pts.append(dst_pts)
    all_pts_cat = np.concatenate(all_pts, axis=0)
    [xmin, ymin] = np.int32(all_pts_cat.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_pts_cat.max(axis=0).ravel() + 0.5)
    raw_w = int(xmax - xmin)
    raw_h = int(ymax - ymin)
    scale = min(1.0, MAX_CANVAS_DIM / raw_w, MAX_CANVAS_DIM / raw_h)
    if scale < 1.0:
        canvas_w = max(1, int(raw_w * scale))
        canvas_h = max(1, int(raw_h * scale))
        T = np.array([[scale, 0, -xmin * scale],
                      [0, scale, -ymin * scale],
                      [0, 0, 1]], dtype=np.float32)
    else:
        canvas_w = raw_w
        canvas_h = raw_h
        T = np.array([[1, 0, float(-xmin)],
                      [0, 1, float(-ymin)],
                      [0, 0, 1]], dtype=np.float32)
    return T, scale, canvas_w, canvas_h


def detect_harsh_breaks(images: List[np.ndarray],
                        placed_indices: List[int], placed_info: dict,
                        min_boundary_px: int = HARSH_BREAK_MIN_BOUNDARY_PX,
                        bg_cache: tuple = None) -> Tuple[List[int], tuple]:
    """
    Detects GPS-fallback images whose placement creates harsh visual seams.

    Only GPS-fallback images are checked, since feature-matched images are
    already validated by the matching + RANSAC pipeline.

    For each GPS-fallback image we render the canvas WITHOUT that image
    (using only nearby overlapping images), then compare:
      - The warped image content in the overlap region
      - The canvas content from other images in the same region
    using SSIM and mean absolute difference.  Images with very low overlap
    similarity are flagged for removal.

    An absolute threshold is used (not adaptive z-scores) to avoid cascade
    removal of good edge images.

    Args:
        bg_cache: Optional cached tuple
                  (bg_canvas, bg_mask, bg_gray, T, canvas_w, canvas_h)
                  built from non-GPS images.  Because non-GPS images never
                  change between harsh-break passes, the cache avoids
                  re-warping ~180+ images on every iteration.

    Returns:
        Tuple of (list of image indices to remove, updated bg_cache).
    """
    # Identify GPS-fallback images to check
    gps_indices = [idx for idx in placed_indices
                   if placed_info[idx].get("gps_fallback", False)]
    if not gps_indices:
        print("  No GPS-fallback images on canvas; skipping check.")
        return [], bg_cache

    print(f"  Checking {len(gps_indices)} GPS-fallback image(s)...")

    # Reuse cached background canvas or build from scratch.
    # The background (non-GPS images) does not change between passes,
    # so caching avoids re-warping all feature-matched images each iteration.
    if bg_cache is not None:
        bg_canvas, bg_mask, bg_gray, T, canvas_w, canvas_h = bg_cache
    else:
        T, _scale, canvas_w, canvas_h = _compute_canvas_transform(
            images, placed_indices, placed_info)

        non_gps_indices = [idx for idx in placed_indices
                           if not placed_info[idx].get("gps_fallback", False)]

        bg_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        bg_mask = np.zeros((canvas_h, canvas_w), dtype=bool)
        for idx in non_gps_indices:
            img = images[idx]
            H_canvas = T.dot(placed_info[idx]["H"])
            warped = cv2.warpPerspective(img, H_canvas, (canvas_w, canvas_h),
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            warped_f = warped.astype(np.float32)
            new_mask = np.any(warped > 10, axis=2)
            overlap = bg_mask & new_mask
            new_only = new_mask & ~bg_mask
            if np.any(overlap):
                bg_canvas[overlap] = warped_f[overlap]  # overwrite (simple)
            if np.any(new_only):
                bg_canvas[new_only] = warped_f[new_only]
            bg_mask |= new_mask

        bg_gray = cv2.cvtColor(bg_canvas.clip(0, 255).astype(np.uint8),
                               cv2.COLOR_BGR2GRAY).astype(np.float32)

    bg_cache_out = (bg_canvas, bg_mask, bg_gray, T, canvas_w, canvas_h)

    # --- Check each GPS-fallback image against the background ---
    # Thresholds (absolute):
    SSIM_REMOVE_THRESH = 0.25       # below this -> harsh break
    DIFF_REMOVE_THRESH = 50.0       # above this -> harsh break
    MIN_OVERLAP_FRAC = 0.10         # need at least 10% overlap with bg to judge

    to_remove = []

    for idx in gps_indices:
        img = images[idx]
        H_canvas = T.dot(placed_info[idx]["H"])

        # Warp this GPS image onto the canvas
        warped = cv2.warpPerspective(img, H_canvas, (canvas_w, canvas_h),
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_mask = np.any(warped > 10, axis=2)
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Overlap: where BOTH this image and the background have content
        overlap = warped_mask & bg_mask
        n_overlap = int(np.sum(overlap))
        n_image = int(np.sum(warped_mask))

        if n_image == 0:
            continue

        overlap_frac = n_overlap / n_image
        if overlap_frac < MIN_OVERLAP_FRAC or n_overlap < min_boundary_px:
            # Not enough overlap to judge; skip (benefit of the doubt)
            print(f"  Image {idx} [GPS]: insufficient overlap ({overlap_frac:.0%}), skipping.")
            continue

        # Mean absolute difference in the overlap
        diff = np.abs(warped_gray[overlap] - bg_gray[overlap])
        mean_diff = float(np.mean(diff))

        # SSIM in the overlap region (block-based for efficiency)
        img_region = warped_gray.copy()
        bg_region = bg_gray.copy()
        # Zero out non-overlap regions
        mask_f = overlap.astype(np.float32)
        img_region[~overlap] = 0
        bg_region[~overlap] = 0

        ksize = (11, 11)
        sigma = 1.5
        mu1 = cv2.GaussianBlur(img_region, ksize, sigma)
        mu2 = cv2.GaussianBlur(bg_region, ksize, sigma)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.GaussianBlur(img_region * img_region, ksize, sigma) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(bg_region * bg_region, ksize, sigma) - mu2_sq
        sigma12 = cv2.GaussianBlur(img_region * bg_region, ksize, sigma) - mu1_mu2
        L = 255.0
        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-12)
        ssim_val = float(np.mean(ssim_map[overlap])) if n_overlap > 0 else 1.0

        is_bad = (ssim_val < SSIM_REMOVE_THRESH) or (mean_diff > DIFF_REMOVE_THRESH)

        status = "REMOVE" if is_bad else "ok"
        print(f"  Image {idx} [GPS]: overlap={overlap_frac:.0%}, "
              f"diff={mean_diff:.1f}, ssim={ssim_val:.3f} -> {status}")

        if is_bad:
            to_remove.append(idx)

    return to_remove, bg_cache_out


def save_output_image(path: str, image: np.ndarray) -> None:
    if image is None or image.size == 0:
        raise ValueError("Output image is empty; nothing to write.")

    output = image
    if output.dtype in (np.float32, np.float64):
        max_val = float(np.max(output)) if output.size else 0.0
        if max_val <= 1.0:
            output = (output * 255.0).clip(0, 255).astype(np.uint8)
        else:
            output = np.clip(output, 0, 255).astype(np.uint8)
    elif output.dtype != np.uint8:
        output = np.clip(output, 0, 255).astype(np.uint8)

    if np.max(output) == 0:
        print("Warning: output image is fully black.")

    if not cv2.imwrite(path, output):
        raise IOError(f"Failed to write image to {path}.")

def stitch_images_in_folder(folder_path: str, normalize: bool = True, use_alpha_blend: bool = True) -> np.ndarray:
    """
    Stitches together all the images in a given folder using their center locations and rotates them to align north.
    """
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    
    if not image_files:
        raise ValueError("No image files found in the specified folder.")
    
    csv_path = folder_path + ".csv"
    metadata, ref_lat, ref_lon = read_csv(csv_path)
    
    processed_images = []
    coordinates = []

    # Prepare arguments for multiprocessing: each item is (image_file, folder_path, metadata, normalize)
    process_args = [(img_file, folder_path, metadata, normalize) for img_file in image_files]

    with Pool(processes=DEFAULT_WORKERS) as pool:
        for result in pool.map(_process_one_image, process_args):
            if result is None:
                continue
            img, coord = result
            processed_images.append(img)
            coordinates.append(coord)

    if not processed_images:
        raise ValueError("No valid images found with corresponding center locations.")
    
    # Calculate pixel coordinates for each image
    # for img_data in processed_images:
    #     pixel_x, pixel_y = coordinates_to_pixels(img_data['lat'], img_data['lon'], ref_lat, ref_lon)
    #     h, w = img_data['dimensions']
    #     img_data['position'] = {
    #         'left': pixel_x - (w // 2),
    #         'right': pixel_x + (w - w // 2),
    #         'top': pixel_y - (h // 2),
    #         'bottom': pixel_y + (h - h // 2),
    #         'center_x': pixel_x,
    #         'center_y': pixel_y
    #     }
    
    mean_altitude = sum(v[2] for v in metadata.values()) / len(metadata)
    ground_width_m = 2.0 * mean_altitude * np.tan(np.radians(CAMERA_HFOV_DEG / 2.0))
    ppm = IMAGE_MAX_DIM / ground_width_m
    print(f"Using PPM={ppm:.2f} (mean altitude={mean_altitude:.1f}m, HFOV={CAMERA_HFOV_DEG}°)")

    _, placed_indices, placed_info = stitch_geolocated_images(
        processed_images, coordinates, use_alpha_blend=use_alpha_blend,
        skip_assembly=True, ppm=ppm
    )
    if not placed_indices:
        print("Stitching failed.")
        return np.zeros((100, 100, 3), dtype=np.uint8)

    # --- Final harsh-break removal loop ---
    # bg_cache is reused across passes: the background canvas (non-GPS
    # images) never changes, so we avoid re-warping ~180+ images each time.
    bg_cache = None
    for pass_num in range(1, HARSH_BREAK_MAX_PASSES + 1):
        print(f"\n=== Harsh-break detection pass {pass_num} ===")
        bad, bg_cache = detect_harsh_breaks(
            processed_images, placed_indices, placed_info,
            bg_cache=bg_cache
        )
        if not bad:
            print("No harsh breaks detected — map looks clean.")
            break

        print(f"Removing {len(bad)} image(s) with harsh breaks: {bad}")
        for idx in bad:
            if idx in placed_indices:
                placed_indices.remove(idx)
            placed_info.pop(idx, None)

        if not placed_indices:
            print("All images removed — returning empty canvas.")
            return np.zeros((100, 100, 3), dtype=np.uint8)
    else:
        print(f"Reached max harsh-break passes ({HARSH_BREAK_MAX_PASSES}); "
              f"some artefacts may remain.")

    # Single final canvas assembly (replaces per-pass rebuilds)
    canvas = assemble_blended_map(
        processed_images, placed_indices, placed_info,
        use_alpha_blend=use_alpha_blend
    )
    return canvas

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stitch DJI drone images into a map.")
    parser.add_argument("folder", help="Path to folder containing images")
    parser.add_argument("--output", default="output.jpg", help="Output file path (default: output.jpg)")
    parser.add_argument("--alpha", action="store_true", default=True, help="Use alpha blending for overlaps (default: on)")
    parser.add_argument("--no-alpha", dest="alpha", action="store_false", help="Disable alpha blending")
    args = parser.parse_args()

    result = stitch_images_in_folder(args.folder, normalize=True, use_alpha_blend=args.alpha)
    save_output_image(args.output, result)