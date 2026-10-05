import sys
import cv2
import os
import numpy as np
import csv
from typing import List, Dict, Tuple
cv2.ocl.setUseOpenCL(False)
from stitcher import stitch_geolocated_images

REFERENCE_ALTITUDE = 100

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
    
    return metadata, 42, -76

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
    final_scale = min(alt_scale, current_scale)
    
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
    
    Args:
    image (np.ndarray): The input image.
    
    Returns:
    np.ndarray: The normalized image.
    """
    # Convert to LAB color space for better color/lighting separation
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=6, tileGridSize=(4,4))
    l = clahe.apply(l)

    l = cv2.multiply(l, 0.75)
    
    # Merge channels
    lab = cv2.merge([l, a, b])
    
    # Convert back to BGR
    normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return normalized

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

def stitch_images_in_folder(folder_path: str) -> np.ndarray:
    """
    Stitches together all the images in a given folder using their center locations and rotates them to align north.
    """
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        raise ValueError("No image files found in the specified folder.")
    
    csv_path = folder_path + ".csv"
    metadata, ref_lat, ref_lon = read_csv(csv_path)
    
    # First pass: Process all images
    processed_images = []
    # coordinates = []
    for image_file in image_files:
        if image_file not in metadata:
            continue
            
        img_path = os.path.join(folder_path, image_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        lat, lon, altitude, rotation_angle = metadata[image_file]
        
        # Add normalization step before resizing
        normalized_img = normalize_image(img)
        resized_img = resize_image(normalized_img, altitude)
        rotated_img = rotate_image(resized_img, float(rotation_angle))
        
        processed_images.append({
            'image': rotated_img,
            'lat': lat,
            'lon': lon,
            'dimensions': rotated_img.shape[:2]  # (height, width)
        })
    
    if not processed_images:
        raise ValueError("No valid images found with corresponding center locations.")
    
    # Calculate pixel coordinates for each image
    for img_data in processed_images:
        pixel_x, pixel_y = coordinates_to_pixels(img_data['lat'], img_data['lon'], ref_lat, ref_lon)
        h, w = img_data['dimensions']
        img_data['position'] = {
            'left': pixel_x - (w // 2),
            'right': pixel_x + (w - w // 2),
            'top': pixel_y - (h // 2),
            'bottom': pixel_y + (h - h // 2),
            'center_x': pixel_x,
            'center_y': pixel_y
        }
    
    # Calculate canvas dimensions
    min_x = min(img['position']['left'] for img in processed_images)
    max_x = max(img['position']['right'] for img in processed_images)
    min_y = min(img['position']['top'] for img in processed_images)
    max_y = max(img['position']['bottom'] for img in processed_images)
    
    canvas_width = max_x - min_x
    canvas_height = max_y - min_y
    
    # Create and fill the canvas
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    for img_data in processed_images:
        img = img_data['image']
        pos = img_data['position']

        # Calculate placement coordinates
        x1 = pos['left'] - min_x
        y1 = pos['top'] - min_y
        x2 = x1 + img.shape[1]
        y2 = y1 + img.shape[0]
        
        # Create mask for non-black pixels
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 10
        
        # Calculate source image coordinates
        img_x1 = max(0, -x1)
        img_y1 = max(0, -y1)
        img_x2 = img.shape[1] - max(0, x2 - canvas_width)
        img_y2 = img.shape[0] - max(0, y2 - canvas_height)
        
        # Copy only non-black pixels with transparency
        region = mask[img_y1:img_y2, img_x1:img_x2]
        alpha = 0.9 # Opacity
        for c in range(3):
            canvas[y1:y2, x1:x2, c][region] = (
                alpha * img[img_y1:img_y2, img_x1:img_x2, c][region] + 
                (1 - alpha) * canvas[y1:y2, x1:x2, c][region]
            ).astype(np.uint8)

    # Draw center points
    # for image_name, (lat, lon, _, _) in metadata.items():
    #     if image_name in image_files:
    #         pixel_x, pixel_y = coordinates_to_pixels(lat, lon, ref_lat, ref_lon)
    #         adjusted_x = pixel_x - min_x
    #         adjusted_y = pixel_y - min_y
            
    #         if 0 <= adjusted_x < canvas_width and 0 <= adjusted_y < canvas_height:
    #             cv2.circle(canvas, (adjusted_x, adjusted_y), 10, (0, 0, 255), -1)
    
    return canvas

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the path to the image folder as an argument.")
        sys.exit(1)

    folder_path = sys.argv[1]
    result = stitch_images_in_folder(folder_path)
    save_output_image("output.jpg", result)