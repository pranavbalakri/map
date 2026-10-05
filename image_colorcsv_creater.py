import cv2
import csv
import numpy as np

# Load the image
img = cv2.imread('Screenshot 2024-12-06 at 5.58.30 PM.png')
if img is None:
    raise ValueError("Could not load stitched_live.jpg")

# Get image dimensions
height, width = img.shape[:2]

# Create CSV file
with open('stitched_colors.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write header
    writer.writerow(['x', 'y', 'blue', 'green', 'red'])
    
    # Write color values for each pixel
    for y in range(height):
        for x in range(width):
            b, g, r = img[y, x]
            writer.writerow([x, y, int(b), int(g), int(r)])

print(f"Created CSV with {height*width} pixel color values")
