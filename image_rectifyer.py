import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
from imageio import imread

# Load image
img = imread('initial_image.jpg')

# *** Rotation Matrix Functions ***

# Matrix for Yaw-rotation about the Z-axis
def R_z(psi):
    return np.array([[np.cos(np.radians(psi)), -np.sin(np.radians(psi)), 0],
                     [np.sin(np.radians(psi)), np.cos(np.radians(psi)),  0],
                     [0, 0, 1]])

# Matrix for Pitch-rotation about the Y-axis
def R_y(theta):
    return np.array([[np.cos(np.radians(theta)), 0, np.sin(np.radians(theta))],
                     [0, 1, 0],
                     [-np.sin(np.radians(theta)), 0, np.cos(np.radians(theta))]])

# Matrix for Roll-rotation about the X-axis
def R_x(phi):
    return np.array([[1, 0, 0],
                     [0, np.cos(np.radians(phi)), -np.sin(np.radians(phi))],
                     [0, np.sin(np.radians(phi)), np.cos(np.radians(phi))]])

# Full rotation matrix (Z-axis not used)
R_rot = np.dot(np.dot(R_y(-60), R_x(10)), R_z(0))

# Strip Z-axis values from R_rot to create 2D transformation matrix
R_2d = np.array([[R_rot[0, 0], R_rot[0, 1], 0],
                 [R_rot[1, 0], R_rot[1, 1], 0],
                 [0, 0, 1]])

# Generate transformation matrix
tform = np.linalg.inv(R_2d)  # Inverse for affine_transform

# Warp the image using the affine transformation
output_image = affine_transform(img, tform[:2, :2])

# Display image
plt.figure(1)
plt.imshow(output_image)
plt.show()