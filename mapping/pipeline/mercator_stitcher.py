# ============== Work in Progress ==========================

from abstract_stitcher import AbstractStitcher
import numpy as np
import cv2
import os
import glob
import json
from PIL import Image
import math

CANVAS_PADDING = 1.2 # Pads the generated canvas width/height to CANVAS_PADDING * dimensions of the minimum bounding box
PYR_N = 6 # depth of the pyramids for Laplacian blending

# Heuristically picks a base height (meters) for resizing purposes.
def get_base_height(img_dicts):
    """
    WIP:    Chooses a standard height to scale the images towards.
            Could pick height with minimum rounding error.
    """
    return 100

# Resizes images to be directly correlated to altitude.
def scale_images(img, altitude, base_height):
    ratio = base_height / altitude
    h, w, _ = img.shape
    h, w = int(ratio * float(h)), int(ratio * float(w))
    return img.resize((w, h), Image.ANTIALIAS)

# TODO: implement pix_to_lonlat, use platform code getPixelCoordinates
def pix_to_lonlat(img_data, pixel_x, pixel_y):
    return [-1, -1]

def _get_rotated_pixels(points, matrix):
    pts = np.array([points])
    transformed_pts = cv2.transform(pts, matrix)
    return transformed_pts[0]

# TODO: implement rotating, then the four pixel corners of the rotated image.
def rotate_img(img_data):
    yaw = -img_data['yaw']
    img = img_data['img']
    h, w, _ = img.shape
    lonlat = img_data['lng'], img_data['lat']
    
    center_y, center_x = h // 2, w // 2
    rot_matrix = cv2.getRotationMatrix2D((center_x, center_y), yaw, 1.0)

    # take abs values of sin and cos from rot_matrix
    rot_cos = np.abs(rot_matrix[0][0])
    rot_sin = np.abs(rot_matrix[0][1])

    # compute height, width of bounding box of rotated img - to be used in warpAffine
    bounding_w = int((h * rot_sin) + (w * rot_cos))
    bounding_h = int((h * rot_cos) + (w * rot_sin))

    # update rot_matrix values with bounding box height, width
    rot_matrix[0][2] += (bounding_w / 2) - center_x
    rot_matrix[1][2] += (bounding_h / 2) - center_y

    rotated = cv2.warpAffine(img, rot_matrix, (bounding_w, bounding_h))
    # WARNING experimental buffer
    buf = 100
    corners = _get_rotated_pixels(
        [[buf, buf], [w-buf-1, buf], 
        [w-buf-1, h-buf-1], [buf, h-buf-1]], rot_matrix)
    
    return rotated, lonlat, corners
    
use_mask = True 

class MercatorStitcher(AbstractStitcher):
    def __init__(self):
        pass

    def get_canvas(img_dicts):
        """
        Returns a tuple ((w, h), (center_lon, center_lat), lonlat_per_pix), which is the shape of the canvas,
        longitude/latidude of the center, and the scaling factor of longitude/latitude distance per
        pixel value of the canvas.
        """
        lonlat_per_pix = 0.0
        mins = 185, 185
        maxes = -185, -185
        for img_data in img_dicts:
            img = img_data['img']
            h, w, _ = img.shape
            TL = pix_to_lonlat(img, 0, 0)
            TR = pix_to_lonlat(img, w-1, 0)
            BL = pix_to_lonlat(img, 0, h-1)
            BR = pix_to_lonlat(img, w-1, h-1)
            # WARNING: doesn't work across long/lat boundaries (-179 to 179)
            mins = np.amin(np.vstack((TL, TR, BL, BR, mins)), axis=0)
            maxes = np.amax(np.vstack((TL, TR, BL, BR, maxes)), axis=0)
            lonlat_per_pix += math.dist(pix_to_lonlat(img, 0, 0), pix_to_lonlat(img, w-1, 0))
        lonlat_per_pix = lonlat_per_pix / len(img_dicts)
        w, h = maxes[0] - mins[0], maxes[1] - mins[1]
        center = np.average((maxes[0], mins[0])), np.average((maxes[1], mins[1]))

        # TODO: Delete following 3 lines when pix_to_lonlat and altitude readings are functional
        lonlat_per_pix = 0.0000002
        w, h = 0.0003, 0.0002
        center = -76.4418, 42.44439

        canvas_w, canvas_h = int(w / lonlat_per_pix * CANVAS_PADDING), int(h / lonlat_per_pix * CANVAS_PADDING)
        return (canvas_w, canvas_h), center, lonlat_per_pix

    def place(map, map_mask, img, img_mask, use_mask):
        """ 
            Stitches an image onto a map. 
            Precondition: map_mask and img_mask have a null intersection. All 4 inputs have the same shape.
            Returns: New stitched mask.
        """
        if not use_mask:
            for j in range(3):
                img_mask = img_mask * ~map_mask
                map[:,:,j] = map[:,:, j] * map_mask
                img[:,:,j] = img[:,:, j] * img_mask
            return map + img
        # Calculate the Gaussian pyramids and downsampled masks
        G_map = map.copy()
        G_img = img.copy()
        G_map_mask = map_mask.copy()
        G_img_mask = img_mask.copy()
        pyr_map = [G_map]
        pyr_img = [G_img]
        pyr_map_mask = [G_map_mask]
        pyr_img_mask = [G_img_mask]
        for i in range(PYR_N):
            h, w = G_img.shape[:2]
            G_img = cv2.resize(G_img, (h//2, w//2))
            G_map = cv2.resize(G_map, (h//2, w//2))
            pyr_img.append(G_img)
            pyr_map.append(G_map)

            shape = G_img.shape[1], G_img.shape[0]
            G_map_mask = cv2.resize(np.float32(G_map_mask), shape, cv2.INTER_LINEAR) > 0 
            G_img_mask = cv2.resize(np.float32(G_img_mask), shape, cv2.INTER_LINEAR) > 0 

            pyr_map_mask.append(G_map_mask) 
            pyr_img_mask.append(G_img_mask)

        # Calculate Laplacian pyramids
        lapl_map = [pyr_map[PYR_N-1]]
        lapl_img = [pyr_img[PYR_N-1]]
        for i in range(PYR_N-1,0,-1):
            size = pyr_img[i-1].shape[1], pyr_img[i-1].shape[0]

            G_img_upper = cv2.resize(pyr_img[i], size)
            L_img = cv2.subtract(pyr_img[i-1],G_img_upper)
            lapl_img.append(L_img)
            G_map_upper = cv2.resize(pyr_map[i], size)
            L_map = cv2.subtract(pyr_map[i-1],G_map_upper)
            lapl_map.append(L_map)
            
        # Add the Laplacians
        LS = []

        for i in range(PYR_N):
            map_mask = pyr_map_mask[PYR_N-1-i]
            img_mask = pyr_img_mask[PYR_N-1-i] * ~map_mask
            ls_map = lapl_map[i] 
            ls_img = lapl_img[i]
            for j in range(3):
                ls_map[:,:, j] = ls_map[:,:, j] * map_mask
                ls_img[:,:, j] = ls_img[:,:, j] * img_mask

            ls = ls_map + ls_img
            LS.append(ls)
            
        stitched_map = LS[0]
        for i in range(1, PYR_N):
            size = LS[i].shape[1], LS[i].shape[0]
            stitched_map = cv2.resize(stitched_map, size)
            stitched_map = cv2.add(stitched_map, LS[i])
        
        return stitched_map
    
    def stitch (self, img_dicts):
        # Scale images
        base_height = get_base_height(img_dicts)
        # WARNING: uncomment when altitude readings are usable
        # for img_data in img_dicts:
        #     img_data['img'] = scale_images(img_data['img'], img_data['altitude'], base_height)
        # Get canvas dimensions, center of canvas (in long/lat), and the long/lat distance per pixel.
        (canvas_w, canvas_h), (canvas_cx, canvas_cy), lonlat_per_pix = self.get_canvas(img_dicts)
        canvas = np.zeros((canvas_h, canvas_w, 3),dtype=np.uint8)

        # Rotate images
        rotated_imgs = []
        for img_data in img_dicts:
            rotated_imgs.append(rotate_img(img_data))

        # Calculate masks, place images and mask onto canvas
        map_mask = np.zeros(canvas.shape[:2],dtype=bool)
        stitched_map = canvas.copy()

        for rotated_img in rotated_imgs:
            img, (image_cx, image_cy), corners = rotated_img
            h, w, _ = img.shape
            new_x = int(canvas_w/2 + int((image_cx - canvas_cx)/lonlat_per_pix) - w/2)
            new_y = int(canvas_h/2 - int((image_cy - canvas_cy)/lonlat_per_pix) - h/2)

            img_mask = np.mean(img, -1)>0
            placed_image = canvas.copy()
            placed_image[new_y: new_y + h, new_x:new_x + w] = img
            placed_mask = np.zeros((canvas_h, canvas_w),dtype=np.bool)
            placed_mask[new_y: new_y + h, new_x:new_x + w] = img_mask
            # placed_image = placed_image * placed_mask
            placed_mask = placed_mask
            stitched_map = self.place(stitched_map, map_mask, placed_image, placed_mask, use_mask)

            map_mask = map_mask | placed_mask
            # stitched_map = np.bitwise_and(stitched_map, map_mask)

        # Muting:
        stitched_map = np.int32(np.float32(stitched_map) / 1.3)
        return stitched_map
 