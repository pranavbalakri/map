import cv2
import numpy as np,sys
from abstract_stitcher import AbstractStitcher

class OpenCVStitcher(AbstractStitcher):
    def stitch(self, img_dicts):
        imgs = list(map(lambda dict: dict['img'], img_dicts))
        stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
        try:
            _, pano = stitcher.stitch(imgs)
        except Exception:
            return imgs[0]
        return pano
