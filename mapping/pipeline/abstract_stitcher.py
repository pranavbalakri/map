import numpy as np
import cv2
from abc import abstractmethod


class AbstractStitcher():
    """
    stitch(self, img_dicts) takes in a list of dicts of the form:
    { img : np.ndarray,
      yaw : float,
      lat : float,
      lng : float,
      altitude : float }
      then returns a single numpy image of 16:9 aspect ratio of the images stitched together.
    """
    @abstractmethod
    def stitch(self, img_dicts) -> np.ndarray:
        raise NotImplementedError

class DefaultStitcher(AbstractStitcher):
    def stitch(self, img_dicts):
        return np.zeros((9, 16, 3), dtype = "uint8")

