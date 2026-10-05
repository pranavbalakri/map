import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
import shutil
import unittest
from stitcher import stitch_geolocated_images

class TestMappingFixes(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_images"
        os.makedirs(self.test_dir, exist_ok=True)
        
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            
    def test_stitching(self):
        # Create two overlapping images
        dataset = []
        coords = []
        
        # Image 1
        img1 = np.zeros((200, 200, 3), dtype=np.uint8)
        # Add texture
        for i in range(0, 200, 20):
            cv2.line(img1, (i, 0), (i, 200), (255, 255, 255), 2)
            cv2.line(img1, (0, i), (200, i), (255, 255, 255), 2)
        cv2.rectangle(img1, (50, 50), (150, 150), (200, 0, 0), -1)
            
        # Image 2 (shifted version of Image 1)
        M = np.float32([[1, 0, 50], [0, 1, 0]]) # Shift x by 50
        img2 = cv2.warpAffine(img1, M, (200, 200))
        
        dataset.append(img1)
        coords.append((42.0, -76.0))
        
        dataset.append(img2)
        # Small GPS shift
        coords.append((42.0, -76.0001))
        
        try:
            results = stitch_geolocated_images(dataset, coords)
            print(f"Stitched {len(results)} groups")
            # Should have at least one result
            self.assertTrue(len(results) > 0)
            
            # Check for result validity
            for res in results:
                self.assertIsNotNone(res.image)
                self.assertGreater(res.image.shape[0], 0)
                self.assertGreater(res.image.shape[1], 0)
                
        except Exception as e:
            self.fail(f"Stitching failed with exception: {e}")

if __name__ == '__main__':
    unittest.main()
