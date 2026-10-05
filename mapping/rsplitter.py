"""Partitions an image into random rectangles 

Args:
    1st pos (int): number of partitions
    2*nd pos (filenames): one or more files to split

Example Usage:

        $python rsplitter.py 10 logo.jpg

    will partition logo.png into 10 images and save them in logo.jpg-split/
    A reference image is also included in the same folder.
    A CSV file with the center pixel locations and rotation direction of each split image is also created.

TODO cmdline options for rotation and overlapping partitions
"""

import os
import sys
import random
import csv
from PIL import Image, ImageDraw, UnidentifiedImageError

USAGE_MSG = "Usage: {} <number of partitions> <image filename(s)>"
SPL_BND = .3  # Must be [0, 1)


class Rect:
    """A rectangle with a coordinate.

    The coordinate system used is the same as the PIL library with 0,0 in the
    upper left corner.

    Attributes:
      left (int): the rectangle's left bound
      upper (int): the rectangle's upper bound
      right (int): the rectangle's right bound
      lower (int): the rectangle's lower bound
    """

    def __init__(self, left, upper, right, lower):
        self.left = left
        self.upper = upper
        self.right = right
        self.lower = lower

    def upperleft(self):
        """Upper left corner of self

        Returns:
            (left, upper)
        """
        return (self.left, self.upper)

    def to_tuple(self):
        """Tuple representation of self

        Returns:
            (left, upper, right, lower)
        """
        return (self.left, self.upper, self.right, self.lower)

    def to_list(self):
        """List representation of self

        Returns:
            [left, upper, right, lower]
        """
        return [self.left, self.upper, self.right, self.lower]

    def with_left_as(self, left):
        """A copy of self with a new left bound

        Args:
            left (int): new left bound

        Returns:
            Rect: the new rectangle
        """
        return Rect(left, self.upper, self.right, self.lower)

    def with_upper_as(self, upper):
        """A copy of self with a new upper bound

        Args:
            upper (int): new upper bound

        Returns:
            Rect: the new rectangle
        """
        return Rect(self.left, upper, self.right, self.lower)

    def with_right_as(self, right):
        """A copy of self with a new right bound

        Args:
            right (int): new right bound

        Returns:
            Rect: the new rectangle
        """
        return Rect(self.left, self.upper, right, self.lower)

    def with_lower_as(self, lower):
        """A copy of self with a new lower bound

        Args:
            lower (int): new lower bound

        Returns:
            Rect: the new rectangle
        """
        return Rect(self.left, self.upper, self.right, lower)

    def center(self):
        """Calculate the center of the rectangle

        Returns:
            tuple: (center_x, center_y)
        """
        center_x = (self.left + self.right) // 2
        center_y = (self.upper + self.lower) // 2
        return (center_x, center_y)


def rsplit(r: Rect):
    """Randomly splits a Rect into two Rect

    The resulting two Rect will cover the same area as original Rect. The split
    occurs randomly either horizontally or vertically. The split happens guided
    by an axis in between the original Rect bounds, with proportional padding
    defined by the constant SPL_BND. Additional noise is added so that the
    resulting two images overlap slightly across the split axis and within the
    proportional padding.

    Args:
        r: rectangle to split

    Returns:
        tuple: the pair of rectangles resulting from the split
    """
    if bool(random.getrandbits(1)):
        pad = int((r.lower - r.upper) * SPL_BND)
        ubnd = (r.upper + pad)
        lbnd = r.lower - pad
        divider = random.randint(ubnd, lbnd)
        lbnd_noise = random.randint(divider, lbnd)
        ubnd_noise = random.randint(ubnd, divider)
        # use divider instead for no noise
        return r.with_lower_as(lbnd_noise), r.with_upper_as(ubnd_noise)
    else:
        pad = int((r.right - r.left) * SPL_BND)
        lbnd = r.left + pad
        rbnd = r.right - pad
        divider = random.randint(lbnd, rbnd)
        rbnd_noise = random.randint(divider, rbnd)
        lbnd_noise = random.randint(lbnd, divider)
        # use divider instead for no noise
        return r.with_right_as(rbnd_noise), r.with_left_as(lbnd_noise)


def partition_rect(n: int, r: Rect, acc: list):
    """Partitions rectangle into random fragments with some overlap

    Args:
        n: number of fragments to produce. Must be >= 1
        r: rectangle to partition
        acc: list to which to append the rectangle fragments

    Effect:
        the Rect objects representing the rectangle partitions are appended to acc
    """
    # print("{}, {}".format(n, r.to_tuple()))
    if n == 1:
        acc.append(r)
        return

    r1, r2 = rsplit(r)

    m = int(n/2)
    partition_rect(n-m, r1, acc)
    partition_rect(m, r2, acc)


def rotate(img: Image):
    """Randomly rotates an Image

    Args:
        img: image to rotate

    Returns:
        tuple: (rotated Image, rotation direction)
    """
    angles = [0, 90, 180, 270]
    rotation_angle = random.choice(angles)
    rotated_img = img.rotate(rotation_angle, expand=True)
    
    return rotated_img, rotation_angle


def partition_image(n: int, img: Image):
    """Partitions image into random fragments

    Subimages are not perfect partitions and have some overlap. Subimages are
    randomly rotated and split.

    Args:
      n: number of fragemnts to produce. Must be >= 1
      img: image to partition

    Effect: creates images in directory '{image-name}-split/'
            creates a CSV file with center pixel locations and rotation direction of each split image
    """
    outdir = "{}-split".format(img.filename)
    def outpath(outfile): return os.path.join(outdir, outfile)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    else:
        for f in os.listdir(outdir):
            os.remove(outpath(f))
    partitions = []
    partition_rect(n, Rect(0, 0, img.width, img.height), partitions)
    ref_skel = Image.new("RGBA", img.size, color=(0, 0, 0, 0))
    d = ImageDraw.Draw(ref_skel)
    
    # Create CSV file for center pixel locations and rotation direction
    csv_filename = outpath("geolocations.csv")
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Image', 'Center_X', 'Center_Y', 'Degrees_CC_from_North'])
        
        for i, rect in enumerate(partitions, start=1):
            outfile = "{}.{}".format(str(i), img.format)
            cropped_img = img.crop(rect.to_tuple())
            rotated_img, rotation_angle = rotate(cropped_img)
            rotated_img.save(outpath(outfile), format=img.format)
            d.rectangle(rect.to_list())
            d.line(rect.to_list())
            d.text(rect.upperleft(), str(i))
            
            # Calculate and write center pixel location and rotation direction
            center_x, center_y = rect.center()
            csvwriter.writerow([outfile, center_x, center_y, rotation_angle])
    
    img.putalpha(128)
    ref_img = Image.alpha_composite(ref_skel, img)
    ref_img.save(outpath("reference.png"))


if __name__ == "__main__":
    # Check arg lens
    if len(sys.argv) < 3:
        print(USAGE_MSG)
        print("Error: missing arguments")
        sys.exit(0)

    # Parse n
    n = 0
    try:
        n = int(sys.argv[1])
    except ValueError:
        print(USAGE_MSG)
        print("Error: number of partitions not an integer")
        sys.exit(0)
    if n < 1:
        print("Error: number of partitions must be at least 1")
        sys.exit(0)

    # Partition images, handling file errors
    for infile in sys.argv[2:]:
        try:
            with Image.open(infile) as img:
                partition_image(n, img)
        except FileNotFoundError:
            print("Error: {} not found".format(infile))
        except UnidentifiedImageError:
            print("Error: {} could not be opened".format(infile))
