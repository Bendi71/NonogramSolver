import numpy as np
from PIL import Image


def image_to_nonogram(image_path, size: tuple, threshold: int):
    """
    Convert an image to a Nonogram grid.

    :param image_path: Path to the image file.
    :param size: Tuple (width, height) representing the size of the Nonogram grid.
    :param threshold: Threshold value for converting the image to binary.
    :return: 2D numpy array representing the Nonogram grid.
    """
    # Load image and convert to grayscale
    img = Image.open(image_path).convert('L')

    # Convert image to binary (black and white) using the given threshold
    binary_img = img.point(lambda p: p > threshold and 255)

    # Cut to square shape
    width, height = binary_img.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = (width + min_dim) // 2
    bottom = (height + min_dim) // 2
    binary_img = binary_img.crop((left, top, right, bottom))

    # Resize to a smaller image
    binary_img = binary_img.resize(size)

    # Convert the binary image to a numpy array
    img_array = np.array(binary_img)

    # Convert to binary: 1 for filled (black), 0 for empty (white)
    grid = (img_array == 0).astype(int)

    return grid


import numpy as np
from PIL import Image


def image_to_nonogram(image_path, size: tuple, threshold: int):
    # Load image and convert to grayscale
    img = Image.open(image_path).convert('L')

    # Convert image to binary (black and white) using the given threshold
    binary_img = img.point(lambda p: p > threshold and 255)

    # Cut to square shape
    width, height = binary_img.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = (width + min_dim) // 2
    bottom = (height + min_dim) // 2
    binary_img = binary_img.crop((left, top, right, bottom))

    # resize to a smaller image
    binary_img = binary_img.resize(size)

    # Convert the binary image to a numpy array
    img_array = np.array(binary_img)

    # Convert to binary: 1 for filled (black), 0 for empty (white)
    grid = (img_array == 0).astype(int)

    return grid
