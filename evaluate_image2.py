import cv2
import numpy as np
import scipy.ndimage as ndimage

def calculate_metrics(image_path):
    # Read color image
    img = cv2.imread(image_path)
    if img is None:
        return 0

    # Calculate symmetry on grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    mid = w // 2
    left_half = gray[:, :mid]

    if w % 2 == 0:
        right_half = gray[:, mid:]
    else:
        right_half = gray[:, mid+1:]

    right_half_mirrored = cv2.flip(right_half, 1)

    diff = cv2.absdiff(left_half, right_half_mirrored)
    _, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

    total_pixels = left_half.size
    different_pixels = np.count_nonzero(thresh)
    symmetrical_pixels = total_pixels - different_pixels
    symmetry_score = symmetrical_pixels / total_pixels

    # Calculate Center of Mass
    # Find bounding box of content (ignore background)
    # Assume top-left pixel is background color
    bg_color = img[0, 0]
    # Create mask of non-background pixels
    mask = np.any(img != bg_color, axis=-1)

    if np.sum(mask) == 0:
        cm_x, cm_y = 0.5, 0.5
    else:
        y_coords, x_coords = np.nonzero(mask)
        cm_y = np.mean(y_coords) / h
        cm_x = np.mean(x_coords) / w

    return {
        "symmetry": symmetry_score,
        "center_of_mass": (cm_x, cm_y),
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(calculate_metrics(sys.argv[1]))
