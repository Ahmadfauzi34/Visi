import cv2
import numpy as np

def calculate_symmetry(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0

    h, w = img.shape
    mid = w // 2
    left_half = img[:, :mid]

    if w % 2 == 0:
        right_half = img[:, mid:]
    else:
        right_half = img[:, mid+1:]

    right_half_mirrored = cv2.flip(right_half, 1)

    # Calculate difference
    diff = cv2.absdiff(left_half, right_half_mirrored)
    # Threshold to find significant differences
    _, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

    # Calculate percentage of symmetrical pixels
    total_pixels = left_half.size
    different_pixels = np.count_nonzero(thresh)
    symmetrical_pixels = total_pixels - different_pixels
    symmetry_score = symmetrical_pixels / total_pixels
    return symmetry_score

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(calculate_symmetry(sys.argv[1]))
