import cv2
import numpy as np


def compare_images(image1: np.ndarray, image2: np.ndarray) -> float:
    similarity_value = 0
    for i in range(3):
        target_hist = cv2.calcHist([image1], [i], None, [256], [0, 256])
        comparing_hist = cv2.calcHist([image2], [i], None, [256], [0, 256])
        similarity_value += cv2.compareHist(target_hist, comparing_hist, 0)

    return similarity_value / 3


if __name__ == '__main__':
    value = compare_images(cv2.imread('Histogram_Comparison_Source_0.jpg'),
                           cv2.imread('Histogram_Comparison_Source_1.jpg'))

    print(value)
