import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from EDA import TUMOR_TRAIN_PATH
else:
    from . import TUMOR_TRAIN_PATH


def load_images_local(directory: Path) -> List[np.ndarray]:
    try:
        img_list = []

        if directory.exists():
            list_files = os.listdir(directory)
            for file_name in list_files:
                img = cv2.imread(str(directory / file_name))
                img_list.append(img)

            return img_list

        return img_list
    except Exception as e:
        raise RuntimeError(f"Failed to load images from the local directory {directory}. Reason: {e}")


# Search for outliers -> Distribution of Mean Pixel Values

# Average mean pixel value AND Std deviation of mean pixel values
def mean_pixels(image: np.ndarray) -> float:
    try:
        mean = np.mean(image.flatten()).__float__()

        return mean
    except Exception as e:
        raise RuntimeError(f"An error has occured while computing the mean pixel value. Reason: {e}")


def avg_mean_images(list_imgs: List[np.ndarray]) -> float:
    try:
        means = []
        for i, img in enumerate(list_imgs):
            mean = mean_pixels(img)
            means.insert(i, mean)

        means_np = np.array(means)
        return np.mean(means_np).__float__()
    except Exception as e:
        raise RuntimeError(f"Failed to get the average of means. Reason: {e}")

# Thresholds for outliers (e.g., >3 std away from mean)

# low_thresh = mean_of_means - 3*std_of_means
# high_thresh = mean_of_means + 3*std_of_means
# outlier_low = np.where(all_means < low_thresh)[0]
# outlier_high = np.where(all_means > high_thresh)[0]



# **Compute Haralick features** (e.g., contrast, correlation, energy, homogeneity) for each texture:
# - Use `skimage.feature.graycomatrix` and `graycoprops`.
#   - Compare values in a Pandas DataFrame.

# Apply Otsu’s thresholding to segment foreground from background.

# Perform the Histogram Equalization technique with CLAHE on Grayscale Image.

# Perform Convolution filters on images (Gaussian Blur, Median Blur, Sharpening, Edge Detection)

# Hough Tranform

if __name__ == '__main__':
    print("Loading images from local directory...")

    list_imgs = load_images_local(TUMOR_TRAIN_PATH)
    print(f"Number of images: {len(list_imgs)}")

    print("Taking mean pixel value of first image")
    mean, std = mean_pixel_value(list_imgs[0])
    print(f"The mean pixel value is: {mean:.2f}")
