import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple
from skimage.feature import graycomatrix, graycoprops


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



def mean_pixels(image: np.ndarray) -> float:
    try:
        # TODO: Compute mean over non-zero parts only (maybe after otsu thresholding)
        mean = np.mean(image.flatten()).__float__()
        return mean
    except Exception as e:
        raise RuntimeError(f"An error has occured while computing the mean pixel value. Reason: {e}")


def avg_mean_images(list_imgs: List[np.ndarray]) -> np.ndarray:
    try:
        means = []
        for img in list_imgs:
            mean = mean_pixels(img)
            means.append(mean)

        means_np = np.array(means)
        return means_np
    except Exception as e:
        raise RuntimeError(f"Failed to compute means across all images. Reason: {e}")


def avg_mean_std_images(list_imgs: List[np.ndarray]) -> Tuple[float, float]:
    try:
        means_np = avg_mean_images(list_imgs)
        return np.mean(means_np).__float__(), np.std(means_np).__float__()
    except Exception as e:
        raise RuntimeError(f"Failed to get the average of means. Reason: {e}")


def outlier_threshold(avg_mean: float, std: float) -> Tuple[float, float]:
    try:
        low_thresh = avg_mean - (3 * std)
        high_thresh = avg_mean + (3 * std)

        return low_thresh, high_thresh
    except Exception as e:
        raise RuntimeError(f"Failed to get outlier low and high thresholds. Reason: {e}")


def grayscale_images(list_imgs: List[np.ndarray]) -> List[np.ndarray]:
    try:
        list_imgs_grayed = []

        for img in list_imgs:
            list_imgs_grayed.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

        return list_imgs_grayed
    except Exception as e:
        raise RuntimeError(f"Failed to grayscale images. Reason: {e}")


def is_grayscaled(img: np.ndarray) -> bool:
    try:
        if len(img.shape) < 3:
            return True
        elif img.shape[2] == 1:
            return True
        else:
            return False
    except Exception as e:
        raise RuntimeError(f"Failed to check if image is grayscaled. Reason: {e}")


def haralick_features(grayed_img_list: List[np.ndarray], properties: List[str]) -> pd.DataFrame:
    try:
        haralick_features = []

        for index, img in enumerate(grayed_img_list):
            if is_grayscaled(img):
                img_uint8 = img.astype(np.uint8)

                glcm = graycomatrix(
                    img_uint8,
                    distances=[1],
                    angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                    levels=256,
                    symmetric=True,
                    normed=True
                )

                haralick_row: Dict[str, Any] = {'image_index': index}

                for prop in properties:
                    haralick_row[prop] = float(np.mean(graycoprops(glcm, prop)))

                haralick_features.append(haralick_row)
            else:
                raise RuntimeError(f"Failed to compute haralick features. Not gray scaled")

        df_haralick_features = pd.DataFrame(haralick_features)

        return df_haralick_features
    except Exception as e:
        raise RuntimeError(f"Failed to compute haralick features. Reason: {e}.")


# Apply Otsu’s thresholding to segment foreground from background.
def otsu_thresholding(list_imgs: List[np.ndarray]) -> List[np.ndarray]:
    try:
        # _, binary_color = cv2.threshold(gray_color, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        list_otsus = []

        for img in list_imgs:
            if is_grayscaled(img):
                img_uint8 = img.astype(np.uint8)
                _, binary_color = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                kernel = np.ones((5, 5), np.uint8)
                closed_binary = cv2.morphologyEx(binary_color, cv2.MORPH_CLOSE, kernel)
                list_otsus.append(closed_binary)
            else:
                raise RuntimeError(f"Failed to apply otsu thresholding on gray scaled images. Not gray scaled")

        return list_otsus
    except Exception as e:
        raise RuntimeError(f"Failed to apply otsu thresholding on gray scaled images. Reason: {e}")


# Perform the Histogram Equalization technique with CLAHE on Grayscale Image.
def clahe_grayscale():
    pass

# Perform Convolution filters on images (Gaussian Blur, Median Blur, Sharpening, Edge Detection)

def summary_statistics():
    pass


if __name__ == '__main__':
    print("Loading images from local directory...")

    list_imgs = load_images_local(TUMOR_TRAIN_PATH)
    print(f"Number of images: {len(list_imgs)}")

    print("Taking mean pixel value of first image")
    mean, std = avg_mean_std_images(list_imgs)
    print(f"The avergae mean across images is: {mean:.2f}")
    print(f"The standard deviation of means across images is: {std:.2f}")
