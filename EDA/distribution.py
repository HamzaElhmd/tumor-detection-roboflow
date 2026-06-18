import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from skimage.feature import graycomatrix, graycoprops

import torch
from torch.utils.data import DataLoader

# --- ImageNet normalization constants (matching preprocess.py) ---
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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


def tensor_batch_to_numpy(
    images_tensor: torch.Tensor,
    denormalize: bool = False
) -> List[np.ndarray]:
    """Convert a (B, 3, H, W) preprocessed tensor batch to a list of numpy (H, W, C) arrays.

    If denormalize=True, reverse ImageNet normalization so pixel values are in [0, 1]
    (suitable for cv2-based EDA functions that expect typical image ranges).
    """
    imgs = []
    for i in range(images_tensor.size(0)):
        img = images_tensor[i].cpu().numpy()           # (3, H, W)
        img = img.transpose(1, 2, 0)                   # (H, W, 3)
        if denormalize:
            img = img * IMAGENET_STD + IMAGENET_MEAN    # back to [0, 1] range
            img = np.clip(img, 0.0, 1.0)
        imgs.append(img)
    return imgs


def load_images_from_dataloader(
    dataloader: DataLoader,
    max_batches: Optional[int] = None,
    denormalize: bool = False
) -> List[np.ndarray]:
    """Extract all images from a DataLoader into a flat list of numpy (H, W, C) arrays.

    This bridges the preprocessed tensor pipeline with existing cv2-based EDA functions.
    Set denormalize=True to reverse ImageNet normalization for functions that expect
    standard pixel ranges (0-1 or 0-255).

    Parameters
    ----------
    dataloader : DataLoader
        PyTorch DataLoader yielding (images, labels) batches.
    max_batches : int, optional
        Cap the number of batches processed (useful for large datasets).
    denormalize : bool
        If True, reverse ImageNet normalization.
    """
    all_images = []
    for batch_idx, (images, _) in enumerate(dataloader):
        all_images.extend(tensor_batch_to_numpy(images, denormalize=denormalize))
        if max_batches is not None and batch_idx + 1 >= max_batches:
            break
    return all_images


def preprocessed_pixel_statistics(
    dataloader: DataLoader,
    max_batches: Optional[int] = None
) -> Dict[str, Any]:
    """Compute per-channel pixel statistics on preprocessed (normalized) data.

    Returns min, max, mean, and std per channel. Useful for verifying that
    normalization produces the expected distributions.
    """
    all_images = []
    for batch_idx, (images, _) in enumerate(dataloader):
        all_images.append(images)
        if max_batches is not None and batch_idx + 1 >= max_batches:
            break

    stacked = torch.cat(all_images, dim=0).cpu()  # (N, 3, H, W)
    n = stacked.size(0)

    return {
        'num_images': n,
        'shape': tuple(stacked.shape),
        'min_per_channel': tuple(stacked.amin(dim=(0, 2, 3)).tolist()),
        'max_per_channel': tuple(stacked.amax(dim=(0, 2, 3)).tolist()),
        'mean_per_channel': tuple(stacked.mean(dim=(0, 2, 3)).tolist()),
        'std_per_channel': tuple(stacked.std(dim=(0, 2, 3)).tolist()),
    }


def preprocessed_channel_distributions(
    dataloader: DataLoader,
    max_batches: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Compute per-channel histograms of preprocessed (normalized) pixel values.

    Returns histograms (counts) and bin edges for each of the 3 channels.
    Bin range is set wide enough to capture normalized values (~ -3 to +3).
    """
    all_images = []
    for batch_idx, (images, _) in enumerate(dataloader):
        all_images.append(images)
        if max_batches is not None and batch_idx + 1 >= max_batches:
            break

    stacked = torch.cat(all_images, dim=0).cpu()  # (N, 3, H, W)
    bins = np.linspace(-4, 4, 129)  # 128 bins from -4 to +4
    hists = {}
    channel_names = ['R', 'G', 'B']

    for c in range(3):
        channel_data = stacked[:, c, :, :].flatten().numpy()
        hist, _ = np.histogram(channel_data, bins=bins)
        hists[channel_names[c]] = hist.astype(np.float64)

    hists['bins'] = bins
    return hists


def detect_corrupted_images(directory: Path) -> List[str]:
    try:
        corrupted: List[str] = []
        if directory.exists():
            for file_name in os.listdir(directory):
                img = cv2.imread(str(directory / file_name))
                if img is None:
                    corrupted.append(file_name)
        return corrupted
    except Exception as e:
        raise RuntimeError(f"Failed to detect corrupted images in {directory}. Reason: {e}")


def blur_score_laplacian(img_gray: np.ndarray) -> float:
    try:
        if not is_grayscaled(img_gray):
            raise RuntimeError("Image must be grayscale for Laplacian blur detection.")
        variance = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        return float(variance)
    except Exception as e:
        raise RuntimeError(f"Failed to compute Laplacian blur score. Reason: {e}")


def image_entropy(img_gray: np.ndarray, bins: int = 256) -> float:
    try:
        if not is_grayscaled(img_gray):
            raise RuntimeError("Image must be grayscale for entropy computation.")
        hist, _ = np.histogram(img_gray.flatten(), bins=bins, range=(0, bins))
        hist = hist.astype(np.float64)
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist)))
    except Exception as e:
        raise RuntimeError(f"Failed to compute image entropy. Reason: {e}")


def check_data_leakage(*directories: Path) -> Set[str]:
    try:
        filename_sets: List[Set[str]] = []
        for d in directories:
            fnames = set(os.listdir(d)) if d.exists() else set()
            filename_sets.append(fnames)

        leaked: Set[str] = set()
        for i in range(len(filename_sets)):
            for j in range(i + 1, len(filename_sets)):
                leaked |= filename_sets[i] & filename_sets[j]
        return leaked
    except Exception as e:
        raise RuntimeError(f"Failed to check data leakage. Reason: {e}")


def per_channel_histogram(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        if len(img.shape) < 3 or img.shape[2] < 3:
            raise RuntimeError("Image must be 3-channel BGR for per-channel histogram.")
        hist_b = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        hist_g = cv2.calcHist([img], [1], None, [256], [0, 256]).flatten()
        hist_r = cv2.calcHist([img], [2], None, [256], [0, 256]).flatten()
        bins = np.arange(257)
        return hist_b, hist_g, hist_r, bins
    except Exception as e:
        raise RuntimeError(f"Failed to compute per-channel histogram. Reason: {e}")



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


def pixelval_frequency_histogram_cdf(img_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        if is_grayscaled(img_gray):
            hist, bins = np.histogram(img_gray.flatten(), 256, (0, 256))
            return hist, bins, hist.cumsum()
        else:
            raise RuntimeError(f"Failed to create histogram if pixel values vs frequency. Not gray scaled")
    except Exception as e:
        raise RuntimeError(f"Failed to create histogram of pixel values vs frequency. Reason: {e}")


def he_grayscale(img_gray: np.ndarray):
    try:
        if is_grayscaled(img_gray):
            return cv2.equalizeHist(img_gray)
        else:
            raise RuntimeError(f"Failed to normally equalize histogram. Not gray scaled.")
    except Exception as e:
        raise RuntimeError(f"Failed to normally equalize histogram. Reason: {e}")


def clahe_grayscale(img_gray: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    try:
        if is_grayscaled(img_gray):
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            return clahe.apply(img_gray)
        else:
            raise RuntimeError(f"Failed to create CLAHE from grayscale images. Not gray scaled.")
    except Exception as e:
        raise RuntimeError(f"Failed to create CLAHE from grayscale images. Reason: {e}")

def gaussian_blur(img: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), sigma: float = 1.0) -> np.ndarray:
    try:
        return cv2.GaussianBlur(img, kernel_size, sigma)
    except Exception as e:
        raise RuntimeError(f"Failed to apply Gaussian blur. Reason: {e}")


def median_blur(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    try:
        return cv2.medianBlur(img, kernel_size)
    except Exception as e:
        raise RuntimeError(f"Failed to apply median blur. Reason: {e}")


def sharpen(img: np.ndarray) -> np.ndarray:
    try:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)
    except Exception as e:
        raise RuntimeError(f"Failed to apply sharpening filter. Reason: {e}")


def edge_detection_canny(img: np.ndarray, low_threshold: float = 50, high_threshold: float = 150) -> np.ndarray:
    try:
        return cv2.Canny(img, low_threshold, high_threshold)
    except Exception as e:
        raise RuntimeError(f"Failed to apply Canny edge detection. Reason: {e}")


def edge_detection_sobel(img_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        if not is_grayscaled(img_gray):
            raise RuntimeError(f"Image is not gray scaled.")
        sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_x_abs = cv2.convertScaleAbs(sobel_x)
        sobel_y_abs = cv2.convertScaleAbs(sobel_y)
        return sobel_x_abs, sobel_y_abs
    except Exception as e:
        raise RuntimeError(f"Failed to apply Sobel edge detection. Reason: {e}")


def image_dimensions(img: np.ndarray) -> Tuple[int, int, Optional[int]]:
    try:
        if len(img.shape) == 2:
            return img.shape[0], img.shape[1], None
        return img.shape[0], img.shape[1], img.shape[2]
    except Exception as e:
        raise RuntimeError(f"Failed to get image dimensions. Reason: {e}")


def class_distribution(directory: Path) -> Dict[str, int]:
    try:
        result: Dict[str, int] = {}
        for class_dir in [d for d in directory.iterdir() if d.is_dir()]:
            result[class_dir.name] = len(os.listdir(class_dir))
        return result
    except Exception as e:
        raise RuntimeError(f"Failed to compute class distribution. Reason: {e}")


def summary_statistics(list_imgs: List[np.ndarray]) -> pd.DataFrame:
    try:
        stats = []
        for idx, img in enumerate(list_imgs):
            h, w, c = image_dimensions(img)
            stats.append({
                'image_index': idx,
                'height': h,
                'width': w,
                'channels': c if c else 1,
                'mean': np.mean(img),
                'std': np.std(img),
                'min': np.min(img),
                'max': np.max(img),
            })
        return pd.DataFrame(stats)
    except Exception as e:
        raise RuntimeError(f"Failed to compute summary statistics. Reason: {e}")


if __name__ == '__main__':
    print("Loading images from local directory...")

    list_imgs = load_images_local(TUMOR_TRAIN_PATH)
    print(f"Number of images: {len(list_imgs)}")

    print("Taking mean pixel value of first image")
    mean, std = avg_mean_std_images(list_imgs)
    print(f"The avergae mean across images is: {mean:.2f}")
    print(f"The standard deviation of means across images is: {std:.2f}")
