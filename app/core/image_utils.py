from PIL import Image
import numpy as np
import cv2


def blur_with_mask(pil_image: Image.Image, mask: np.ndarray) -> Image.Image:
    """
    Blur regions of the image where mask > 0.
    
    Args:
        pil_image: PIL.Image.Image object
        mask: numpy array with shape (height, width), dtype np.uint8
              Values: 0 = no blur, >0 = blur applied
    
    Returns:
        PIL.Image.Image with blurred regions
    """
    # Convert PIL Image to numpy array
    img = np.array(pil_image) 

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)    # if grayscale, make 3-channel

    h, w = img.shape[:2]  # numpy array shape is (height, width)
    if mask.shape != (h, w):
        raise ValueError(f"Mask shape {mask.shape} != image shape {(h, w)}")

    # Apply Gaussian blur to entire image
    blurred = cv2.GaussianBlur(img, (31, 31), 0)

    # Create 3-channel mask: values > 0 indicate blur regions
    # mask > 0 creates boolean array where True = blur region
    mask_binary = (mask > 0).astype(bool)
    mask_3ch = np.stack([mask_binary] * 3, axis=-1)
    
    # Copy original and replace masked regions with blurred version
    out = img.copy()
    out[mask_3ch] = blurred[mask_3ch]

    # Convert back to PIL Image
    return Image.fromarray(out)