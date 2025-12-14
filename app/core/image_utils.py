from PIL import Image
import numpy as np
import cv2


def blur_with_mask(pil_image: Image.Image, mask: np.ndarray) -> Image.Image:
    import logging
    logger = logging.getLogger("image-moderation")
    
    img = np.array(pil_image) 

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)    # if grayscale, make 3-channel

    h, w = img.shape[:2]
    if mask.shape != (h, w):
        logger.error(f"Mask shape mismatch: mask {mask.shape} != image {(h, w)}")
        raise ValueError(f"Mask shape {mask.shape} != image shape {(h, w)}")

    # Ensure mask is binary (0 or 1)
    mask_binary = (mask > 0).astype(np.uint8)
    mask_pixels = np.sum(mask_binary)
    logger.info(f"[Blur] Mask has {mask_pixels} pixels to blur (out of {h*w} total)")

    # Apply Gaussian blur to entire image
    blurred = cv2.GaussianBlur(img, (31, 31), 0)

    # Create 3-channel mask for RGB
    mask_3ch = np.stack([mask_binary] * 3, axis=-1).astype(bool)
    
    # Copy original and replace masked regions with blurred version
    out = img.copy()
    out[mask_3ch] = blurred[mask_3ch]
    
    # Verify blur was applied
    changed_pixels = np.sum(out[mask_3ch] != img[mask_3ch])
    logger.info(f"[Blur] Changed {changed_pixels} pixels in masked region")

    return Image.fromarray(out)