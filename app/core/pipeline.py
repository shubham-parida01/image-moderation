from typing import Dict, Any, List

import numpy as np
from PIL import Image

from app.models import NSFWClassifier, ViolationDetector, ViolationSegmenter
from app.core.image_utils import blur_with_mask


class ModerationPipeline:
    """
    1. Classify - possible violation?
    2. Detect - violating boxes
    3. Segment - precise masks
    4. Blur - only masked regions
    """

    def __init__(self, output_dir: str = "safe_images"):
        self.classifier = NSFWClassifier()
        self.detector = ViolationDetector()
        
        self.segmenter = ViolationSegmenter()

    def run(self, image: Image.Image) -> Dict[str, Any]:
        import logging
        logger = logging.getLogger("image-moderation")
        
        cls_result = self.classifier.predict(image)
        possible_violation = cls_result.get("possible_violation", False)
        logger.info(f"[Pipeline] Classifier result: possible_violation={possible_violation}")

        if not possible_violation:
            return {
                "status": "SAFE",
                "image": image,
                "violations": []
            }

        detections = self.detector.predict(image)
        logger.info(f"[Pipeline] Detector found {len(detections)} total detections")
        violating_boxes = self._filter_violations(detections)
        logger.info(f"[Pipeline] Filtered to {len(violating_boxes)} violating boxes")

        if not violating_boxes:
            logger.warning("[Pipeline] No violating boxes found - returning unblurred image")
            return {
                "status": "SAFE_AFTER_DETECTION",
                "image": image,
                "violations": []
            }

        height, width = image.size[1], image.size[0]
        full_mask = np.zeros((height, width), dtype=np.uint8)
        labels: List[str] = []

        for det in violating_boxes:
            labels.append(det["label"])
            bbox = det["bbox"]  # [x1, y1, x2, y2]
            logger.info(f"[Pipeline] Processing violation: {det['label']} at {bbox}")
            mask = self.segmenter.segment(image, bbox)
            full_mask = np.maximum(full_mask, mask)

        mask_sum = np.sum(full_mask)
        logger.info(f"[Pipeline] Total mask pixels: {mask_sum} (shape: {full_mask.shape})")

        if not np.any(full_mask):
            logger.warning("[Pipeline] Mask is empty - returning unblurred image")
            return {
                "status": "SAFE_NO_MASK",
                "image": image,
                "violations": []
            }

        logger.info(f"[Pipeline] Applying blur to {mask_sum} pixels")
        blurred_image = blur_with_mask(image, full_mask)
        logger.info(f"[Pipeline] Blur applied successfully")

        return {
            "status": "VIOLATION_BLURRED",
            "image": blurred_image,
            "violations": sorted(set(labels)),
        }

    @staticmethod
    def _filter_violations(detections):
        violating_labels = {
            "cigarette",
            "vape",
            "joint",
            "gun",
            "knife",
            "alcohol_bottle",
            "nudity_explicit",
        }
        threshold = 0.3

        if not detections:
            return []

        filtered = [
            d for d in detections
            if isinstance(d, dict)
            and d.get("label") in violating_labels
            and float(d.get("score", 0)) >= threshold
        ]
        return filtered



