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

    def __init__(self):
        self.classifier = NSFWClassifier()
        self.detector = ViolationDetector()
        
        self.segmenter = ViolationSegmenter()

    def run(self, image: Image.Image) -> Dict[str, Any]:
        cls_result = self.classifier.predict(image)
        possible_violation = cls_result.get("possible_violation", False)

        if not possible_violation:
            return {
                "status": "SAFE",
                "image": image,
                "violations": []
            }

        detections = self.detector.predict(image)
        violating_boxes = self._filter_violations(detections)

        if not violating_boxes:
            return {
                "status": "SAFE_AFTER_DETECTION",
                "image": image,
                "violations": []
            }

        # PIL Image.size returns (width, height)
        # For numpy arrays, we use (height, width) convention
        width, height = image.size  # width = image.size[0], height = image.size[1]
        full_mask = np.zeros((height, width), dtype=np.uint8)
        labels: List[str] = []

        for det in violating_boxes:
            labels.append(det["label"])
            bbox = det["bbox"]  # [x1, y1, x2, y2]
            mask = self.segmenter.segment(image, bbox)
            # segmenter returns mask with shape (height, width) - same as full_mask
            full_mask = np.maximum(full_mask, mask)

        if not np.any(full_mask):
            return {
                "status": "SAFE_NO_MASK",
                "image": image,
                "violations": []
            }

        # image is PIL.Image.Image, full_mask is np.ndarray with shape (height, width)
        blurred_image = blur_with_mask(image, full_mask)

        return {
            "status": "VIOLATION_BLURRED",
            "image": blurred_image,  # This is a PIL.Image.Image object
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



