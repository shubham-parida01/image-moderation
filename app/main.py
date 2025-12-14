import os
import io
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from PIL import Image

from app.core import ModerationPipeline

try:
    from app.models.detector import get_detector
    _HAS_GET_DETECTOR = True
except Exception:
    get_detector = None
    _HAS_GET_DETECTOR = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("image-moderation")

app = FastAPI(
    title="Image Moderation + Segmentation API",
    version="0.1.0",
)

pipeline = ModerationPipeline(output_dir="safe_images")


def _try_load_detector_from_env():
    if not _HAS_GET_DETECTOR:
        logger.info("get_detector() not found in app.models.detector — skipping detector load.")
        return
    
    candidates = []
    env_path = os.environ.get("DETECTOR_WEIGHTS")
    if env_path:
        candidates.append(env_path)

    candidates.append(os.path.join("app", "models", "weights", "detector_best.pt"))

    candidates.append(os.path.join("ml", "weights", "detector_best.pt"))
    candidates.append(os.path.join("ml", "weights", "my_exported_best.pt"))

    for p in candidates:
        if not p:
            continue
        p = os.path.abspath(p)
        if os.path.exists(p):
            logger.info(f"[Detector] Found weights at: {p} — attempting to load.")
            try:
                det = get_detector(weights_path=p, device=None)  
                if getattr(det, "ready", False):
                    pipeline.detector = det
                    logger.info(f"[Detector] Loaded detector from: {p}")
                else:
                    pipeline.detector = det
                    logger.warning(f"[Detector] Detector instance created but not ready (check ultralytics install).")
                return
            except Exception as e:
                logger.exception(f"[Detector] Failed to load detector weights from {p}: {e}")
        else:
            logger.debug(f"[Detector] Weights not found at {p}")
    logger.info("[Detector] No detector weights loaded — using default pipeline detector (may be dummy).")


_try_load_detector_from_env()


@app.post("/moderate-image")
async def moderate_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    content = await file.read()

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
        logger.info(f"[API] Received image: {image.size[0]}x{image.size[1]}")
    except Exception as e:
        logger.error(f"[API] Failed to open image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        result = pipeline.run(image)
        logger.info(f"[API] Pipeline result status: {result['status']}")
        processed_image = result["image"]
    except Exception as e:
        logger.exception(f"[API] Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    processed_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return Response(
        content=img_byte_arr.getvalue(),
        media_type="image/png",
        headers={
            "X-Status": result["status"],
            "X-Violations": ",".join(result["violations"]) if result["violations"] else ""
        }
    )


@app.post("/test-blur")
async def test_blur(file: UploadFile = File(...)):
    """
    Test endpoint that forces blurring on the center of the image.
    Useful for testing if the blur function works correctly.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    content = await file.read()

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
        logger.info(f"[Test] Received image: {image.size[0]}x{image.size[1]}")
    except Exception as e:
        logger.error(f"[Test] Failed to open image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Create a test mask in the center of the image
    import numpy as np
    from app.core.image_utils import blur_with_mask
    
    w, h = image.size
    mask = np.zeros((h, w), dtype=np.uint8)
    # Blur center 30% of image
    x1, y1 = int(w * 0.35), int(h * 0.35)
    x2, y2 = int(w * 0.65), int(h * 0.65)
    mask[y1:y2, x1:x2] = 1
    
    logger.info(f"[Test] Creating test mask at ({x1},{y1}) to ({x2},{y2})")
    
    try:
        blurred_image = blur_with_mask(image, mask)
        logger.info("[Test] Blur applied successfully")
    except Exception as e:
        logger.exception(f"[Test] Blur error: {e}")
        raise HTTPException(status_code=500, detail=f"Blur error: {str(e)}")
    
    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    blurred_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return Response(
        content=img_byte_arr.getvalue(),
        media_type="image/png",
        headers={
            "X-Status": "TEST_BLURRED",
            "X-Test-Mask": f"{x1},{y1},{x2},{y2}"
        }
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "detector_ready": getattr(pipeline.detector, "ready", False),
        "detector_weights": getattr(pipeline.detector, "weights_path", None),
    }