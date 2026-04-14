from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import logging
import gc

logger = logging.getLogger(__name__)

model = YOLO("best.pt")

async def detect_trash(image_bytes : bytes):
    try:
        logger.info(f"Processing image of size: {len(image_bytes)} bytes")
        
        image = Image.open(BytesIO(image_bytes))
        image = image.resize((320, 320))

        results = model(image, verbose=False, conf=0.25)

        trash_class_id = 0

        for result in results:
            mask = result.boxes.cls == trash_class_id

            confidences = result.boxes.conf[mask]

            if any(conf > 0.25 for conf in confidences):
                logger.info("Trash detected")
                return True

        logger.info("No trash detected")
        return False
    except Exception as e:
        logger.error(f"Error in detect_trash: {e}", exc_info=True)
        raise
    finally:
        gc.collect()