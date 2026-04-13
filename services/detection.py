from model_loader import get_model
from PIL import Image
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

async def detect_trash(image_bytes : bytes):
    """Detect trash in image. Model should be preloaded."""
    try:
        model = get_model()
        image = Image.open(BytesIO(image_bytes))
        image = image.resize((320, 320))

        results = model(image)

        trash_class_id = 0

        for result in results:
            mask = result.boxes.cls == trash_class_id

            confidences = result.boxes.conf[mask]

            if any(conf > 0.25 for conf in confidences):
                return True

        return False
    except Exception as e:
        logger.error(f"Error in detect_trash: {e}")
        raise