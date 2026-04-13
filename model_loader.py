from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)
model = None
model_loaded = False

def get_model():
    """Get cached YOLO model. Should only be called after startup."""
    global model
    if model is None:
        logger.warning("Model not preloaded! Loading on demand...")
        model = YOLO("best.pt")
    return model

def load_model_startup():
    """Load model during application startup to avoid request delays."""
    global model, model_loaded
    try:
        logger.info("Loading YOLO model on startup...")
        model = YOLO("best.pt")
        model_loaded = True
        logger.info("YOLO model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        model_loaded = False
        raise

def is_model_ready():
    """Check if model is ready to use."""
    return model_loaded and model is not None