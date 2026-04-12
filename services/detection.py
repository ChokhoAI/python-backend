from ultralytics import YOLO
from PIL import Image
from io import BytesIO

model = YOLO("best.pt")

async def detect_trash(image_bytes : bytes):
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