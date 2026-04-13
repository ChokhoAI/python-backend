from ultralytics import YOLO

model = None

def get_model():
    global model
    if model is None:
        model = YOLO("best.pt")
    return model