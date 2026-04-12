from ultralytics import YOLO

model = None

def load_model():
    global model
    model = YOLO("best.pt")
    print("YOLO model loaded successfully")

def get_model():
    return model