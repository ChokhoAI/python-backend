from ultralytics import YOLO

model = YOLO("best.pt")

model.predict(source= "images", conf = 0.25, save = True)