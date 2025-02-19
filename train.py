from ultralytics import YOLO

def train_model():
    model = YOLO("yolov8m.pt")
    results = model.train(data='dataset/data.yaml', epochs=100, batch=4)
