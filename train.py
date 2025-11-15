from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-cls.yaml")  # build a new model from YAML

# Train the model
results = model.train(
    data=r"C:\Users\thith\Downloads\dataset_image", epochs=15, imgsz=520
)
