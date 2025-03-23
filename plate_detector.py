from ultralytics import YOLO
import torch

class PlateDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        device = "cpu"
        self.device = device

    def detect_plate(self, image_path):
        # image = Image.open(image_path)
        image = image_path
        result = self.model.predict(image, device=self.device,verbose = False)
        if result[0] :
            x1, y1, x2, y2 = map(int, result[0].boxes.xyxy[0])
            cropped_object = image.crop((x1, y1, x2, y2))
            return cropped_object
        else:
            print("Not Found License plate from this camera angle")
            return None