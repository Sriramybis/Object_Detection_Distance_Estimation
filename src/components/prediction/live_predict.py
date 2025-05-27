import cv2
from ultralytics import YOLO
from src.logger import logging
from src.exception import CustomException
import sys
from dataclasses import dataclass
from src.components.prediction.distance_estimator import DistanceEstimator

@dataclass
class LivePredictorConfig:
    model_path: str = "artifacts/best.pt"
    class_names: list = ("bottle",)

class LivePredictor:
    def __init__(self, config: LivePredictorConfig):
        self.config = config
        self.model = YOLO(self.config.model_path)
        self.distance_estimator = DistanceEstimator()

    def draw_boxes(self, frame, results):
        detections = []

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < 0.8:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox_width = x2 - x1
                distance_cm = self.distance_estimator.estimate_distance(bbox_width)
                cls_id = int(box.cls[0])
                label = f"{self.config.class_names[cls_id]} {conf:.2f}, {distance_cm:.1f}cm"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                detections.append({
                    "label": self.config.class_names[cls_id],
                    "confidence": round(conf, 2),
                    "distance_cm": round(distance_cm, 1),
                    "bbox": [x1, y1, x2, y2]
                })

        return frame, detections

    def predict_from_frame(self, frame):
        try:
            results = self.model.predict(source=frame, stream=True)
            return self.draw_boxes(frame, results)
        except Exception as e:
            raise CustomException(e, sys)
