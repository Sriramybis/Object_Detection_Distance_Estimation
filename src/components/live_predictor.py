import cv2
from ultralytics import YOLO
from src.logger import logging
from src.exception import CustomException
import sys
from dataclasses import dataclass

@dataclass
class LivePredictorConfig:
    model_path: str = "artifacts/best.pt"
    class_names: list = ("bottle",)
    webcam_index: int = 1


class LivePredictor:
    def __init__(self, config):
        self.config = config
        self.model = YOLO(self.config.model_path)

    def draw_boxes(self, frame, results):
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                if conf < 0.8:
                    continue
                cls_id = int(box.cls[0])
                label = f"{self.config.class_names[cls_id]} {conf:.2f}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame

    def run(self):
        try:
            cap = cv2.VideoCapture(self.config.webcam_index)

            if not cap.isOpened():
                logging.error("Webcam not accessible")
                return

            logging.info("Starting live webcam detection")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model.predict(source=frame, stream=True)
                frame = self.draw_boxes(frame, results)

                cv2.imshow("Live Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            raise CustomException(e, sys)