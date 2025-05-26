from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import sys
import os
import shutil

@dataclass
class ModelTrainerConfig:
    data_yaml: str = "data.yaml"
    model_arch: str = "yolov5s.pt"
    epochs: int = 10
    imgsz: int = 640
    project: str = "runs/train"
    name: str = "bottle-model"
    saved_model_path: str = "artifacts/best.pt"

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig = ModelTrainerConfig()):
        self.config = config

    def initiate_model_trainer(self):
        try:
            logging.info("Starting YOLOv5 training...")

            from ultralytics import YOLO
            model = YOLO(self.config.model_arch)

            model.train(
                data=self.config.data_yaml,
                epochs=self.config.epochs,
                imgsz=self.config.imgsz,
                project=self.config.project,
                name=self.config.name
            )

            logging.info("Training completed. Model saved in runs/train/")

            # Path to YOLO's best model
            best_model_src = os.path.join(self.config.project, self.config.name, "weights", "best.pt")
            best_model_dst = self.config.saved_model_path

            os.makedirs(os.path.dirname(best_model_dst), exist_ok=True)
            shutil.copy2(best_model_src, best_model_dst)

            logging.info(f"Model copied to {best_model_dst}")

        except Exception as e:
            raise CustomException(e, sys)
