import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

@dataclass
class DistanceEstimatorConfig:
    known_width_cm: float = 13.0      # Real-world width of the object (e.g., bottle)
    focal_length_px: float = 1080.0   # Camera focal length (estimated or calibrated)

class DistanceEstimator:
    def __init__(self, config: DistanceEstimatorConfig = DistanceEstimatorConfig()):
        self.config = config

    def estimate_distance(self, pixel_width: float) -> float:
        try:
            if pixel_width <= 0:
                logging.warning("Invalid pixel width received for distance estimation.")
                return -1.0
            distance = (self.config.known_width_cm * self.config.focal_length_px) / pixel_width
            logging.info(f"Estimated distance: {distance:.2f} cm for bbox width: {pixel_width:.2f} px")
            return distance
        except Exception as e:
            raise CustomException(e, sys)
