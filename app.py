from flask import Flask
from src.components.prediction.live_predict import LivePredictor, LivePredictorConfig

if __name__ == "__main__":
    config = LivePredictorConfig()
    predictor = LivePredictor(config)
    predictor.run()