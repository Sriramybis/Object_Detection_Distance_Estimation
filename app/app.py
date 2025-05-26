from flask import Flask
from src.components.live_predictor import LivePredictor, LivePredictorConfig

if __name__ == "__main__":
    config = LivePredictorConfig()
    predictor = LivePredictor(config)
    predictor.run()


# app = Flask(__name__)

# @app.route('/')
# def index():
#     return "<h2>YOLOv5 Object Detection — visit /detect to open webcam</h2>"

# @app.route('/detect')
# def detect():
#     from src.components.live_predictor import LivePredictor, LivePredictorConfig
#     config = LivePredictorConfig()
#     predictor = LivePredictor(config)
#     predictor.run()
#     return "Detection finished."

# if __name__ == "__main__":
#     app.run(debug=True)
