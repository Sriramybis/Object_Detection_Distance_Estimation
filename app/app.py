from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np


app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    from src.components.prediction.live_predict import LivePredictor, LivePredictorConfig
    predictor = LivePredictor(LivePredictorConfig())
    file = request.files['frame']
    img_array = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    result_frame, detections = predictor.predict_from_frame(frame)

    # Optional: encode frame and return for preview
    _, buffer = cv2.imencode('.jpg', result_frame)
    encoded_img = buffer.tobytes()

    return jsonify({
        "detections": detections
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
