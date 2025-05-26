from flask import Flask, render_template, Response


app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    from src.components.prediction.live_predict import LivePredictor, LivePredictorConfig
    predictor = LivePredictor(LivePredictorConfig())
    return Response(predictor.flask_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
