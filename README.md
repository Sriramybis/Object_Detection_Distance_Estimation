# Real-Time Object Detection & Distance Estimation using YOLOv5 + Flask

This project performs real-time object detection and distance estimation using YOLOv5 and OpenCV, served through a Flask web interface.

Users can start and stop the webcam stream, with bounding boxes and distance labels drawn over detected objects (e.g., bottles). The app uses a custom-trained YOLOv5 model.

---

## Features

- Live video stream from webcam
- Object detection using YOLOv5
- Distance estimation using bounding box width + focal length
- MJPEG video feed displayed in browser
- Start/Stop button to control the camera stream
- Clean frontend with HTML/CSS

---

## Project Structure

```
obj_det_dist_est/
├── app/
│   ├── app.py               # Flask entry point
│   ├── templates/
│   │   └── index.html       # Frontend HTML
│   └── static/
│       └── style.css        # CSS styles
├── src/
│   ├── components/
│   │   ├── live_predictor.py         # Live webcam logic
│   │   └── prediction/
│   │       └── distance_estimator.py # Distance calculation logic
│   └── ...
├── artifacts/
│   └── best.pt              # Trained YOLOv5 model
├── data/
├── README.md
```

---

## Getting Started

### Step 1: Clone the repo and set up environment

```bash
git clone https://github.com/your-username/obj_det_dist_est.git
cd obj_det_dist_est
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r docker/requirements.txt
```

---

### Step 2: Make sure you have a trained model

Ensure `artifacts/best.pt` exists — this is the YOLOv5 model trained on your custom object (e.g., a bottle).  
Train using `ultralytics` if not done yet:

```bash
yolo detect train data=data.yaml model=yolov5s.pt epochs=50 imgsz=640
```

Copy the resulting best model to `artifacts/best.pt`.

---

### Step 3: Run the Flask app

```bash
python -m app.app
```

Go to: [http://localhost:5050](http://localhost:5050)

---

## Distance Estimation

Estimated using the pinhole camera model:

\[
\text{distance} = \frac{\text{known object width} \times \text{focal length}}{\text{bounding box pixel width}}
\]

You can change these parameters in:

```python
src/components/prediction/distance_estimator.py
```

---

## Frontend Controls

- ▶ **Start Detection** — begins webcam stream with real-time YOLO predictions
- ⏹ **Stop Detection** — stops the video feed
- Stream refreshes dynamically without restarting Flask

---

## Deployment Notes & Issues Faced

### ✅ Deployment Method:

- Docker → Amazon ECR → Amazon EC2 (using GitHub Actions)

### 🛠️ Issues Encountered & Fixes:

| Issue                                        | Solution                                                                       |
| -------------------------------------------- | ------------------------------------------------------------------------------ |
| `cv2.imshow()` error in Flask                | Replaced with MJPEG streaming using `yield` in `flask_stream()` method         |
| Port conflict (Flask default 5000)           | Changed to port `5050` and updated `EXPOSE 5050` in Dockerfile                 |
| `pyqt5` and `lxml` install failure in Docker | Removed unused packages from `requirements.txt`                                |
| YOLOv5 unable to write settings in `/root`   | YOLO defaulted to `/tmp/Ultralytics` without issues                            |
| Webcam not accessible on deployed EC2        | Shifted to JavaScript-based webcam input or ensured camera device availability |
| HTTPS for webcam                             | Used Ngrok as a tunneling solution when no domain was available                |

---

## Requirements

- Python 3.8+
- Flask
- OpenCV
- Ultralytics (YOLOv5): `pip install ultralytics`
- Webcam

---

## Coming Soon

- [ ] User uploads for custom object training
- [ ] Webcam stream to remote clients
- [ ] Stream recording and snapshots
- [ ] Model selection dropdown
- [ ] Confidence threshold slider

---

## Credits

- [Ultralytics YOLOv5](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [Flask](https://flask.palletsprojects.com/)
- [makesense.ai](https://www.makesense.ai/)

---

> Built by Mythili Sekar Sriram
