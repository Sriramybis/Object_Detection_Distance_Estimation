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


## Getting Started

### Step 1: Clone the repo and set up environment

```bash
git clone https://github.com/your-username/obj_det_dist_est.git
cd obj_det_dist_est
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

### Step 2: Add images and annotations to train

Folder structure:

Output Directories created:
obj_det_dist_est/
├── data/
│   ├── images/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   ├── annotations/
│   │   ├── img1.txt
│   │   ├── img2.txt


Use [makesense.ai](https://www.makesense.ai/) for creating annotations. (Any annotation tool works. But output must be in yolo format)

---

### Step 3: Train your model


Train using `training_pipeline.py` if not done yet:

```bash
python -m src.pipeline.training_pipeline
```

Ensure `artifacts/best.pt` exists — this is the YOLOv5 model trained on your custom object (e.g., a bottle).  

---

### Step 4: Run the Flask app

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
- [ ] Model selection dropdown
- [ ] Confidence threshold slider
- [ ] Webcam stream to remote clients (This feature has been implemented for locally trained models.)
- [ ] Stream recording and snapshots (This feature has been implemented for locally trained models.)

---

## Credits

- [Ultralytics YOLOv5](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [Flask](https://flask.palletsprojects.com/)
- [makesense.ai](https://www.makesense.ai/)

---

> Built by Mythili Sekar Sriram
