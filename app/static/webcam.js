const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const toggleButton = document.getElementById("toggleButton");

let detectionInterval = null;

// Setup webcam stream
navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
  video.srcObject = stream;
});

video.addEventListener("loadeddata", () => {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
});

function startDetection() {
  detectionInterval = setInterval(() => {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
      const formData = new FormData();
      formData.append("frame", blob);

      fetch("/predict_frame", {
        method: "POST",
        body: formData,
      })
        .then((res) => res.json())
        .then((data) => {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          data.detections.forEach((det) => {
            const [x1, y1, x2, y2] = det.bbox;
            ctx.strokeStyle = "lime";
            ctx.lineWidth = 2;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            ctx.fillStyle = "lime";
            ctx.fillText(
              `${det.label} ${det.confidence} (${det.distance_cm}cm)`,
              x1,
              y1 - 5
            );
          });
        });
    }, "image/jpeg");
  }, 700);
  toggleButton.textContent = "Stop Detection";
}

function stopDetection() {
  clearInterval(detectionInterval);
  detectionInterval = null;
  toggleButton.textContent = "Start Detection";
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

toggleButton.addEventListener("click", () => {
  if (detectionInterval) {
    stopDetection();
  } else {
    startDetection();
  }
});
