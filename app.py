from flask import Flask, request, render_template, jsonify
from ultralytics import YOLO
import os, uuid
import cv2

app = Flask(__name__)

MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

UPLOAD_DIR = os.path.join("static", "uploads")
RESULT_DIR = os.path.join("static", "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file"}), 400

    f = request.files["image"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    uid = str(uuid.uuid4())
    safe_name = f.filename.replace(" ", "_")
    in_name = f"{uid}_{safe_name}"
    in_path = os.path.join(UPLOAD_DIR, in_name)
    f.save(in_path)

    # Predict (save=False)
    results = model.predict(source=in_path, conf=0.25, iou=0.5, save=False)
    r0 = results[0]

    detections = []
    for box in r0.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        detections.append({
            "class_id": cls_id,
            "class_name": model.names[cls_id],
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })

    # Create result image ourselves (always consistent)
    out_folder = os.path.join(RESULT_DIR, uid)
    os.makedirs(out_folder, exist_ok=True)
    out_name = in_name  # same name
    out_path = os.path.join(out_folder, out_name)

    plotted = r0.plot()  # numpy image with boxes
    cv2.imwrite(out_path, cv2.cvtColor(plotted, cv2.COLOR_RGB2BGR))

    # URLs (must start with /static/...)
    input_url = f"/static/uploads/{in_name}"
    result_url = f"/static/results/{uid}/{out_name}"

    # OPTIONAL: single best class (highest confidence)
    best = None
    if detections:
        best = max(detections, key=lambda d: d["confidence"])

    return jsonify({
        "input_image": input_url,
        "result_image": result_url,
        "best_prediction": best,   # one final answer (optional)
        "detections": detections
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
