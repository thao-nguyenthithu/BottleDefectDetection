import os
import cv2
import numpy as np
import base64
import tempfile
import glob
from flask import Flask, request, render_template, Response, session, send_file
from ultralytics import YOLO

# --- CẤU HÌNH ---
app = Flask(__name__)
app.secret_key = "bottle_defect_secret"

# ĐƯỜNG DẪN
MODEL_PATH = "runs/detect/yolo11s_run_100_epochs2/weights/best.pt"
TRAIN_DIR = "runs/detect/yolo11s_run_100_epochs2"

# Load Model
try:
    model = YOLO(MODEL_PATH)
    print("Tải model thành công!")
except Exception as e:
    print(f"Lỗi tải model: {e}")
    model = None


# --- STREAM VIDEO ---
def gen_frames(temp_path):
    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        return
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
        if frame_count % 3 != 0:
            continue

        height, width = frame.shape[:2]
        new_width = 640
        new_height = int(height * (new_width / width))
        frame = cv2.resize(frame, (new_width, new_height))

        if model:
            results = model(frame, conf=0.6)
            annotated_frame = results[0].plot(img=frame)
        else:
            annotated_frame = frame

        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
    cap.release()


@app.route("/video_feed")
def video_feed():
    temp_path = session.get("temp_video_path", None)
    if not temp_path:
        return "No video", 204
    return Response(
        gen_frames(temp_path), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# --- PHỤC VỤ FILE ---
@app.route("/media/<path:filename>")
def serve_media_file(filename):
    file_path = os.path.join(TRAIN_DIR, filename)
    if os.path.exists(file_path):
        return send_file(file_path)
    return "File not found", 404


# --- TRANG KẾT QUẢ TRAIN ---
@app.route("/train_results")
def train_results():
    train_files = []
    if os.path.exists(TRAIN_DIR):
        for f in glob.glob(os.path.join(TRAIN_DIR, "**", "*.*"), recursive=True):
            ext = f.rsplit(".", 1)[-1].lower()
            if ext in {"jpg", "jpeg", "png", "bmp", "gif", "mp4", "avi", "mov"}:
                train_files.append(os.path.relpath(f, TRAIN_DIR).replace("\\", "/"))
    train_files.sort()
    return render_template(
        "train_results.html", train_files=train_files, train_dir=TRAIN_DIR
    )


# --- TRANG CHỦ ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        files = request.files.getlist("file")
        if not files or files[0].filename == "":
            return render_template("index.html")

        image_results = []
        video_found = False

        for file in files:
            filename = file.filename.lower()

            # XỬ LÝ ẢNH
            if filename.endswith((".jpg", ".jpeg", ".png")):
                file_bytes = np.frombuffer(file.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if model:
                    results = model(img, conf=0.6)
                    annotated_img = results[0].plot()
                else:
                    annotated_img = img

                _, buffer = cv2.imencode(".jpg", annotated_img)
                img_b64 = base64.b64encode(buffer).decode("utf-8")
                image_results.append({"name": file.filename, "b64": img_b64})

            # XỬ LÝ VIDEO
            elif filename.endswith((".mp4", ".avi", ".mov")) and not video_found:
                ext = os.path.splitext(filename)[1]
                temp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                temp.write(file.read())
                temp.close()
                session["temp_video_path"] = temp.name
                video_found = True

        return render_template(
            "index.html", image_results=image_results, video_mode=video_found
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
