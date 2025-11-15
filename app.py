import os
import zipfile
import shutil
import glob
import re
import cv2
from flask import Flask, request, render_template_string, send_file, Response, session
from ultralytics import YOLO
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.secret_key = "bottle_defect_secret"
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "gif", "mp4", "avi", "mov", "zip"}
MODEL_PATH = "runs/detect/yolo11s_run_100_epochs2/weights/best.pt"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)


def gen_frames(video_path=None):
    if video_path is None:
        video_path = "test/final_test.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không mở được video: {video_path}")
        return
    bbox_colors = [
        (164, 120, 87),
        (68, 148, 228),
        (93, 97, 209),
        (178, 182, 133),
        (88, 159, 106),
        (96, 202, 231),
        (159, 124, 168),
        (169, 162, 241),
        (98, 118, 150),
        (172, 176, 184),
    ]
    while True:
        success, frame = cap.read()
        if not success:
            break
        results = model(frame)
        detections = results[0].boxes
        labels = model.names
        for i in range(len(detections)):
            xyxy_tensor = detections[i].xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)
            classidx = int(detections[i].cls.item())
            classname = labels[classidx]
            conf = detections[i].conf.item()
            if conf > 0.5:
                color = bbox_colors[classidx % 10]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                label = f"{classname}: {int(conf * 100)}%"
                labelSize, baseLine = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(
                    frame,
                    (xmin, label_ymin - labelSize[1] - 10),
                    (xmin + labelSize[0], label_ymin + baseLine - 10),
                    color,
                    cv2.FILLED,
                )
                cv2.putText(
                    frame,
                    label,
                    (xmin, label_ymin - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )
        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.route("/video_feed")
def video_feed():
    video_path = session.get("uploaded_video_path", None)
    if not video_path or not os.path.exists(video_path):

        def blank_frame():
            import numpy as np

            frame = np.zeros((384, 640, 3), dtype=np.uint8)
            ret, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

        return Response(
            blank_frame(), mimetype="multipart/x-mixed-replace; boundary=frame"
        )
    return Response(
        gen_frames(video_path), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/train_results")
def train_results():
    train_dir = os.path.join("runs", "detect", "yolo11s_run_100_epochs2")
    train_files = []
    if os.path.exists(train_dir):
        for f in glob.glob(os.path.join(train_dir, "**", "*.*"), recursive=True):
            ext = f.rsplit(".", 1)[-1].lower()
            if ext in ALLOWED_EXTENSIONS:
                rel_path = os.path.relpath(f, train_dir).replace("\\", "/")
                train_files.append(rel_path)
    html = """
    <!doctype html>
    <html lang=\"en\">
    <head>
        <meta charset=\"utf-8\">
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
        <title>Kết quả train model</title>
        <link href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css\" rel=\"stylesheet\">
        <style>
            body { background: #f8f9fa; }
            .container { max-width: 900px; margin-top: 40px; }
            .result-img, .result-video { max-width: 100%; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.12); margin-bottom: 8px; }
            .result-box { background: #fff; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.07); padding: 16px; margin-bottom: 18px; }
            .result-title { font-weight: bold; font-size: 1.05rem; margin-bottom: 8px; color: #343a40; }
            .back-btn { background: #007bff; color: #fff; font-weight: bold; border-radius: 8px; padding: 6px 18px; margin-bottom: 24px; text-decoration: none; display: inline-block; }
            .back-btn:hover { background: #0056b3; color: #fff; }
        </style>
    </head>
    <body>
        <div class=\"container\">
            <a href=\"/\" class=\"back-btn\">Quay về trang chủ</a>
            <h2 class=\"mb-4\">Kết quả train model</h2>
            <div class=\"row\">
                {% for f in train_files %}
                    <div class=\"col-md-6 col-lg-4 mb-2\">
                        <div class=\"result-box h-100\">
                            <div class=\"result-title text-truncate\">{{ f.split('/')[-1] }}</div>
                            {% if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif')) %}
                                <img src=\"/train_result/{{ f }}\" class=\"result-img\" alt=\"Result Image\">
                            {% elif f.lower().endswith(('mp4', 'avi', 'mov')) %}
                                <video src=\"/train_result/{{ f }}\" class=\"result-video\" controls></video>
                            {% endif %}
                        </div>
                    </div>
                {% endfor %}
            </div>
        </div>
        <script src=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js\"></script>
    </body>
    </html>
    """
    return render_template_string(html, train_files=train_files)


@app.route("/", methods=["GET", "POST"])
def upload_file():
    train_dir = os.path.join("runs", "detect", "yolo11s_run_100_epochs2")
    train_files = []
    if os.path.exists(train_dir):
        for f in glob.glob(os.path.join(train_dir, "**", "*.*"), recursive=True):
            ext = f.rsplit(".", 1)[-1].lower()
            if ext in ALLOWED_EXTENSIONS:
                rel_path = os.path.relpath(f, train_dir).replace("\\", "/")
                train_files.append(rel_path)
    html = """
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Bottle Defect Detection</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background: #f8f9fa; }
            .container { max-width: 900px; margin-top: 40px; }
            .card { margin-bottom: 30px; }
            .upload-btn { background: #007bff; color: #fff; font-weight: bold; }
            .upload-btn:hover { background: #0056b3; }
            .result-img, .result-video { max-width: 100%; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.12); margin-bottom: 8px; }
            .train-btn { background: #6c757d; color: #ffc107; font-weight: bold; padding: 4px 16px; font-size: 1rem; border-radius: 8px; position: absolute; top: 18px; left: 18px; z-index: 10; border: none; box-shadow: 0 2px 8px rgba(0,0,0,0.08); transition: background 0.2s, color 0.2s; }
            .train-btn:hover { background: #495057; color: #ffe066; }
        </style>
    </head>
    <body>
        <div class="container position-relative">
            <a href="/train_results" class="btn train-btn">Kết quả train model</a>
            <div class="text-center mb-4">
                <h1 class="display-5">Bottle Defect Detection</h1>
                <p class="lead">Upload an image, video, or zip file để phát hiện lỗi chai bằng YOLO</p>
            </div>
            <div class="card p-4 shadow-sm">
                <form method="post" enctype="multipart/form-data" class="row g-3 align-items-center justify-content-center">
                    <div class="col-auto">
                        <input type="file" name="file" class="form-control" required>
                    </div>
                    <div class="col-auto">
                        <button type="submit" class="btn upload-btn">Upload & Detect</button>
                    </div>
                </form>
            </div>
            <div class="card p-4 shadow-sm mb-4">
                <h2 class="mb-3">YOLO Realtime Stream</h2>
                <img src="/video_feed" width="640" height="384" style="border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.12);">
            </div>
            {% if uploaded_video %}
            <div class="card p-4 shadow-sm mb-4">
                <h2 class="mb-3">Video bạn vừa upload</h2>
                <video src="{{ uploaded_video }}" class="result-video" controls></video>
            </div>
            {% endif %}
            {% if result_files %}
            <div class="card p-4 shadow-sm">
                <h2 class="mb-4">Prediction Results</h2>
                <div class="row">
                    {% for f in result_files %}
                        <div class="col-md-6 col-lg-4 mb-4">
                            <div class="result-box h-100">
                                <div class="result-title text-truncate">{{ f.split('/')[-1] }}</div>
                                {% if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif')) %}
                                    <img src="/train_result/{{ f }}" class="result-img" alt="Result Image">
                                {% elif f.lower().endswith(('mp4', 'avi', 'mov')) %}
                                    <video src="/train_result/{{ f }}" class="result-video" controls></video>
                                {% endif %}
                            </div>
                        </div>
                    {% endfor %}
                </div>
            </div>
            {% endif %}
        </div>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    result_files = []
    uploaded_video = None
    if request.method == "POST":
        if "file" not in request.files:
            return render_template_string(
                html,
                result_files=None,
                train_files=train_files,
                error="Không có file upload!",
            )
        file = request.files["file"]
        if file.filename == "":
            return render_template_string(
                html,
                result_files=None,
                train_files=train_files,
                error="Chưa chọn file!",
            )
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            input_files = []
            ext = file_path.rsplit(".", 1)[-1].lower()
            if filename.lower().endswith(".zip"):
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(UPLOAD_FOLDER)
                input_files = [
                    f
                    for f in glob.glob(
                        os.path.join(UPLOAD_FOLDER, "**", "*.*"), recursive=True
                    )
                    if f.rsplit(".", 1)[-1].lower()
                    in {"jpg", "jpeg", "png", "bmp", "gif", "mp4", "avi", "mov"}
                    and f != file_path
                ]
                if not input_files:
                    return render_template_string(
                        html,
                        result_files=None,
                        error="Không tìm thấy ảnh/video hợp lệ trong file zip!",
                        train_files=train_files,
                        uploaded_video=None,
                    )
            else:
                if ext in {"jpg", "jpeg", "png", "bmp", "gif", "mp4", "avi", "mov"}:
                    input_files = [file_path]
                    # Nếu là video thì show preview và lưu vào session để stream realtime
                    if ext in {"mp4", "avi", "mov"}:
                        uploaded_video = "/" + file_path.replace("\\", "/")
                        session["uploaded_video_path"] = file_path
                else:
                    return render_template_string(
                        html,
                        result_files=None,
                        train_files=train_files,
                        error="File không phải là ảnh hoặc video!",
                        uploaded_video=None,
                    )

            if len(input_files) == 1 and input_files[0].rsplit(".", 1)[-1].lower() in {
                "mp4",
                "avi",
                "mov",
            }:
                model.predict(
                    source=input_files[0],
                    save=True,
                    project=RESULT_FOLDER,
                    name="predict",
                    imgsz=640,
                    device="cpu",
                    show=False,
                )
            else:
                model.predict(
                    source=input_files,
                    save=True,
                    project=RESULT_FOLDER,
                    name="predict",
                    imgsz=640,
                    device="cpu",
                    show=False,
                )
            predict_dirs = [
                d for d in os.listdir(RESULT_FOLDER) if re.match(r"^predict\d*$", d)
            ]
            if not predict_dirs:
                predict_dirs = ["predict"]

            def predict_sort_key(x):
                num = x.replace("predict", "")
                return int(num) if num.isdigit() else 0

            predict_dirs.sort(key=predict_sort_key, reverse=True)
            latest_predict_dir = predict_dirs[0]
            predict_dir = os.path.join(RESULT_FOLDER, latest_predict_dir)
            if os.path.exists(predict_dir):
                video_exts = {"mp4", "avi", "mov"}
                image_exts = {"jpg", "jpeg", "png", "bmp", "gif"}
                video_files = []
                image_files = []
                for f in glob.glob(os.path.join(predict_dir, "*.*")):
                    ext = f.rsplit(".", 1)[-1].lower()
                    rel_path = os.path.relpath(f, RESULT_FOLDER).replace("\\", "/")
                    if ext in video_exts:
                        video_files.append(rel_path)
                    elif ext in image_exts:
                        image_files.append(rel_path)
                if video_files:
                    result_files.extend(video_files)
                else:
                    result_files.extend(image_files)
            shutil.rmtree(UPLOAD_FOLDER)
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            return render_template_string(
                html,
                result_files=result_files,
                train_files=train_files,
                uploaded_video=uploaded_video,
            )
        else:
            return render_template_string(
                html,
                result_files=None,
                train_files=train_files,
                error="File không hợp lệ!",
                uploaded_video=None,
            )
    uploaded_video = None
    video_path = session.get("uploaded_video_path", None)
    if video_path and os.path.exists(video_path):
        uploaded_video = "/" + video_path.replace("\\", "/")
    return render_template_string(
        html, result_files=None, train_files=train_files, uploaded_video=uploaded_video
    )


@app.route("/train_result/<path:filename>")
def train_result_file(filename):
    train_dir = os.path.join("runs", "detect", "yolo11s_run_100_epochs2")
    file_path_train = os.path.join(train_dir, filename)
    if os.path.exists(file_path_train):
        return send_file(file_path_train)
    file_path_result = os.path.join(RESULT_FOLDER, filename)
    if os.path.exists(file_path_result):
        return send_file(file_path_result)
    return "File not found", 404


if __name__ == "__main__":
    app.run(debug=True)
