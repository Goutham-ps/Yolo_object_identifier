from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from ultralytics import YOLO
from PIL import Image

import uuid
import os

app = FastAPI()

# Templates
templates = Jinja2Templates(directory="templates")

# Static files
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

app.mount(
    "/uploads",
    StaticFiles(directory="uploads"),
    name="uploads"
)

# Create upload folders if they don't exist
os.makedirs("uploads/originals", exist_ok=True)
os.makedirs("uploads/results", exist_ok=True)

# Load YOLO model once at startup
model = YOLO("yolov8n.pt")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...)
):
    # Generate unique filename
    unique_name = f"{uuid.uuid4()}_{file.filename}"

    # Save original image
    file_path = f"uploads/originals/{unique_name}"

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Run YOLO inference
    results = model(file_path)

    # Create annotated image
    annotated_image = results[0].plot()

    annotated_path = f"uploads/results/result_{unique_name}"

    Image.fromarray(
        annotated_image
    ).save(
        annotated_path
    )

    # Detection details
    detections = []

    for box in results[0].boxes:

        class_id = int(box.cls[0])

        confidence = float(box.conf[0])

        detections.append(
            {
                "name": model.names[class_id],
                "confidence": round(confidence * 100, 2)
            }
        )

    # Detection summary
    summary = {}

    for item in detections:

        name = item["name"]

        summary[name] = summary.get(
            name,
            0
        ) + 1

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "detections": detections,
            "summary": summary,
            "image": "/" + annotated_path,
            "original_image": "/" + file_path
        }
    )