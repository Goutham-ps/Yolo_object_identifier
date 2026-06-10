from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uuid

from ultralytics import YOLO
from PIL import Image

import os

from PIL import Image

app = FastAPI()

templates = Jinja2Templates(directory="templates")

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

os.makedirs("uploads", exist_ok=True)

# Load YOLO once
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


    unique_name = f"{uuid.uuid4()}_{file.filename}"

    file_path = f"uploads/originals/{unique_name}"

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    results = model(file_path)
    annotated_image = results[0].plot()

    detections = []



    annotated_path = f"uploads/results/result_{unique_name}"

    Image.fromarray(
        annotated_image
    ).save( 
        annotated_path
    )


    

    for box in results[0].boxes:

        class_id = int(box.cls[0])

        confidence = float(box.conf[0])

        detections.append(
            {
                "name": model.names[class_id],
                "confidence": round(confidence * 100, 2)
            }
        )
    summary = {}

    for item in detections:

        name = item["name"]

        summary[name] = summary.get(
            name,
            0
        ) + 1


    context={
    "detections": detections,
    "summary": summary,
    "image": "/" + annotated_path
}


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