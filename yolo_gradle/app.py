from ultralytics import YOLO
import gradio as gr
from PIL import Image

# Load YOLO model once
model = YOLO("yolov8n.pt")


def detect(image, confidence):

    if image is None:
        return None, "Please upload an image first.", "0"

    results = model(
        image,
        conf=confidence
    )

    annotated = results[0].plot()

    detections = {}

    total_objects = 0

    for box in results[0].boxes:

        cls = int(box.cls[0])

        name = model.names[cls]

        detections[name] = detections.get(name, 0) + 1

        total_objects += 1

    summary = "📊 Objects Found\n\n"

    if detections:

        for obj, count in detections.items():

            summary += f"• {obj}: {count}\n"

    else:

        summary += "No objects detected."

    return (
        annotated,
        summary,
        str(total_objects)
    )


css = """
.gradio-container {
    max-width: 1400px !important;
    margin: auto;
}

footer {
    display: none !important;
}

h1 {
    text-align: center;
}

.stats-box {
    text-align:center;
    font-size:20px;
    font-weight:bold;
}
"""


with gr.Blocks(
    css=css,
    theme=gr.themes.Soft(
        primary_hue="blue"
    )
) as demo:

    gr.Markdown(
        """
        # 🚀 YOLO Object Detector

        Detect people, vehicles, phones, animals and 80+ objects using **YOLOv8**.
        """
    )

    with gr.Row():

        input_img = gr.Image(
            type="pil",
            label="📤 Upload Image",
            height=500
        )

        output_img = gr.Image(
            label="🎯 Detection Result",
            height=500
        )

    confidence = gr.Slider(
        minimum=0.10,
        maximum=1.00,
        value=0.25,
        step=0.05,
        label="🎚 Confidence Threshold"
    )

    detect_btn = gr.Button(
        "🔍 Detect Objects",
        variant="primary"
    )

    with gr.Row():

        total_count = gr.Textbox(
            label="📦 Total Objects Detected"
        )

    summary = gr.Textbox(
        label="📊 Detection Summary",
        lines=8
    )

    detect_btn.click(
        fn=detect,
        inputs=[
            input_img,
            confidence
        ],
        outputs=[
            output_img,
            summary,
            total_count
        ]
    )

    gr.Markdown(
        """
        ---
        ### ⚡ Built with YOLOv8 + Gradio + Hugging Face

        Portfolio Project by **Goutham**
        """
    )

demo.launch()