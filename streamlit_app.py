import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path

st.set_page_config(page_title="CFU Colony Counter", layout="wide")

MODEL_PATH = Path(__file__).parent / "best.pt"


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


model = load_model()

st.title("CFU Colony Counter")
st.markdown(
    "Upload a petri dish image to count bacterial colonies. "
    "Model: YOLOv8s trained on E. coli petri dish images. Validation mAP@50: 0.812."
)

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Petri Dish Image", type=["png", "jpg", "jpeg"])

    with st.expander("Detection Settings"):
        conf_threshold = st.slider(
            "Confidence Threshold", 0.05, 0.90, 0.25, 0.05,
            help="Lower values detect more colonies including uncertain ones."
        )
        iou_threshold = st.slider(
            "IoU Threshold", 0.10, 0.90, 0.30, 0.05,
            help="Lower values reduce duplicate detections."
        )

    run_btn = st.button("Count Colonies", type="primary")

with col2:
    if uploaded_file is not None and run_btn:
        image = Image.open(uploaded_file).convert("RGB")
        image_rgb = np.array(image)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        with st.spinner("Running detection..."):
            results = model(
                image_bgr, conf=conf_threshold, iou=iou_threshold,
                imgsz=1280, verbose=False
            )

        boxes = results[0].boxes
        cfu_count = len(boxes)
        avg_conf = float(boxes.conf.mean()) if cfu_count > 0 else 0.0
        min_conf = float(boxes.conf.min()) if cfu_count > 0 else 0.0
        max_conf = float(boxes.conf.max()) if cfu_count > 0 else 0.0

        st.image(image_rgb, caption="Input Image", use_container_width=True)
        st.metric("CFU Count", cfu_count)
        if cfu_count > 0:
            st.text(f"Avg: {avg_conf:.2f}   Min: {min_conf:.2f}   Max: {max_conf:.2f}")
        else:
            st.text("No colonies detected.")
    elif uploaded_file is not None:
        st.image(uploaded_file, caption="Preview — click 'Count Colonies' to run detection", use_container_width=True)
    else:
        st.info("Upload an image to get started.")

st.markdown("---")
st.markdown(
    "Model: YOLOv8s fine-tuned on AGAR dataset (E. coli subset) and Roboflow annotated petri dish images. "
    "Training set: 1,368 images."
)
