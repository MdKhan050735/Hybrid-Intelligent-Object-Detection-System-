import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLOWorld
import gemini_engine
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(page_title="Hybrid AI Vision", page_icon="🤖")

st.title("🤖 Hybrid Intelligent Object Detection")
st.markdown("### YOLO-World + Google Gemini 2.0")

# Load model (cached)
@st.cache_resource
def load_yolo():
    model = YOLOWorld('yolov8s-world.pt')
    model.set_classes(["pen", "pencil", "mobile phone", "laptop", "bottle", "keyboard", "mouse", "book", "cup", "watch", "wallet", "remote control", "person"])
    return model

model = load_yolo()

# Sidebar for settings
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("YOLO Confidence", 0.0, 1.0, 0.25)
padding = st.sidebar.slider("Crop Padding", 0, 100, 40)

# Input choice
input_mode = st.radio("Choose Input:", ["Upload Image", "Camera Input", "Live Webcam"])

input_image = None
if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        input_image = Image.open(uploaded_file)
elif input_mode == "Camera Input":
    cam_image = st.camera_input("Take a photo")
    if cam_image:
        input_image = Image.open(cam_image)
else:
    st.info("Live Video Mode: YOLO is running in real-time. Gemini analysis is disabled in Live Video for performance.")
    
    # We use st.camera_input or a custom component for REAL real-time, 
    # but Streamlit's native live webcam is usually best handled via camera_input 
    # for simplicity, or a loop with cv2 for local use.
    # Since the user asked for "Real time detection", let's use a CV2 loop if it's local.
    
    run_live = st.button("Start Live Feed")
    if run_live:
        stop_live = st.button("Stop Feed")
        FRAME_WINDOW = st.image([])
        cap = cv2.VideoCapture(0)
        
        while not stop_live:
            ret, frame = cap.read()
            if not ret:
                st.error("Cannot access webcam.")
                break
            
            # Resize for speed
            display_frame = cv2.resize(frame, (640, 480))
            
            # Run YOLO
            results = model(display_frame, conf=conf_threshold, verbose=False)
            
            # Draw results
            res_plotted = results[0].plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            # Display
            FRAME_WINDOW.image(res_rgb)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        st.write("Live Feed Stopped.")

if input_image:
    # Convert to OpenCV format
    frame = np.array(input_image)
    if frame.shape[2] == 4: # Handle RGBA
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Run YOLO
    results = model(frame_bgr, conf=conf_threshold)
    
    # Process results
    st.subheader("Results")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Draw on frame
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        st.image(res_rgb, caption="Processed Image", use_container_width=True)
        
    with col2:
        st.write("🔍 **Deep Analysis**")
        found_uncertain = False
        for result in results:
            for box in result.boxes:
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = model.names[cls]
                
                # Check for ambiguity (same logic as main.py)
                if 0.15 < conf < 0.45 and label != "person":
                    found_uncertain = True
                    st.warning(f"Ambiguous {label} detected ({conf:.2f}). Consulting Gemini...")
                    
                    # Crop for Gemini
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    h, w, _ = frame_bgr.shape
                    py1, py2 = max(0, y1 - padding), min(h, y2 + padding)
                    px1, px2 = max(0, x1 - padding), min(w, x2 + padding)
                    crop = frame_bgr[py1:py2, px1:px2]
                    
                    if crop.size > 0:
                        with st.spinner('Gemini is thinking...'):
                            description = gemini_engine.get_description(crop)
                            st.success(f"**Gemini Result:** {description}")
        
        if not found_uncertain:
            st.info("No ambiguous objects found for Gemini analysis.")
