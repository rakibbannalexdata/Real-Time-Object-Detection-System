import streamlit as st
import requests
import time
import json
from PIL import Image, ImageDraw, ImageFont
import io
import os
import cv2
import numpy as np
import hashlib

# --- Configuration ---
st.set_page_config(
    page_title="Observer AI | Object Detection Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
    }
    [data-testid="stSidebar"] {
        background-color: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(59, 130, 246, 0.2);
        border-bottom: 2px solid #3b82f6 !important;
    }
    div.stButton > button:first-child {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #2563eb;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("🤖 Observer AI")
    st.markdown("---")
    
    api_base_url = st.text_input("Backend API URL", value="http://127.0.0.1:8000")
    st.markdown(f"**Status:** {'🟢 Online' if requests.get(f'{api_base_url}/health').status_code == 200 else '🔴 Offline'}" if api_base_url else "")
    
    st.markdown("---")
    st.subheader("📁 Project Management")
    
    # Try to fetch existing projects
    projects = []
    try:
        # In a real scenario, we might have an endpoint for this. 
        # For now, we assume coco128-seg as default.
        projects = ["coco128-seg","weedsVsCrops","stout"] 
    except:
        pass
        
    project_name = st.selectbox("Current Project", options=projects if projects else ["coco128-seg"])
    
    st.info("Select a project to manage its training or run inference with its best weights.")

# --- Main Content ---
st.title("Object Detection Hub")
st.caption("End-to-end training and inference workflow for YOLOv8 Segmentation models.")

tab1, tab2 = st.tabs(["🚀 Training Center", "🖼️ Inference Studio"])

# --- Tab 1: Training Center ---
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("New Training Session")
        with st.form("training_form"):
            train_project = st.text_input("Project Name", value=project_name)
            epochs = st.slider("Epochs", 1, 300, 50)
            imgsz = st.number_input("Image Size", value=640, step=32)
            batch = st.number_input("Batch Size", value=16, step=1)
            
            submit_train = st.form_submit_button("Start Training")
            
            if submit_train:
                try:
                    payload = {
                        "project_name": train_project,
                        "epochs": epochs,
                        "imgsz": imgsz,
                        "batch": batch
                    }
                    resp = requests.post(f"{api_base_url}/api/v1/training/start", json=payload)
                    if resp.status_code == 200:
                        st.success("🚀 Training started in background!")
                    else:
                        st.error(f"Error: {resp.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")

    with col2:
        st.subheader("Training Progress")
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        metrics_placeholder = st.empty()
        
        # Polling Loop
        # We use a state to handle polling toggle if needed, or just auto-poll
        if st.checkbox("Toggle Auto-Polling Status", value=True):
            try:
                while True:
                    resp = requests.get(f"{api_base_url}/api/v1/training/status?project={train_project}")
                    if resp.status_code == 200:
                        data = resp.json()
                        status = data['status']
                        progress = data['progress'] / 100.0
                        msg = data['message']
                        epoch = data['current_epoch']
                        total = data['total_epochs']
                        
                        status_placeholder.markdown(f"**Current Status:** `{status.upper()}`")
                        progress_bar.progress(progress)
                        metrics_placeholder.info(f"📍 **Epoch:** {epoch}/{total} | **Message:** {msg}")
                        
                        if status in ["completed", "failed"]:
                            if status == "completed":
                                st.balloons()
                                st.success("Done!")
                            break
                    time.sleep(5)
                    st.rerun() # Use experimental_rerun or just wait
            except Exception as e:
                st.write("Waiting for active training session...")

# --- Tab 2: Inference Studio ---
with tab2:
    st.subheader("Run Inference")
    
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)
    
    if uploaded_file is not None:
        # Load and convert to RGB to ensure drawing works reliably
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original Image", use_column_width=True)
        
        debug_mode = st.checkbox("Show Raw API Response", value=False)
        
        if st.button("🚀 Run Inference", use_container_width=True):
            with st.spinner("🔬 Analyzing image..."):
                try:
                    # Reset file pointer just in case
                    uploaded_file.seek(0)
                    file_bytes = uploaded_file.getvalue()
                    
                    # Use best weights for the project
                    model_path = f"models/{project_name}/weights/best.pt"
                    params = {
                        "model_path": model_path,
                        "confidence_threshold": conf_threshold
                    }
                    
                    resp = requests.post(
                        f"{api_base_url}/api/v1/detect/image",
                        params=params,
                        files={"file": (uploaded_file.name, file_bytes, "image/jpeg")},
                        timeout=30
                    )
                    
                    if resp.status_code == 200:
                        result = resp.json()
                        detections = result.get('detections', [])
                        
                        if debug_mode:
                            st.json(result)
                            
                        if not detections:
                            st.warning(f"No objects detected above {conf_threshold} confidence.")
                        else:
                            st.success(f"✅ Found {len(detections)} objects!")
                            
                            # --- Visualization Logic ---
                            # Convert PIL image to OpenCV BGR for advanced drawing
                            img_np = np.array(image.convert("RGB"))
                            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                            h, w = img_np.shape[:2]
                            
                            # Create a separate overlay for semi-transparent polygons
                            overlay = img_np.copy()
                            
                            # Color palette for consistent class colors
                            class_colors = {}
                            def get_colors(cls_name):
                                if cls_name not in class_colors:
                                    import hashlib
                                    h_hash = hashlib.md5(cls_name.encode()).hexdigest()
                                    r = int(h_hash[0:2], 16) % 200 + 55
                                    g = int(h_hash[2:4], 16) % 200 + 55
                                    b = int(h_hash[4:6], 16) % 200 + 55
                                    class_colors[cls_name] = (b, g, r) # BGR
                                base = class_colors[cls_name]
                                return base

                            sorted_dets = sorted(detections, key=lambda x: (x['bbox'][2]-x['bbox'][0]) * (x['bbox'][3]-x['bbox'][1]), reverse=True)
                            
                            for det in sorted_dets:
                                cls_name = det['class']
                                color_bgr = get_colors(cls_name)
                                
                                # 1. Draw Polygon / Mask
                                if 'segmentation' in det and det.get('segmentation'):
                                    polygon = det['segmentation']
                                    # DetectionService returns list of [x, y]
                                    pts = np.array([[p[0] * w, p[1] * h] for p in polygon], np.int32)
                                    pts = pts.reshape((-1, 1, 2))
                                    
                                    if len(pts) > 2:
                                        # Fill on overlay
                                        cv2.fillPoly(overlay, [pts], color_bgr)
                                        # Sharp border on main image with anti-aliasing
                                        cv2.polylines(img_np, [pts], True, color_bgr, 1, cv2.LINE_AA)

                            # Blend overlay with main image (semi-transparent fill)
                            alpha = 0.3
                            cv2.addWeighted(overlay, alpha, img_np, 1 - alpha, 0, img_np)
                            
                            # Convert back to PIL for high-quality text drawing
                            image = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
                            draw_final = ImageDraw.Draw(image)
                            
                            # Final pass: Draw labels on top of everything
                            for det in sorted_dets:
                                cls_name = det['class']
                                conf = det['confidence']
                                color_bgr = get_colors(cls_name)
                                color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0]) # Convert back for PIL
                                box = [det['bbox'][0]*w, det['bbox'][1]*h, det['bbox'][2]*w, det['bbox'][3]*h]
                                label = f"{cls_name} {conf:.2f}"
                                
                                try:
                                    font = None
                                    font_size = max(24, int(h / 30))
                                    # Try various font paths
                                    font_paths = [
                                        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                                        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
                                        "arial.ttf"
                                    ]
                                    for path in font_paths:
                                        try:
                                            font = ImageFont.truetype(path, font_size)
                                            break
                                        except: continue
                                    
                                    if not font: font = ImageFont.load_default()
                                    
                                    # Position label: slightly above the top of the box
                                    text_coords = (box[0], box[1] - font_size - 5 if box[1] > (font_size + 10) else box[1])
                                    txt_bbox = draw_final.textbbox(text_coords, label, font=font)
                                    
                                    # Add padding to background box
                                    pad = 2
                                    padded_bbox = (txt_bbox[0]-pad, txt_bbox[1]-pad, txt_bbox[2]+pad, txt_bbox[3]+pad)
                                    
                                    # Draw solid background box
                                    draw_final.rectangle(padded_bbox, fill=color_rgb)
                                    # Draw crisp white text
                                    draw_final.text(text_coords, label, fill="white", font=font)
                                except Exception as e:
                                    # Fallback
                                    draw_final.text((box[0], box[1]), label, fill="red")

                            st.image(image, caption="Enhanced Detection Results", use_column_width=True)
                    else:
                        st.error(f"❌ API Error ({resp.status_code}): {resp.text}")
                except Exception as e:
                    st.error(f"🔥 Critical Error: {str(e)}")
                    st.exception(e)
