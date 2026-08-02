# Hybrid Intelligent Object Detection System

A hybrid system that combines the real-time detection capabilities of **YOLO (You Only Look Once)** with the reasoning and zero-shot identification power of **Google Gemini**.

## 🚀 Overview

This project implements a multi-stage object detection pipeline:
1. **YOLO-World Detection**: Uses `yolov8s-world.pt` for efficient, open-vocabulary object detection.
2. **Ambiguity Resolution**: When detection confidence is low or the object is ambiguous, a cropped image is sent to the **Gemini 2.0 Flash** model.
3. **Generative Verification**: Gemini provides a short, context-aware description to verify or refine the initial detection.

## 🛠️ Tech Stack

- **Computer Vision**: OpenCV, Ultralytics YOLOv8 (YOLO-World)
- **Generative AI**: Google Gemini API (`google-generativeai`)
- **Programming**: Python
- **Environment**: Threading for asynchronous API calls to maintain high FPS.

## 📂 Project Structure

- `main.py`: Core application logic, handles camera input, YOLO inference, and UI rendering.
- `streamlit_app.py`: Streamlit interface for image upload, camera input, and live webcam inference.
- `gemini_engine.py`: Integration with Google Gemini API for image analysis.
- `yolov8s-world.pt`: Pre-trained YOLOv8 World model for open-vocabulary detection.
- `.env`: (To be created) Stores your `GEMINI_API_KEY`.

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.8+
- Webcam

### 2. Install Dependencies
```powershell
pip install opencv-python ultralytics google-generativeai python-dotenv pillow numpy
```

If you want the Streamlit interface, install from `requirements.txt`.

### 3. API Configuration
Create a `.env` file in the root directory and add your Google Gemini API key:
```env
GEMINI_API_KEY=your_api_key_here
```

## 🖥️ Usage

Run the main script:
```powershell
python main.py
```

Or launch the Streamlit app:
```powershell
streamlit run streamlit_app.py
```

### Key Controls
- **Detection**: The system automatically highlights objects.
- **Trigger**: When detection confidence falls between 15% and 40%, a crop is sent to Gemini.
- **Exit**: Press `q` to close the application.

## 🧠 How it Works

1. **Real-time Stream**: Captures video feed using OpenCV.
2. **YOLO-World**: Identifies common objects (pens, phones, bottles, etc.) instantly.
3. **Conditional Generative AI**:
   - If YOLO is "uncertain" (low confidence), it triggers `gemini_engine.py`.
   - The system adds a **40px padding** to the crop to provide Gemini with enough context.
   - A **15-second cooldown** is implemented to stay within API rate limits.
4. **Overlay**: Displays "GenAI: [Result]" on the live feed.

---

*Developed as a Hybrid Intelligent Object Detection System.*
