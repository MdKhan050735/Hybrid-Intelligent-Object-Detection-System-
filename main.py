import cv2
import time
from ultralytics import YOLOWorld
import gemini_engine
import threading

# Load a YOLO-World model
model = YOLOWorld('yolov8s-world.pt')

# Define the classes
model.set_classes(["pen", "pencil", "mobile phone", "laptop", "bottle", "keyboard", "mouse", "book", "cup", "watch", "wallet", "remote control", "person"])

# Gemini response management
gemini_output = ""
is_analyzing = False
last_analysis_time = 0
COOLDOWN = 30  # Safety cooldown for free tier limits
PADDING = 40  

def call_gemini_async(crop):
    global gemini_output, is_analyzing, last_analysis_time
    is_analyzing = True
    gemini_output = "Analyzing..."
    description = gemini_engine.get_description(crop)
    gemini_output = description
    is_analyzing = False
    last_analysis_time = time.time()

def main():
    global gemini_output, is_analyzing, last_analysis_time
    
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Standard resolution for better recognition accuracy
        frame_proc = cv2.resize(frame, (640, 480))

        # Run YOLO inference with standard parameters for better stability
        results = model(frame_proc, conf=0.25)
        
        for result in results:
            for box in result.boxes:
                # Get coordinates and metadata
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = model.names[cls]

                # Draw bounding box
                color = (0, 255, 0) if conf > 0.45 else (0, 165, 255)
                cv2.rectangle(frame_proc, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_proc, f"{label} {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # TRIGGER: Conditional check for Gen-AI analysis
                if (0.15 < conf < 0.40) and label != "person" and not is_analyzing:
                    current_time = time.time()
                    if current_time - last_analysis_time > COOLDOWN:
                        h, w, _ = frame_proc.shape
                        py1, py2 = max(0, y1 - PADDING), min(h, y2 + PADDING)
                        px1, px2 = max(0, x1 - PADDING), min(w, x2 + PADDING)
                        
                        crop = frame_proc[py1:py2, px1:px2]
                        if crop.size > 0:
                            threading.Thread(target=call_gemini_async, args=(crop,)).start()

        # Display Gemini analysis result
        if gemini_output:
            cv2.putText(frame_proc, f"GenAI: {gemini_output}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Hybrid AI Vision (YOLO + Gemini)", frame_proc)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
