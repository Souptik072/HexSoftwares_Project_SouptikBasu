import cv2
import numpy as np
import os
import urllib.request
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════
#  ✏️  USER SETTINGS — Change MODE here
# ═══════════════════════════════════════════════════════════
# Choose: "webcam", "image", or "video"

MODE = "image"  # change mode according to your preference  

# File Paths (Use r"" for Windows paths)
IMAGE_INPUT = r"C:/Users/SOUPTIK/children.jpg"
VIDEO_INPUT = r"C:/Users/SOUPTIK/cid.mp4"

# Output Filenames
IMAGE_OUTPUT = "detected_face.jpg"
VIDEO_OUTPUT = "processed_video.mp4"

# Detection Sensitivity (0.1 to 1.0)
CONFIDENCE_THRESHOLD = 0.5

# ─────────────────────────────────────────────
#  1. Model Setup (Auto-Download)
# ─────────────────────────────────────────────

MODEL_PROTOTXT   = "deploy.prototxt"
MODEL_CAFFEMODEL = "res10_300x300_ssd_iter_140000.caffemodel"

def prepare_models():
    """Ensures model files are present in the Spyder working directory."""
    urls = {
        MODEL_PROTOTXT: "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        MODEL_CAFFEMODEL: "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    }
    for name, url in urls.items():
        if not os.path.exists(name):
            print(f"[INFO] Downloading {name}... please wait.")
            urllib.request.urlretrieve(url, name)
    return cv2.dnn.readNetFromCaffe(MODEL_PROTOTXT, MODEL_CAFFEMODEL)

# ─────────────────────────────────────────────
#  2. Core Detection Logic
# ─────────────────────────────────────────────

def detect_and_draw(net, frame):
    """Processes a single frame and draws bounding boxes."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    face_count = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD:
            face_count += 1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            
            # Clip coordinates to stay within image frame
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w-1, x2), min(h-1, y2)
            
            # Draw Box and Label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Face {face_count}: {confidence*100:.1f}%"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    return frame, face_count

# ─────────────────────────────────────────────
#  3. Mode Functions
# ─────────────────────────────────────────────

def run_image_mode(net):
    img = cv2.imread(IMAGE_INPUT)
    if img is None:
        print(f"[ERROR] Image not found at {IMAGE_INPUT}"); return
    
    result, count = detect_and_draw(net, img)
    cv2.imwrite(IMAGE_OUTPUT, result)
    
    # Display in Spyder's Plots pane
    plt.figure(figsize=(10,6))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected {count} Face(s)")
    plt.axis('off'); plt.show()
    print(f"[SUCCESS] Saved to {IMAGE_OUTPUT}")

def run_video_mode(net):
    cap = cv2.VideoCapture(VIDEO_INPUT)
    if not cap.isOpened():
        print(f"[ERROR] Video not found at {VIDEO_INPUT}"); return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    out = cv2.VideoWriter(VIDEO_OUTPUT, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    print("[INFO] Processing video... Press 'q' in the popup to cancel.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        processed_frame, _ = detect_and_draw(net, frame)
        out.write(processed_frame)
        cv2.imshow("Video Processing (Press Q to Quit)", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); out.release(); cv2.destroyAllWindows()
    print(f"[SUCCESS] Video saved as {VIDEO_OUTPUT}")

def run_webcam_mode(net):
    cap = cv2.VideoCapture(0)
    print("[INFO] Webcam active. Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        output, count = detect_and_draw(net, frame)
        cv2.putText(output, f"Total Faces: {count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Live Face Detection", output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release(); cv2.destroyAllWindows()

# ─────────────────────────────────────────────
#  4. Main Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    try:
        face_net = prepare_models()
        
        if MODE == "image":
            run_image_mode(face_net)
        elif MODE == "video":
            run_video_mode(face_net)
        elif MODE == "webcam":
            run_webcam_mode(face_net)
        else:
            print("[ERROR] Invalid MODE. Use 'image', 'video', or 'webcam'.")
            
    except Exception as e:
        print(f"[CRITICAL ERROR] {e}")