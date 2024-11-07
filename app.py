from dotenv import load_dotenv
import os
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from keras.models import load_model
from flask import Flask, render_template, Response

load_dotenv()

# Load model and labels for gesture classification
MODEL_PATH = os.getenv("MODEL_PATH")
model = load_model(MODEL_PATH, compile=False)
LABELS_PATH = os.getenv("LABELS_PATH")
with open(LABELS_PATH, "r") as f:
    labels = [label.strip() for label in f]

# Initialize hand detector and camera (once to avoid overhead)
detector = HandDetector(maxHands=1)
cap = cv2.VideoCapture(0)

# Set up Flask application
app = Flask(__name__)

def get_frame():
    """Capture frames, predict gestures, and yield frames with predictions."""
    while True:
        success, img = cap.read()
        if not success or img is None:
            continue  # Skip if frame capture fails

        imgOutput = img.copy()
        hands, img = detector.findHands(img)  # Detect hand(s) in the frame
        
        if hands:
            # Extract and adjust region of interest around the detected hand
            hand = hands[0]
            x, y, w, h = hand['bbox']
            x, y = max(0, x), max(0, y)
            w, h = min(w, img.shape[1] - x), min(h, img.shape[0] - y)
            imgCrop = img[y:y + h, x:x + w]
            
            if imgCrop.size != 0:
                # Preprocess the image for prediction
                imgPreprocessed = cv2.resize(imgCrop, (224, 224))
                imgPreprocessed = np.asarray(imgPreprocessed, dtype=np.float32).reshape(1, 224, 224, 3)
                imgPreprocessed = (imgPreprocessed / 127.5) - 1
                
                try:
                    prediction1 = model.predict(imgPreprocessed)
                    predicted_class_index1 = np.argmax(prediction1)
                    predicted_label1 = labels[predicted_class_index1] if predicted_class_index1 < len(labels) else "Unrecognized gesture"
                
                except Exception as e:
                    print(f"Prediction error: {e}")
                    predicted_label1 = "Prediction error"

                # Display prediction and bounding box on output image
                cv2.putText(imgOutput, predicted_label1, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(imgOutput, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        else:
            # Show message if no hand is detected
            cv2.putText(imgOutput, "No hand detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Encode the frame to JPEG format for streaming
        _, jpeg = cv2.imencode('.jpg', imgOutput)
        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Flask routes setup
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pag3', methods=['POST', 'GET'])
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pag3/template')
def render_pag3_template():
    return render_template('pag3.html')

# Release resources when the app finishes
if __name__ == "__main__":
    try:
        app.run(debug=True)
    finally:
        cap.release()  # Release camera to avoid resource lock
