import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from keras.models import load_model
from flask import Flask, render_template, Response

# load model and labels for gesture classification
model_path = "./Model/keras_model.keras"
model = load_model(model_path, compile=False)
labels_path = "./Model/labels.txt"
with open(labels_path, "r") as f:
    labels = [label.strip() for label in f]

# initialize hand detector and camera (once to avoid overhead)
detector = HandDetector(maxHands=1)
cap = cv2.VideoCapture(0)

# set up flask application
app = Flask(__name__)


def get_frame():

    while True:
        """Capture frames in real-time, predict gestures, and return the frame with the result."""
        success, img = cap.read()
        if not success or img is None:
            continue 

        imgOutput = img.copy()
        hands, img = detector.findHands(img) # detect hand or hands in the frame
        
        if hands:
            # extract and adjust region of interest around the detected hand
            hand = hands[0]
            x, y, w, h = hand['bbox']
            x, y = max(0, x), max(0, y)
            w, h = min(w, img.shape[1] - x), min(h, img.shape[0] - y)
            imgCrop = img[y:y + h, x:x + w]
            
            if imgCrop.size != 0:
                # preprocess the image for prediction
                imgPreprocessed = cv2.resize(imgCrop, (224, 224))
                imgPreprocessed = np.asarray(imgPreprocessed, dtype=np.float32).reshape(1, 224, 224, 3)
                imgPreprocessed = (imgPreprocessed / 127.5) - 1
                
                try:
                    prediction1 = model.predict(imgPreprocessed)
                    predicted_class_index1 = np.argmax(prediction1)

                    predicted_label1 = labels[predicted_class_index1] if predicted_class_index1 < len(labels) else "unrecognized gesture"
                
                except Exception as e:
                    print(f"prediction error: {e}")
                    predicted_label1 = "prediction error"

                # display prediction and bounding box on output image
                cv2.putText(imgOutput, predicted_label1, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(imgOutput, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            else:
                # show message if no hand is detected
                cv2.putText(imgOutput, "no hand detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # encode the frame to jpeg format for streaming
        _, jpeg = cv2.imencode('.jpg', imgOutput)
        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

# flask routes setup
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pag3', methods=['POST', 'GET'])
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pag3/template')
def render_pag3_template():
    return render_template('pag3.html')

# release resources when the app finishes
if __name__ == "__main__":
    try:
        app.run(debug=True)
    finally:
        cap.release()  # release camera to avoid resource lock