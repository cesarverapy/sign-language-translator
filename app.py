import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from keras.models import load_model
from flask import Flask, render_template, Response

# Cargar el modelo previamente entrenado
model_path = "C:\\Users\\59598\\Desktop\\Simeio\\Model\\keras_model.keras"
model = load_model(model_path, compile=False)

# Etiquetas para la clasificación
labels_path = "C:\\Users\\59598\\Desktop\\Simeio\\Model\\labels.txt"
with open(labels_path, "r") as f:
    labels = f.readlines()

# Quitar caracteres de espacio adicionales de las etiquetas
labels = [label.strip() for label in labels]

# Imprimir el número de etiquetas y las etiquetas para depuración
print("Número de etiquetas:", len(labels))
print("Etiquetas:", labels)

# Inicializar el detector de manos
detector = HandDetector(maxHands=1)

# Inicializar la aplicación Flask
app = Flask(__name__)

# Función para obtener el frame
def get_frame():
    # Capturar video desde la cámara
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        if not success or img is None:  # Verificar si el frame se capturó correctamente
            continue  # Saltar la iteración actual si el frame es None

        imgOutput = img.copy()
        
        # Encontrar manos en el frame
        hands, img = detector.findHands(img)
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            # Asegurarse de que el cuadro delimitador esté dentro de los límites de la imagen
            x = max(0, x)
            y = max(0, y)
            w = min(w, img.shape[1] - x)
            h = min(h, img.shape[0] - y)
            
            # Recortar la región de interés
            imgCrop = img[y:y + h, x:x + w]
            
            if imgCrop.size != 0:  # Verificar si la imagen recortada no está vacía
                # Preprocesar la imagen (cambiar tamaño, normalizar, etc.)
                imgPreprocessed = cv2.resize(imgCrop, (224, 224))
                imgPreprocessed = np.asarray(imgPreprocessed, dtype=np.float32).reshape(1, 224, 224, 3)
                imgPreprocessed = (imgPreprocessed / 127.5) - 1
                
                # Realizar inferencia con el modelo cargado
                prediction1 = model.predict(imgPreprocessed)
                predicted_class_index1 = np.argmax(prediction1)

                # Imprimir el índice predicho para depuración
                print("Índice predicho:", predicted_class_index1)

                if predicted_class_index1 >= len(labels):
                    print("Error: Índice predicho fuera de rango:", predicted_class_index1)
                    predicted_label1 = "Unknown"
                else:
                    predicted_label1 = labels[predicted_class_index1]

                # Visualizar la predicción del modelo
                cv2.putText(imgOutput, predicted_label1.strip(), (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(imgOutput, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Convertir la imagen al formato JPEG
        _, jpeg = cv2.imencode('.jpg', imgOutput)
        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pag3', methods=['POST', 'GET'])
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pag3/template')
def render_pag3_template():
    return render_template('pag3.html')

if __name__ == "__main__":
    app.run(debug=True)
