# Importar las bibliotecas necesarias
import cv2  # OpenCV para procesamiento de imágenes
from cvzone.HandTrackingModule import HandDetector  # Detector de manos
import numpy as np  # Manipulación de matrices
import math  # Funciones matemáticas
import time  # Funciones relacionadas con el tiempo
import os  # Interacción con el sistema operativo

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Inicializar el detector de manos con la capacidad de detectar una sola mano
detector = HandDetector(maxHands=1)

# Definir un valor de desplazamiento para el recorte de imágenes
offset = 20

# Definir el tamaño de la imagen para el recorte
imgSize = 300

# Inicializar un contador para contar las imágenes guardadas
counter = 0

# Establecer la carpeta base donde se guardarán las imágenes
base_folder = r"C:\Users\59598\Desktop\Simeio\Data\train"

# Definir las clases disponibles para etiquetar las imágenes
classes = ["class 1", "class 2", "class 3", "class 4", "class 5", "class 6", "class 7"]

# Inicializar la clase actual en 0
current_class = 0   

# Función para asegurarse de que el directorio existe
def ensure_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

# Bucle principal para procesar los fotogramas del video
while True:
    # Leer un fotograma de la cámara
    success, img = cap.read()
    if not success:
        continue

    # Utilizar el detector de manos para encontrar las manos en el fotograma
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Crear una imagen blanca del tamaño especificado
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Calcular las coordenadas del área de recorte alrededor de la mano
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        # Recortar la región de interés de la imagen original
        imgCrop = img[y1:y2, x1:x2]
        imgCropShape = imgCrop.shape

        # Proceder solo si el área de recorte no está vacía
        if imgCropShape[0] > 0 and imgCropShape[1] > 0:
            aspectRatio = h / w

            # Redimensionar la imagen de manera acorde manteniendo la relación de aspecto original
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = (imgSize - wCal) // 2
                imgWhite[:, wGap: wGap + wCal] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = (imgSize - hCal) // 2
                imgWhite[hGap: hGap + hCal, :] = imgResize

            # Mostrar las imágenes recortadas y redimensionadas
            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

    # Mostrar la imagen original
    cv2.imshow('Image', img)

    # Esperar una tecla durante 1 milisegundo
    key = cv2.waitKey(1)
    
    # Si se presiona la tecla "s", guardar la imagen recortada en la carpeta correspondiente a la clase actual
    if key == ord("s"):
        counter += 1
        folder = os.path.join(base_folder, classes[current_class])
        ensure_dir(folder)
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Saved {counter} images to {classes[current_class]}")

    # Si se presiona la tecla "c", cambiar a la siguiente clase disponible
    if key == ord("c"):
        current_class = (current_class + 1) % len(classes)
        print(f"Switched to {classes[current_class]}")

    # Si se presiona la tecla "q", salir del bucle y terminar el programa
    if key == ord("q"):
        break

# Liberar los recursos de la cámara y cerrar todas las ventanas de OpenCV
cap.release()
cv2.destroyAllWindows()
