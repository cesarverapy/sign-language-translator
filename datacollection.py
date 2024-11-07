import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# configuration settings
IMG_SIZE = 300
OFFSET = 20
BASE_FOLDER = "./Data/train"
CLASSES = ["class 1", "class 2", "class 3", "class 4", "class 5", "class 6", "class 7"]

# initialize camera and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
current_class = 0
counter = 0

def ensure_dir(path):
    """ensure that the directory exists; if not, create it."""
    os.makedirs(path, exist_ok=True)

def save_image(img, class_name):
    """save processed image to the specified class directory."""
    global counter
    folder_path = os.path.join(BASE_FOLDER, class_name)
    ensure_dir(folder_path)
    filename = f"image_{time.time():.0f}.jpg"
    cv2.imwrite(os.path.join(folder_path, filename), img)
    counter += 1
    print(f"saved {counter} images to {class_name}")

def switch_class():
    """cycle to the next class and print the updated class."""
    global current_class
    current_class = (current_class + 1) % len(CLASSES)
    print(f"Switched to {CLASSES[current_class]}")

def preprocess_image(img, bbox):
    """crop, pad, and resize the image around the hand bounding box while maintaining aspect ratio."""
    x, y, w, h = bbox
    x1, y1 = max(0, x - OFFSET), max(0, y - OFFSET)
    x2, y2 = min(img.shape[1], x + w + OFFSET), min(img.shape[0], y + h + OFFSET)
    img_crop = img[y1:y2, x1:x2]

    img_white = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
    aspect_ratio = h / w

    if aspect_ratio > 1:
        k = IMG_SIZE / h
        w_cal = math.ceil(k * w)
        img_resize = cv2.resize(img_crop, (w_cal, IMG_SIZE))
        w_gap = (IMG_SIZE - w_cal) // 2
        img_white[:, w_gap:w_gap + w_cal] = img_resize
    else:
        k = IMG_SIZE / w
        h_cal = math.ceil(k * h)
        img_resize = cv2.resize(img_crop, (IMG_SIZE, h_cal))
        h_gap = (IMG_SIZE - h_cal) // 2
        img_white[h_gap:h_gap + h_cal, :] = img_resize

    return img_white

while True:
    success, img = cap.read()
    if not success:
        continue

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        img_processed = preprocess_image(img, hand['bbox'])
        cv2.imshow('processed Image', img_processed)
    else:
        img_processed = None

    cv2.imshow('original Image', img)

    key = cv2.waitKey(1)
    if key == ord("s") and img_processed is not None:
        save_image(img_processed, CLASSES[current_class])
    elif key == ord("c"):
        switch_class()
    elif key == ord("q"):
        break

# release resources
cap.release()
cv2.destroyAllWindows()
