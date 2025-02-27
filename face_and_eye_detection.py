# Face and Eye Detection from an Image

import cv2
import numpy as np
from matplotlib import pyplot as plt

class FaceEyeDetectorImage:
    def __init__(self, face_cascade_path, eye_cascade_path):
        self.face_classifier = cv2.CascadeClassifier(face_cascade_path)
        self.eye_classifier = cv2.CascadeClassifier(eye_cascade_path)
    
    def imshow(self, title, image, size=10, save_path="output.jpg"):
        w, h = image.shape[:2]
        aspect_ratio = w / h
        plt.figure(figsize=(size * aspect_ratio, size))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.savefig(save_path)  # Save the image instead of displaying it
        print(f"Image saved as {save_path}")

    
    def detect_faces_eyes(self, image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))
        
        if len(faces) == 0:
            print("No faces found")
        
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 255), 8)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = image[y:y + h, x:x + w]
            eyes = self.eye_classifier.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 4)
        
        self.imshow('Face & Eye Detection', image)

# Usage Example
face_eye_detector = FaceEyeDetectorImage('haarcascade_frontalface_default.xml', 'haarcascade_eye.xml')
face_eye_detector.detect_faces_eyes('face5.jpg')