# Face and Eye Detection from Webcam

import cv2

class FaceEyeDetectorWebcam:
    def __init__(self, face_cascade_path, eye_cascade_path):
        self.face_classifier = cv2.CascadeClassifier(face_cascade_path)
        self.eye_classifier = cv2.CascadeClassifier(eye_cascade_path)
    
    def face_detector(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

        if len(faces) == 0:
            return img

        for (x, y, w, h) in faces:
            # Expand face bounding box slightly
            x, w, y, h = max(0, x - 20), w + 40, max(0, y - 20), h + 40
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            # Detect eyes only within the detected face region
            eyes = self.eye_classifier.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=10, minSize=(30, 30))
            
            for (ex, ey, ew, eh) in eyes:
                if ey < h // 2:  
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

        return img

    def run_webcam(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            cv2.imshow('Face & Eye Detector', self.face_detector(frame))
            if cv2.waitKey(1) == 13:  # Enter key
                break
        
        cap.release()
        cv2.destroyAllWindows()

# Usage Example
face_eye_detector_webcam = FaceEyeDetectorWebcam('haarcascade_frontalface_default.xml', 'haarcascade_eye.xml')
face_eye_detector_webcam.run_webcam()