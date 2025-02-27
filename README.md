# Face and Eye Detection using OpenCV  

This project detects **faces and eyes** in images and real-time webcam feed using OpenCV's Haar cascades.  

## Features  
- Face detection in images  
- Eye detection in detected faces  
- Real-time face and eye detection using webcam  

## Installation  

1. Clone this repository:  
   ```sh
   git clone https://github.com/MusapYildiz/Face-Eye-Detection.git
   cd Face-Eye-Detection
2. Install required dependencies:
   ```sh
   pip install -r requirements.txt
   
3. Download the required Haar cascade XML files from OpenCV’s GitHub:
   ```sh
   wget https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
   
   wget https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_eye.xml

## Usage

1. Face & Eye Detection in an Image

   Run the following command:
   Make sure to replace 'face5.jpg' with your image file.
   ```sh
   python face_eye_image.py
   
2. Real-time Face & Eye Detection using Webcam
   ```sh
   python face_eye_webcam.py


## Author
	https://github.com/MusapYildiz
