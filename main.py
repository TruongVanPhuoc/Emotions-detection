import os
import cv2
import keras
import numpy as np
import time
#from tensorflow.keras.models 
#import model_from_json
from PIL import Image, ImageFilter
from torchvision import transforms
from threading import Thread

def speak():
    facial_expression = ["None", "angry", "disgust", "fear", "happy", "neutral", "surprise", "sad"]
    print(facial_expression[STATUS])
    
STATUS = 0

loaded_model = keras.models.load_model('mode.h5')

print("Loaded model from disk")


detector = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

transformations = transforms.Compose([transforms.Resize(48),
                                       transforms.RandomCrop(48),
                                       transforms.ToTensor()])



facial_expression = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Surprise", "Sad"]


cap = cv2.VideoCapture(0)
print('Start up done')
color = (255, 0, 0)
thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX # định dạng font chữ

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
             minNeighbors=5, minSize=(100, 100),
             flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        STATUS = 0
    for x,y,w,h in rects:
        x_min = x
        x_max = x + w
        y_min = y
        y_max = y + h

        roi = gray[y_min:y_max, x_min:x_max]
        roi = roi/255
        roi = Image.fromarray(roi)

        roi_resized = transformations(roi)
        roi_resized = roi_resized.numpy()
        roi_resized = np.transpose(roi_resized, (1, 2, 0))
        roi_resized = roi_resized.reshape(1, 48, 48, 1)

        Y = loaded_model.predict(roi_resized)
        classes = np.argmax(Y, axis = 1)
        emotion = facial_expression[classes[0]]
        start_point = (x, y)
        end_point = (x + h, y + w)

        STATUS = classes[0] + 1
        frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness) # vẽ hình chữ nhật xung quanh mặt người
        frame = cv2.putText(frame, emotion, start_point,
                font, 1, (0, 255, 0), 1) # ghi cảm xúc lên
    cv2.imshow("Frame", frame) # chiếu bức ảnh lên


    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()