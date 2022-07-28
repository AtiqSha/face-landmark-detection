import argparse
import dlib
import cv2
import numpy as np
# import parser

# parser = argparse.ArgumentParser()


import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

files = filedialog.askopenfilenames()
print(files[0])
# image_path = r'image_pred\art-hauntington-jzY0KRJopEI-unsplash.jpg'
image_path = files[0]
predictor_path = r'shape_predictor_68_face_landmarks.dat'

print(f'predicting {files[0]}.....')
JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))

predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector(gray, 1)
for result in faces:
    x = result.left()
    y = result.top()
    x1 = result.right()
    y1 = result.bottom()
    cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
try: 
    dlib_rect = dlib.rectangle(int(x), int(y), int(x1), int(y1))
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img, dlib_rect).parts()])
    landmarks_display = landmarks[RIGHT_EYE_POINTS + LEFT_EYE_POINTS + NOSE_POINTS + JAWLINE_POINTS]
    for idx, point in enumerate(landmarks_display):
        pos = (point[0, 0], point[0, 1])
        cv2.circle(img, pos, 7, color=(0, 255, 255), thickness=-1)


    resized = cv2.resize(img, (800, 1000))

    cv2.imshow("Landmarks found", resized)
    cv2.waitKey(0)
except:
    print('Landmark not detected. Please make sure face are on the correct position (front)')


