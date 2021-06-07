import tensorflow as tf
import tensorflow.keras as keras
import cv2
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model = keras.models.load_model("Mask/")

cv2.__version__

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(3, 480)  # set Width
cap.set(4, 480)  # set Height
while True:
    ret, img = cap.read()

    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
    for (x, y, w, h) in faces:
        cropped = img[y:y + h, x:x + w]
        cropped = cv2.resize(cropped, (100, 100))
        cropped = cropped.reshape((1, 100, 100, 3))
        output = model.predict(cropped)
        prediction = np.argmax(output)
        if prediction == 1:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    cv2.imshow('real time detector for masks', img)

    if cv2.waitKey(1) == ord("x"):
        break

cap.release()
cv2.destroyAllWindows()