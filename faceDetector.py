import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# img = cv2.imread('data/crowd.jpg')
webcam = cv2.VideoCapture(0)

while True:
    success, frame = webcam.read()

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 5)

    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1)

    # break the while loop if q or Q is pressed
    if key == 81 or key == 113:
        break

# Release webcam
webcam.release()
