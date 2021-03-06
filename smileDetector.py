import cv2

from random import randrange

image_file = "data/elon_test.jpg"

face_detector_file = "data/haarcascade_frontalface_default.xml"
smile_detector_file = "data/haarcascade_smile.xml"

# img = cv2.imread(image_file)

face_cascade = cv2.CascadeClassifier(face_detector_file)
smile_cascade = cv2.CascadeClassifier(smile_detector_file)

webcam = cv2.VideoCapture(0)

while True:
    successful_read, frame = webcam.read()

    if not successful_read:
        break

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_detector = face_cascade.detectMultiScale(grayscaled_img)
    smile_detector = smile_cascade.detectMultiScale(grayscaled_img)

    for (x, y, w, h) in face_detector:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for (x, y, w, h) in smile_detector:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    cv2.imshow("Smile Detection", frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

webcam.release()
cv2.destroyAllWindows()
