import cv2
from random import randrange

# img_file = "data/cars_and_pedestrians.jpg"
# video_file = "data/tesla_dashcam.mp4"
video_file = "data/vehicle_sample480p.mp4"

car_classifier_file = "data/car_detector.xml"
pedestrian_classifier_file = "data/haarcascade_fullbody.xml"

# img = cv2.imread(img_file)
video = cv2.VideoCapture(video_file)

car_classifier = cv2.CascadeClassifier(car_classifier_file)
pedestrian_classifier = cv2.CascadeClassifier(pedestrian_classifier_file)

while True:
    (read_success, frame) = video.read()

    if read_success:
        grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    car_detector = car_classifier.detectMultiScale(grayscaled_img)
    pedestrian_detector = pedestrian_classifier.detectMultiScale(grayscaled_img)

    for (x, y, w, h) in car_detector:
        # Drawing two rectangles over cars just for visual purposes - no functional operation
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x+1, y+1), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Vehicle", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for (x, y, w, h) in pedestrian_detector:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)
        cv2.putText(frame, "Pedestrian", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.imshow("Vehicle and Pedestrian Detection", frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

video.release()
cv2.destroyAllWindows()
