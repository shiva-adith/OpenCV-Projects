import cv2

# from random import randrange

image_file = "data/elon_test.jpg"
video_file = "data/smiles.mp4"

face_detector_file = "data/haarcascade_frontalface_default.xml"
smile_detector_file = "data/haarcascade_smile.xml"

# img = cv2.imread(image_file)

face_detector = cv2.CascadeClassifier(face_detector_file)
smile_detector = cv2.CascadeClassifier(smile_detector_file)

video = cv2.VideoCapture(video_file)

while True:
    successful_read, frame = video.read()

    if not successful_read:
        break

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(grayscaled_img)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Logic: instead of detecting smiles across the entire frame, we crop the detected face/faces out from the frame
        # and then detect smiles only on those cropped/sliced sections.
        # face refers to the cropped section which contains all the faces that were detected.
        face = frame[y:y+h, x:x+w]

        face_grayscaled = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Scale Factor determines how blurriness should be added to the frame. A blurry image reduces the chances of non
        # facial features being incorrectly picked up by the detector, eg. window frame, curtain waves etc.
        # minNeighbors determines how many detections are present in close proximity. Higher the number, greater the
        # accuracy since a greater number of detections over a particular area implies a greater chances of true +ves.
        smiles = smile_detector.detectMultiScale(face_grayscaled, scaleFactor=1.7, minNeighbors=20)

        # displaying the text "Smiling" when detected instead of just a rectangle over the smiles.
        if len(smiles) > 0:
            cv2.rectangle(frame, (x, y+h), (x+100, y+h+40), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, "Smiling", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # for (x_smile, y_smile, w_smile, h_smile) in smiles:
        #     cv2.rectangle(face, (x_smile, y_smile), (x_smile + w_smile, y_smile + h_smile), (0, 255, 255), 2)

    cv2.imshow("Smile Detection", frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

video.release()
cv2.destroyAllWindows()
