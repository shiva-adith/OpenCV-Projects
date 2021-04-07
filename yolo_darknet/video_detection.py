import cv2
import numpy as np

# from image_detection import class_labels

# video_file = "../data/yolo_images/testing/video_sample.mp4"
video_file = "../data/vehicle_sample480p.mp4"
configuration_file = "yolov3.cfg"
weights_file = "yolov3.weights"

# set of 80 class labels
class_labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball",
                "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket",
                "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair",
                "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator",
                "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"]

np.random.seed(42)
class_colours = np.random.randint(0, 255, size=(len(class_labels), 3), dtype='uint8')

model = cv2.dnn.readNetFromDarknet(configuration_file, weights_file)

model_layers = model.getLayerNames()

output_layer = [model_layers[layer[0] - 1] for layer in model.getUnconnectedOutLayers()]

video = cv2.VideoCapture(video_file)

(frame_width, frame_height) = (None, None)

while True:
    successful_read, frame = video.read()

    if not successful_read:
        break

    if frame_width is None or frame_height is None:
        # obtain height (.shape[0]) and width (.shape[1])
        (frame_height, frame_width) = frame.shape[:2]

    frame_blob = cv2.dnn.blobFromImage(frame, scalefactor=0.003922, size=(416, 416), swapRB=True, crop=False)

    model.setInput(frame_blob)
    predictions = model.forward(output_layer)

    class_id_list = []
    bboxes_list = []
    confidence_list = []

    # loop over each of the layer outputs
    for layer in predictions:
        # loop over each of the detections in every layer output
        for detection in layer:
            class_scores = detection[5:]
            predicted_class_id = np.argmax(class_scores)
            prediction_confidence = class_scores[predicted_class_id]

            if prediction_confidence > 0.2:
                predicted_class_label = class_labels[int(predicted_class_id)]

                bboxes = detection[:4] * np.array([frame_width, frame_height, frame_width, frame_height])

                (centre_x, centre_y, w, h) = bboxes.astype('int')

                start_x = int(centre_x - (w / 2))
                start_y = int(centre_y - (h / 2))

                class_id_list.append(predicted_class_id)
                bboxes_list.append([start_x, start_y, int(w), int(h)])
                confidence_list.append(float(prediction_confidence))

    max_value_ids = cv2.dnn.NMSBoxes(bboxes_list, confidence_list, score_threshold=0.5, nms_threshold=0.4)

    if len(max_value_ids) > 0:
        # loop over the indexes that are to be kept (ie. the max value ones)
        for id in max_value_ids.flatten():
            (x, y) = (bboxes_list[id][0], bboxes_list[id][1])
            (w, h) = (bboxes_list[id][2], bboxes_list[id][3])

            bbox_colour = class_colours[class_id_list[id]]

            bbox_colour = [int(colour) for colour in bbox_colour]

            predicted_class_label = class_labels[int(class_id_list[id])]
            prediction_confidence = confidence_list[id]

            # print prediction label and corresponding confidence %
            # print(f"Prediction: {predicted_class_label}, {prediction_confidence * 100:.2f}")

            # draw rectangle and display text withing a filled rectangle (according to text size)
            label_size = cv2.getTextSize(predicted_class_label, cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.4, thickness=1)
            cv2.rectangle(frame, (x, y), (x + label_size[0][0], y - int(label_size[0][1]) - 4), bbox_colour, cv2.FILLED)
            cv2.rectangle(frame, (x, y), (x + w, y + h), bbox_colour, 1)
            cv2.putText(frame, predicted_class_label.capitalize(), (x, y - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 0, 0),
                        1)

    cv2.imshow("Video Detection using YOLO", frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

video.release()
cv2.destroyAllWindows()
