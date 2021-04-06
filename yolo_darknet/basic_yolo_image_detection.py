import cv2
import numpy as np

image_file = "../data/yolo_images/testing/scene2.jpg"
configuration_file = "yolov3.cfg"
weights_file = "yolov3.weights"
img = cv2.imread(image_file)
img_height = img.shape[0]
img_width = img.shape[1]

# the image is converted into a blob - a format of images stored in database
# recommended values(for yolo) for scalefactor, width and height are 0.003922(=1/255), 320, 320.
# acceptable height and width values are: 320x320, 416x416, 609x609
# opencv uses the BGR format for colours, hence the R and B need to be swapped.
img_blob = cv2.dnn.blobFromImage(img, scalefactor=0.003922, size=(416, 416), swapRB=True, crop=False)

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

# setting five different colours used for drawing rectangles around detected objects.
class_colours = ['255,0,0', '0,255,0', '0,0,255', '255,255,0', '0,255,255']

# converting the strings into integers and turning the list into a numpy array
class_colours = [np.array(colour.split(',')).astype('int') for colour in class_colours]

class_colours = np.array(class_colours)

# Each colour is repeated 16 times. 16x5 gives 80 elements, which is the number of different classes provided above.
class_colours = np.tile(class_colours, (16, 1))
# test: print(len(class_colours))

# Loading a pretrained model
model = cv2.dnn.readNetFromDarknet(configuration_file, weights_file)

# getting all the layers
model_layers = model.getLayerNames()
# test: print(model_layers)

# identifying the output layer
output_layer = [model_layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()]

model.setInput(img_blob)
# passes the blob through the network till the output layer, and returns the outputs for each layer.
# each layer produces multiple detections.
object_detection = model.forward(output_layer)

for layer in object_detection:
    # looping through all the detected objects in each layer
    for detection in layer:
        # detection[1 to 4] provide coordinates for the rectangle around the detected object(s)
        # detection[5:] provides the scores for all objects within the box from above.

        # returns scores for every detected class
        class_scores = detection[5:]

        # determines the class with the highest score
        predicted_class_id = np.argmax(class_scores)

        # obtain the confidence % for the classes detected.
        prediction_confidence = class_scores[predicted_class_id]

        # use predictions to draw the bounding box only if confidence is greater than 20%
        if prediction_confidence > 0.20:
            # obtain the label of the predicted class
            predicted_class_label = class_labels[int(predicted_class_id)]

            # obtain rectangle coordinates and resize to actual image size. (img was resized during blob conversion)
            # detection[:4] provides four coordinates and each of them are upscaled back to img size.
            bounding_box = detection[:4] * np.array([img_width, img_height, img_width, img_height])
            # test: print(type(bounding_box))

            # converting numpy array of x,y, width and height to int
            (x, y, w, h) = bounding_box.astype('int')

            # obtain coordinates for the rectangle
            start_x = int(x - (w/2))
            start_y = int(y - (h/2))
            end_x = start_x + w
            end_y = start_y + h

            # obtain a colour for the box
            box_colour = class_colours[int(predicted_class_id)]

            # convert numpy array of colours into list
            box_colour = [int(colour) for colour in box_colour]

            # print prediction label and corresponding confidence %
            print(f"Prediction: {predicted_class_label}, {prediction_confidence*100:.2f}")

            # draw rectangle and display text
            cv2.rectangle(img, (start_x, start_y), (end_x, end_y), box_colour, 1)
            cv2.putText(img, predicted_class_label, (start_x, start_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_colour, 1)

cv2.imshow("Object Detection using YOLO", img)
cv2.waitKey(0)
cv2.destroyAllWindows()






