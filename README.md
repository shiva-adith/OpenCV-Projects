<h1 style="text-align: center;"><u>OpenCV Projects</u></h1>

### Prerequisites

***
`pip install opencv`

If you run into installation issues with the above, try:

`pip install opencv-headless`

### Requirements 

---
For the following projects, the Haar Cascade Frontal Face-**Default**, Fullbody and Smile xml files are required.

Download the Face and Body Cascades from [Haar Cascades-GitHub Repo][haar_cascade]

Download the Haar Cascade for Car detection from [vehicle_detection_haarcascade by Andrews Sobral
][vehicle_cascade]


## 😃 Smile Detection 😁

---

This app makes uses of Computer vision technology to identify faces and detect smiles. Faces and smiles are marked by drawing rectangles around the detected areas. This technology finds use in analysing moods/emotions with AI.

### Results

---

* Running the Smile Detection app on ***[this][smile_vid]*** video obtained from BuzzFeedVideo's YouTube channel.

![smile_detection_results][smile_ref]

- As you can see, the app detects the faces as well as smiles and draws rectangles around the area. Owing to the high accuracy of Haar cascade face classifiers, we are able to achieve smooth detection even with rapid movement of the head. However, the Smile classifier requires parameter tuning and larger datasets to learn from, in order to achieve better results (certain facial features are mistaken as smiles).

---

## 🚗 Vehicle and Pedestrian Detection 🏃‍♂️🏃‍♀️👨‍🦯

---
This app makes use of Computer vision technology to identify and mark vehicles and pedestrians in the frame. This technology is widely used in the field of Self-driving vehicles such as cars, drones etc.

### Results

---

* Running the Vehicle and Pedestrian Detection app on ***[this][vehicle_vid]*** video obtained from RoyalJordanian's YouTube channel.

![vehicle_detection_results][v&p_ref]

- The pedestrian detecting algorithm faces issues with large groups of people. However, it is able to accurately detect and mark individuals and small but distinct groups with ease. The vehicle detection works much more efficiently (fewer false positives), but is certainly not 100 % accurate. Better tuning of the algorithm and parameters of the classifiers will produce more accurate results.





<!-- References and Links -->
[haar_cascade]: https://github.com/opencv/opencv/tree/master/data/haarcascades
[vehicle_cascade]: https://github.com/andrewssobral/vehicle_detection_haarcascades
[smile_vid]: https://www.youtube.com/watch?v=f8OmSWxF6h8&t=1s
[vehicle_vid]: https://www.youtube.com/watch?v=WriuvU1rXkc&t=22s
[smile_ref]: results/smile_detection.gif
[v&p_ref]: results/vehicle_and_pedestrian_detection.gif