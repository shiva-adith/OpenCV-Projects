import cv2
import glob
import os
import time

path = "./yolov4/darknet/traffic_data/traffic_images/archive/images/"

start_time = time.process_time()
for file in glob.iglob(path + '*.png'):

    png_img = cv2.imread(file)

    cv2.imwrite(f'{file[:-4]}.jpg', png_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    try:
        os.remove(file)
    except FileNotFoundError:
        print("There was an error in deleting the file:", file)

for i, file in enumerate(glob.iglob(path + '*.jpg')):
    os.rename(file, os.path.join(path, f"{str(i)}.jpg"))

print(f"Conversion took: {round(time.process_time()-start_time, 3)} seconds")

# Conversion took: 13.703 seconds
