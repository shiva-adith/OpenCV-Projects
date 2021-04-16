import cv2
import glob
import os
import time
import argparse

# path = "./yolov4/darknet/traffic_data/traffic_images/archive/images/"


def convert_file(filepath):
    start_time = time.process_time()
    for file in glob.iglob(filepath + '*.png'):

        png_img = cv2.imread(file)

        cv2.imwrite(f'{file[:-4]}.jpg', png_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        try:
            os.remove(file)
        except FileNotFoundError:
            print("There was an error in deleting the file:", file)

    print(f"Conversion took: {round(time.process_time() - start_time, 3)} seconds")


def rename_file(filepath):
    start_time = time.process_time()

    for i, file in enumerate(glob.iglob(filepath + '*.jpg')):
        os.rename(file, os.path.join(filepath, f"{str(i)}.jpg"))

    print(f"Conversion took: {round(time.process_time() - start_time, 3)} seconds")


# Conversion took: 13.703 seconds

parser = argparse.ArgumentParser(prog="image_editor", description="Rename or Convert Images")

parser.add_argument('--rename',
                    type=rename_file,
                    dest='PATH',
                    help="specify the path")

parser.add_argument('--convert',
                    action='store',
                    type=convert_file,
                    dest='PATH',
                    help="specify the path")

parsed_args = parser.parse_args()
