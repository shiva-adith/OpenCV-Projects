import glob
import os
import argparse
import sys
import time
import xml.etree.ElementTree as ET

path = "./yolov4/darknet/traffic_data/traffic_labels/"
images_folder = 'traffic_images'


# Rename xml files
def rename_file():
    # path = getattr(parsed_args, 'input')
    if not os.path.isdir(path):
        print("The specified path does not exist!")
        sys.exit()

    for i, file in enumerate(glob.iglob(path + "*.xml")):
        # os.remove(file)
        if not os.path.exists(file):
            os.rename(file, os.path.join(path, f"{str(i)}.xml"))


# Edit filename within XML tags in the file.
def edit_content():
    # path = getattr(parsed_args, 'input')
    if not os.path.isdir(path):
        print("The specified path does not exist!")
        sys.exit()

    start_time = time.process_time()
    for i, file in enumerate(glob.iglob(path + "*.xml")):
        tree = ET.parse(file)
        root = tree.getroot()

        name = root.find('filename')
        name.text = f"{str(i)}.jpg"

        folder = root.find('folder')
        folder.text = f"{images_folder}"

        tree.write(path+f"{str(i)}.xml")

    print(f"Conversion took: {round(time.process_time() - start_time, 3)} seconds")


# def validate_path(path):
#     if not os.path.exists(path):
#         raise argparse.ArgumentTypeError(f"{path} does not exit")
#     return path

# FUNCTION_MAP = {'r': rename_file(),
#                 'e': edit_content()}


my_parser = argparse.ArgumentParser(prog="xml_editor", description="Rename or Edit Contents of an XML file")

my_parser.add_argument('--rename',
                       dest='action',
                       action='store_const',
                       const=rename_file(),
                       help="specify the function to use")

my_parser.add_argument('--edit',
                       dest='action',
                       action='store_const',
                       const=edit_content(),
                       help="specify the function to use")

# my_parser.add_argument('-i',
#                        '--input',
#                        required=True,
#                        help='path to files')

parsed_args = my_parser.parse_args()

# if parsed_args.action is None:
#     my_parser.parse_args(['-h'])
# parsed_args.action(parsed_args)

# func = FUNCTION_MAP[args.command]()
