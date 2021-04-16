import glob
import os
import argparse
import sys
import time
import xml.etree.ElementTree as ET

images_folder = 'traffic_images'


# Rename xml files
def rename_file(filepath):
    # print(filepath)

    if not os.path.isdir(filepath):
        print("The specified path does not exist!")
        sys.exit()

    for i, file in enumerate(glob.iglob(filepath + "*.xml")):
        # os.remove(file)
        if os.path.exists(file):
            os.rename(file, os.path.join(filepath, f"{str(i)}.xml"))
            # print("success")
        else:
            print("Does not exist")


# Edit filename within XML tags in the file.
def edit_content(path):

    # test = args.split()
    # print(test)
    # path = args[0]
    # print(path)
    # images_folder = args[1]

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
        # folder.text = "test"

        tree.write(file)

    print(f"Conversion took: {round(time.process_time() - start_time, 3)} seconds")


my_parser = argparse.ArgumentParser(prog="xml_editor", description="Rename or Edit Contents of an XML file")

my_parser.add_argument('--rename',
                       type=rename_file,
                       dest='PATH',
                       help="specify the path")

my_parser.add_argument('--edit',
                       action='store',
                       nargs="+",
                       type=edit_content,
                       dest='PATH, FOLDER',
                       help="specify the path")

# my_parser.add_argument('-i',
#                        '--images',
#                        action='store_const',
#                        type=str,
#                        help='Images folder name')


parsed_args = my_parser.parse_args()
