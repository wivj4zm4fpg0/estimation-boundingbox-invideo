import os
import time
from model import YOLO
from PIL import Image

yolo = YOLO(
    model_path="model_data/yolov3.h5",
    anchors_path="model_data/yolov3_anchors.txt",
    model_image_size=(320, 320))

while True:
    file_name = input("Enter an image name or type 'quit': ")

    if file_name == 'quit':
        yolo.close_session()
        break

    if not os.path.exists(file_name):
        print("Error: '{}' does not exist.".format(file_name))
        continue

    image = Image.open(file_name)

    # Run detection
    start = time.time()
    r_image = yolo.detect_image(image)
    end = time.time()
    print("Detection Time: {:.3f} sec".format(end - start))

    r_image.show()
