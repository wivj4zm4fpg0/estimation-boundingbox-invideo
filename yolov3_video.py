import time

import cv2
import numpy as np
from PIL import Image

from model import YOLO
from opt import parse_opts

args = parse_opts()

yolo = YOLO(
    model_path=args.model,
    anchors_path=args.anchors,
    model_image_size=(args.size, args.size),
    score=args.conf_threshold,
    iou=args.nms_threshold
)

video_path = args.input

vid = cv2.VideoCapture(video_path)
if not vid.isOpened():
    print("Error: Video '{}' does not exist.".format(video_path))
    exit(1)

while True:
    return_value, frame = vid.read()
    if not return_value:
        break
    image = Image.fromarray(frame)

    # Run detection
    start = time.time()
    image = yolo.detect_image(image)
    end = time.time()
    print("Detection Time: {:.3f} sec".format(end - start))

    result = np.asarray(image)
    cv2.imshow("result", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

yolo.close_session()
