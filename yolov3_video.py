# import time

import cv2
import numpy as np
from PIL import Image

from model import YOLO
from opt import parse_opts
from trace_boxes import TraceBoxesDatabase

args = parse_opts()

video_path = args.input

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Video '{}' does not exist.".format(video_path))
    exit(1)

trace_boxes_database = TraceBoxesDatabase(cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                                          cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

yolo = YOLO(
    model_path=args.model,
    anchors_path=args.anchors,
    model_image_size=(args.size, args.size),
    score=args.conf_threshold,
    iou=args.nms_threshold,
    trace_boxes_database=trace_boxes_database
)

i = 0
while True:
    return_value, frame = cap.read()
    if not return_value:
        trace_boxes_database.print_boxes(args.input)
        cv2.waitKey(0)
        break
    image = Image.fromarray(frame)

    # Run detection
    image, return_boxes = yolo.detect_image(image)

    result = np.asarray(image)
    if len(trace_boxes_database.trace_boxes) == 0:
        for return_box in return_boxes:
            trace_boxes_database.add_box_all(return_box[0])
    else:
        for return_box in return_boxes:
            trace_boxes_database.add_box(return_box[0])
        trace_boxes_database.all_update(return_boxes)
    for trace_box in trace_boxes_database.trace_boxes:
        cv2.rectangle(result, (trace_box.left, trace_box.top),
                      (trace_box.right, trace_box.bottom), (255, 255, 255), 1)

    cv2.imshow("result", result)
    i += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

yolo.close_session()
