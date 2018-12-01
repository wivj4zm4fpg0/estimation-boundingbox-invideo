import cv2
import numpy as np
from PIL import Image

from model import YOLO
from opt import parse_opts
from trace_boxes import TraceBoxesDatabase

args = parse_opts()

video_path = args.input

cap = cv2.VideoCapture(video_path)
assert cap.isOpened()

size = (
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

out = None
if args.save_video:
    codecs = 'H264'
    fourcc = cv2.VideoWriter_fourcc(*codecs)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter('out.mp4', fourcc, fps, size)

trace_boxes_database = TraceBoxesDatabase(size[0], size[1])

yolo = YOLO(
    model_path=args.model,
    anchors_path=args.anchors,
    model_image_size=(args.size, args.size),
    score=args.conf_threshold,
    iou=args.nms_threshold,
    trace_boxes_database=trace_boxes_database
)

result = None

while True:

    return_value, frame = cap.read()

    if not return_value:
        trace_boxes_database.padding_boxes()
        for trace_box in trace_boxes_database.trace_boxes:
            cv2.rectangle(result, (trace_box.left, trace_box.top),
                          (trace_box.right, trace_box.bottom), (255, 255, 0), 2)
        cv2.imshow('result', result)
        if args.save_video:
            out.write(result)
        trace_boxes_database.print_boxes(args.input, args.output)
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

        for i in range(len(trace_box.point_x_list)):
            cv2.circle(result, (trace_box.point_x_list[i], trace_box.point_y_list[i]),
                       3, (100, 100, 255), thickness=1, lineType=cv2.LINE_AA)
            if i != 0:
                cv2.line(
                    result,
                    (trace_box.point_x_list[i], trace_box.point_y_list[i]),
                    (trace_box.point_x_list[i - 1], trace_box.point_y_list[i - 1]),
                    (100, 100, 255),
                    lineType=cv2.LINE_AA
                )

    cv2.imshow('result', result)
    if args.save_video:
        out.write(result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

yolo.close_session()
