import numpy as np

from compare_hist_test import compare_images
from yolov3_video import args


class TraceBoxesDatabase:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.trace_boxes = []

    def add_box(self, box: tuple, image: np.ndarray):
        for trace_box in self.trace_boxes:
            if trace_box.update(box, image):
                return
        self.trace_boxes.append(TraceBox(box, image))

    def padding_boxes(self):
        for trace_box in self.trace_boxes:
            trace_box.top = int(
                max(0, trace_box.top - trace_box.top * args.padding_size))
            trace_box.left = int(
                max(0, trace_box.left - trace_box.left * args.padding_size))
            trace_box.bottom = int(
                min(self.height, trace_box.bottom * (1 + args.padding_size)))
            trace_box.right = int(
                min(self.width, trace_box.right * (1 + args.padding_size)))


class TraceBox:
    def __init__(self, box: tuple, image: np.ndarray):
        self.top = box[0]
        self.left = box[1]
        self.bottom = box[2]
        self.right = box[3]
        self.current_image = image

    def update(self, box: tuple, image: np.ndarray) -> bool:
        if compare_images(self.current_image, image) > args.compare_hist_threshold:
            self.top = min(self.top, box[0])
            self.left = min(self.left, box[1])
            self.bottom = max(self.bottom, box[2])
            self.right = max(self.right, box[3])
            self.current_image = image

            return True

        else:
            return False
