import numpy as np
import math

from compare_hist_test import compare_images
from opt import parse_opts

args = parse_opts()


class TraceBoxesDatabase:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.trace_boxes = []

    def add_box(self, box: tuple, image: np.ndarray):
        # if image.shape[1] > self.width / 3 or image.shape[0] > self.height / 2:
        #     return
        near_dict = {}
        iou_dict = {}
        for i, trace_box in enumerate(self.trace_boxes):
            if not trace_box.is_contain_position(box):
                continue
            near_dict[i] = math.sqrt((box[0] - trace_box.current_top) ** 2 + (
                    box[1] - trace_box.current_left) ** 2)
            iou_dict[i] = self.trace_boxes[i].iou(box, image)
        if len(near_dict) != 0:
            near_index = \
                [k for k, v in near_dict.items() if v == min(near_dict.values())][0]
            iou_index = \
                [k for k, v in iou_dict.items() if v == max(iou_dict.values())][0]
            if compare_images(
                    image,
                    self.trace_boxes[near_index].current_image
            ) < args.compare_hist_threshold:
                return
            if self.trace_boxes[near_index].iou(box, image) < args.iou_threshold:
                return
                # self.trace_boxes.append(TraceBox(box, image))
                # return
            self.trace_boxes[iou_index].update(box, image)
            # self.trace_boxes[near_index].update(box, image)
        else:
            # if len(self.trace_boxes) == 3:
            #     return
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

        self.current_top = box[0]
        self.current_left = box[1]
        self.current_bottom = box[2]
        self.current_right = box[3]

        self.current_image = image

    def update(self, box: tuple, image: np.ndarray):
        self.top = min(self.top, box[0])
        self.left = min(self.left, box[1])
        self.bottom = max(self.bottom, box[2])
        self.right = max(self.right, box[3])

        self.current_top = box[0]
        self.current_left = box[1]
        self.current_bottom = box[2]
        self.current_right = box[3]

        self.current_image = image

    def is_contain_position(self, box: tuple) -> bool:
        if (self.current_top > box[0] and self.current_top > box[2]) or (
                self.current_bottom < box[0] and self.current_bottom < box[2]) or (
                self.current_left > box[1] and self.current_left > box[3]) or (
                self.current_right < box[1] and self.current_right < box[3]):
            return False
        else:
            return True

    def iou(self, box: tuple, image: np.ndarray) -> float:

        if self.current_left < box[1]:
            if self.current_right < box[3]:
                width = self.current_right - box[1]
            else:
                width = image.shape[1]
        else:
            if box[3] < self.current_right:
                width = box[3] - self.current_left
            else:
                width = self.current_right - self.current_left

        if self.current_top < box[0]:
            if self.current_bottom < box[2]:
                height = self.current_bottom - box[0]
            else:
                height = image.shape[0]
        else:
            if box[2] < self.current_bottom:
                height = box[2] - self.current_top
            else:
                height = self.current_bottom - self.current_top

        area = width * height
        all_area = (self.current_right - self.current_left) * (
                self.current_bottom - self.current_top) + \
                   (box[3] - box[1]) * (box[2] - box[0]) - area
        print('iou = {}'.format(area / all_area))
        return area / all_area
