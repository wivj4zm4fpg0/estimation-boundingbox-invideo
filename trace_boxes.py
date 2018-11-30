import math

import numpy as np

from opt import parse_opts

args = parse_opts()


class TraceBoxesDatabase:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.trace_boxes = []

    def add_box(self, box: tuple, image: np.ndarray):
        self.trace_boxes.append(TraceBox(box, image))

    def get_dict(self, box: tuple, image: np.ndarray) -> tuple:
        near_dict = {}
        iou_dict = {}
        for i, trace_box in enumerate(self.trace_boxes):
            # if not trace_box.is_contain_position(box):
            #     continue
            near_dict[i] = math.sqrt((box[0] - trace_box.current_top) ** 2 + (
                    box[1] - trace_box.current_left) ** 2)
            iou_dict[i] = self.trace_boxes[i].iou(box, image)
        near_index = \
            [k for k, v in near_dict.items() if v == min(near_dict.values())][0]
        iou_index = \
            [k for k, v in iou_dict.items() if v == max(iou_dict.values())][0]
        return near_index, iou_index

    def all_update(self, box_list: list):
        near_list = []
        iou_list = []
        for box in box_list:
            near, iou = self.get_dict(box[0], box[1])
            near_list.append(near)
            iou_list.append(iou)
        for i in range(len(box_list)):
            self.trace_boxes[iou_list[i]].update(box_list[i][0], box_list[i][1])

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

    def print_boxes(self):
        for trace_box in self.trace_boxes:
            option = '-vf crop={}:{}:{}:{}'.format(
                trace_box.right - trace_box.left,
                trace_box.bottom - trace_box.top,
                trace_box.left,
                trace_box.top
            )
            yield option


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
        # print('iou = {}'.format(area / all_area))
        return area / all_area
