import re

from opt import parse_opts

args = parse_opts()


class TraceBoxesDatabase:
    # 動画の縦幅と横幅を引数とする
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.trace_boxes = []

    # 追跡対象のバウンディングボックスを追加
    def add_box_all(self, box: tuple):
        self.trace_boxes.append(TraceBox(box))

    # 追跡対象のバウンディングボックスを追加
    def add_box(self, box: tuple):
        for trace_box in self.trace_boxes:
            if trace_box.is_contain_position(box):
                return
        self.trace_boxes.append(TraceBox(box))

    def add_box_include(self, box: tuple):
        for trace_box in self.trace_boxes:
            if trace_box.is_include(box):
                return
        self.trace_boxes.append(TraceBox(box))

    # 追跡対象のバウンディングボックスと現フレームのバウンディングボックスのそれぞれのiou値を総当りで調べて一番近いものを出力
    def get_index(self, box: tuple) -> tuple:
        iou_dict = {}
        for i, trace_box in enumerate(self.trace_boxes):
            if not trace_box.is_contain_position_current(box):
                iou_dict[i] = 0
            else:
                iou_dict[i] = trace_box.iou(box)
        iou_index = \
            [k for k, v in iou_dict.items() if v == max(iou_dict.values())][0]
        return iou_index, iou_dict[iou_index]

    # iouが近いものを更新する
    def all_update(self, box_list: list):
        iou_list = []
        value_list = []
        for box in box_list:
            iou, value = self.get_index(box[0])
            iou_list.append(iou)
            value_list.append(value)
        for i in range(len(box_list)):
            # if value_list[i] == 0:
            #     continue
            if value_list[i] < args.iou_threshold:
                self.add_box_include(box_list[i][0])
                continue
            self.trace_boxes[iou_list[i]].update(box_list[i][0])

    # バウンディングボックスを拡張する
    def padding_boxes(self):
        for trace_box in self.trace_boxes:
            box_width = trace_box.right - trace_box.left
            box_height = trace_box.bottom - trace_box.top
            pad_width = (box_width * args.padding_size - box_width) / 2
            pad_height = (box_height * args.padding_size - box_height) / 2
            trace_box.top = int(
                max(0, trace_box.top - pad_height))
            trace_box.left = int(
                max(0, trace_box.left - pad_width))
            trace_box.bottom = int(
                min(self.height, trace_box.bottom + pad_height))
            trace_box.right = int(
                min(self.width, trace_box.right + pad_width))

    # ffmpegでクロッピングするためのコマンドを出力
    def print_boxes(self, input_name):
        for i, trace_box in enumerate(self.trace_boxes):
            output_name = '{}_{}.mp4'.format(re.sub(r'\.mp4', '', input_name), i)
            command = 'ffmpeg -y -i {} -vf crop={}:{}:{}:{} {}'.format(
                input_name,
                trace_box.right - trace_box.left,
                trace_box.bottom - trace_box.top,
                trace_box.left,
                trace_box.top,
                output_name
            )
            print(command)


class TraceBox:
    def __init__(self, box: tuple):
        self.top = box[0]
        self.left = box[1]
        self.bottom = box[2]
        self.right = box[3]

        self.current_top = box[0]
        self.current_left = box[1]
        self.current_bottom = box[2]
        self.current_right = box[3]

    # 追跡対象のバウンディングボックスを更新する
    def update(self, box: tuple):
        self.top = min(self.top, box[0])
        self.left = min(self.left, box[1])
        self.bottom = max(self.bottom, box[2])
        self.right = max(self.right, box[3])

        self.current_top = box[0]
        self.current_left = box[1]
        self.current_bottom = box[2]
        self.current_right = box[3]

    def is_contain_position(self, box: tuple) -> bool:
        if (self.top > box[0] and self.top > box[2]) or (
                self.bottom < box[0] and self.bottom < box[2]) or (
                self.left > box[1] and self.left > box[3]) or (
                self.right < box[1] and self.right < box[3]):
            return False
        else:
            return True

    def is_contain_position_current(self, box: tuple) -> bool:
        if (self.current_top > box[0] and self.current_top > box[2]) or (
                self.current_bottom < box[0] and self.current_bottom < box[2]) or (
                self.current_left > box[1] and self.current_left > box[3]) or (
                self.current_right < box[1] and self.current_right < box[3]):
            return False
        else:
            return True

    def is_include(self, box: tuple) -> bool:
        if (self.top <= box[0] and self.left <= box[1] and self.bottom > -box[2]
                and self.right >= box[3]):
            return True
        else:
            return False

    def iou(self, box: tuple) -> float:

        if self.current_left < box[1]:
            if self.current_right < box[3]:
                width = self.current_right - box[1]
            else:
                width = box[3] - box[1]
        else:
            if box[3] < self.current_right:
                width = box[3] - self.current_left
            else:
                width = self.current_right - self.current_left

        if self.current_top < box[0]:
            if self.current_bottom < box[2]:
                height = self.current_bottom - box[0]
            else:
                height = box[2] - box[0]
        else:
            if box[2] < self.current_bottom:
                height = box[2] - self.current_top
            else:
                height = self.current_bottom - self.current_top

        area = width * height
        all_area = (self.current_right - self.current_left) * (
                self.current_bottom - self.current_top) + \
                   (box[3] - box[1]) * (box[2] - box[0]) - area
        return area / all_area
