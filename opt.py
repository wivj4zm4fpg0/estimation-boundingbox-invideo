import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', type=str, default=None, required=True
    )
    parser.add_argument(
        '--size', type=int, default=416, required=False
    )
    parser.add_argument(
        '--model', type=str, default=None, required=True
    )
    parser.add_argument(
        '--anchors', type=str, default=None, required=True
    )
    parser.add_argument(
        '--conf_threshold', type=float, default=0.3, required=False
    )
    parser.add_argument(
        '--nms_threshold', type=float, default=0.4, required=False
    )
    parser.add_argument(
        '--compare_hist_threshold', type=float, default=0.4, required=False
    )
    parser.add_argument(
        '--padding_size', type=float, default=1.2, required=False
    )
    parser.add_argument(
        '--iou_threshold', type=float, default=0.5, required=False
    )
    return parser.parse_args()
