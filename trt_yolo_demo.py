"""trt_yolo_demo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""


import os
import time
import argparse
import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver
from trt_yolo import TrtYOLO

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--input", type=str, default="traffic.mp4",
                        help="video source.")
    parser.add_argument("--trt_weights", default="prepared_models/yolov4.trt",
                        help="yolo weights path")
    parser.add_argument("--config_file", default="prepared_models/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="prepared_models/coco.names",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.35,
                        help="remove detections with confidence below this value")
    parser.add_argument('-l', '--letter_box', action='store_true',
                        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args


args = parse_args()
trt_yolo = TrtYOLO(args.config_file, args.trt_weights, args.data_file, args.letter_box)
cap = cv2.VideoCapture(args.input)

fps = 0.0
tic = time.time()
while True:
    ret, img = cap.read()

    key = cv2.waitKey(1)
    if not ret or key == 27: # ESC key: quit program
        break

    boxes, confs, clss = trt_yolo.detect(img, conf_th=args.thresh)
    img = trt_yolo.draw_bboxes(img, boxes, confs, clss)
    cv2.imshow("yolov4", img)

    toc = time.time()
    curr_fps = 1.0 / (toc - tic)
    # calculate an exponentially decaying average of fps number
    fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
    tic = toc
    print("FPS:", fps)

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
