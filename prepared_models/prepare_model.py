import argparse
from yolo_to_onnx import yolo_to_onnx
from onnx_to_tensorrt import onnx_to_tensorrt
    
    
parser = argparse.ArgumentParser()
parser.add_argument("--weights", default="yolov4.weights",
                    help="yolo weights path")
parser.add_argument("--config_file", default="yolov4.cfg",
                    help="path to config file")
parser.add_argument("--data_file", default="coco.names",
                    help="path to data file")

parser.add_argument('-v', '--verbose', action='store_true',
                    help='enable verbose output (for debugging)')
parser.add_argument('--int8', action='store_true',
                    help='build INT8 TensorRT engine')
parser.add_argument('--dla_core', type=int, default=-1,
                    help='id of DLA core for inference (0 ~ N-1)')
args = parser.parse_args()

onnx_file_path = yolo_to_onnx(args.weights, args.config_file, args.data_file)
onnx_to_tensorrt(onnx_file_path, args.config_file, args.data_file, args.int8, args.dla_core, args.verbose)