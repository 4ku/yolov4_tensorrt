# YOLOv4 TensorRT

This is a forked version of YOLOv4 and YOLOv3 demo [tensorrt_demos](https://github.com/jkjung-avt/tensorrt_demos) repository.  
Code was a little restructured for more convenient development.  
  
The code in this repository was tested on Jetson Nano. This demo requires **TensorRT 6.x+**.   
You can find more information in [tensorrt_demos](https://github.com/jkjung-avt/tensorrt_demos) repository.  

## Installation  
Follow these steps:

1. Install "pycuda".

   ```shell
   ./install_pycuda.sh
   ```

2. Install **version "1.4.1" (not the latest version)** of python3 **"onnx"** module.  Note that the "onnx" module would depend on "protobuf" as stated in the [Prerequisite](#prerequisite) section.  Reference: [information provided by NVIDIA](https://devtalk.nvidia.com/default/topic/1052153/jetson-nano/tensorrt-backend-for-onnx-on-jetson-nano/post/5347666/#5347666).

   ```shell
   sudo pip3 install onnx==1.4.1
   ```

3. Go to the `trt_yolo/yolo_layer` subdirectory and build the `yolo_layer` plugin.  When done, a `libyolo_layer.so` would be generated.

   ```shell
   cd trt_yolo/yolo_layer
   make
   ```

4. Download the pre-trained yolov3/yolov4 COCO models and convert the targeted model to ONNX and then to TensorRT engine.  
In `prepared_models/download_yolo.sh` script you can comment/uncomment desired models you want. Weights and cfg file will be downloaded. You can change width and height of model in cfg file in `net` section. I use `yolov4` as example below.  

   Supported models: "yolov3-tiny-288", "yolov3-tiny-416", "yolov3-288", "yolov3-416", "yolov3-608", "yolov3-spp-288", "yolov3-spp-416", "yolov3-spp-608", "yolov4-tiny-288", "yolov4-tiny-416", "yolov4-288", "yolov4-416", "yolov4-608", "yolov4-csp-256", "yolov4-csp-512", "yolov4x-mish-320", "yolov4x-mish-640", and [custom models](https://jkjung-avt.github.io/trt-yolov3-custom/) such as "yolov4-416x256".

   ```shell
   cd prepared_models
   ./download_yolo.sh
   python3 prepare_model.py --weights yolov4.weights --config_file yolov4.cfg --data_file coco.names
   ```

   `data_file` is a list of model classes. One class should be in a single line. `coco.names` is a default data file.   

   `prepare_model.py` takes a little bit more than half an hour to complete on Jetson Nano DevKit.  When that is done, the optimized TensorRT engine would be saved as `yolov4.trt`.

   In case `prepare_model.py` fails (process "Killed" by Linux kernel), it could likely be that the Jetson platform runs out of memory during conversion of the TensorRT engine.  This problem might be solved by adding a larger swap file to the system.  Reference: [Process killed in onnx_to_tensorrt.py Demo#5](https://github.com/jkjung-avt/tensorrt_demos/issues/344).

5. Run demo

   ```shell
   python3 trt_yolo_demo.py --weights prepared_models/yolov4.trt --config_file prepared_models/yolov4.cfg --data_file prepared_models/coco.names --thresh 0.3 --input traffic.mp4
   ```

## Comparison of models 

Models are evaluated with COCO "val2017" data.  

   | TensorRT engine         | mAP @<br>IoU=0.5:0.95 |  mAP @<br>IoU=0.5  | FPS on Nano |
   |:------------------------|:---------------------:|:------------------:|:-----------:|
   | yolov3-tiny-288 (FP16)  |         0.077         |        0.158       |     35.8    |
   | yolov3-tiny-416 (FP16)  |         0.096         |        0.202       |     25.5    |
   | yolov3-288 (FP16)       |         0.331         |        0.601       |     8.16    |
   | yolov3-416 (FP16)       |         0.373         |        0.664       |     4.93    |
   | yolov3-608 (FP16)       |         0.376         |        0.665       |     2.53    |
   | yolov3-spp-288 (FP16)   |         0.339         |        0.594       |     8.16    |
   | yolov3-spp-416 (FP16)   |         0.391         |        0.664       |     4.82    |
   | yolov3-spp-608 (FP16)   |         0.410         |        0.685       |     2.49    |
   | yolov4-tiny-288 (FP16)  |         0.179         |        0.344       |     36.6    |
   | yolov4-tiny-416 (FP16)  |         0.196         |        0.387       |     25.5    |
   | yolov4-288 (FP16)       |         0.376         |        0.591       |     7.93    |
   | yolov4-416 (FP16)       |         0.459         |        0.700       |     4.62    |
   | yolov4-608 (FP16)       |         0.488         |        0.736       |     2.35    |
   | yolov4-csp-256 (FP16)   |         0.336         |        0.502       |     12.8    |
   | yolov4-csp-512 (FP16)   |         0.436         |        0.630       |     4.26    |
   | yolov4x-mish-320 (FP16) |         0.400         |        0.581       |     4.79    |
   | yolov4x-mish-640 (FP16) |         0.470         |        0.668       |     1.46    |

   ## Example  

![](example.gif)