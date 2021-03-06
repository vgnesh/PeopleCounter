# Project Write-Up

## Explaining Custom Layers

In this project I have used the faster-rcnn-inception-v2 from tensorflow model zoo [link to download the model](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz) of object detection to detect people from anyone of input modes (recoded video, live cam stream or image). OpenVINO toolkit from intel was used to make the model small so that it can be used in edge AI applications.

Following command was used to convert the original model to IR:

python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json --tensorflow_object_detection_api faster_rcnn_inception_v2_coco_2018_01_28/pipeline.config --reverse_input_channels

By default the input was reshaped to 600x600 as mentioned in pipeline.config file of the model under keep_aspect_ratio_resizer min-size  dimension. 

Following command was used to run the application:

python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.2 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://localhost:3004/fac.ffm

## Comparing Model Performance

Below is the result of model performance on inference before and after converting it with OpenVINO toolkit (Latency might change based on the processor on which the model is run):

| Model      | Latency in microseconds     | Memory in mb     |
| :------------- | :----------: | -----------: |
|  faster_rcnn_inception_v2 (OpenVINO) | 260   | 53.2    |
| faster_rcnn_inception_v2 (Tensorflow)   | 690 | 57.2 |

## Model Use Cases

Imagine that you were a retailer with a sales goal for a specific product. You could analyze the effectiveness of your promotion by tracking post-interaction customer behavior. If a customer sees a message about the promotion of a specific product upon arrival, does that customer proceed to the merchandise immediately or later in their shopping experience? Customer tracking along with the count across an array of store cameras can help you understand the effectiveness of the promotion for that product.

## Effects on End User Needs

- If Lighting condition changes frequently then model gives a false positive or even sometimes does not detect due to fewer data. 
- Bad camera angle could lead to occlusion problem where two or more person are counted as one, so it will give false result in counting.

## Model Research

### Original model

Image, name: image_tensor, shape: (1x600x1024x3), format: (BxHxWxC), color order: RGB

### Converted model

- Image, name: image_tensor, shape: (1x3x600x1024), format: (BxCxHxW), color order: BGR
- Information of input image size, name: image_info, shape: (1x3), format: (BxC) where C is a vector of 3 values in format (H,W,S), where H - image height, W - imahe width, S - image scale factor (usually 1)
