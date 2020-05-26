"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""



import json
import cv2
import os
import sys
import socket
import numpy as np
import logging as log
import paho.mqtt.client as mqtt
import time

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser

def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    inference(args, client)

def inference(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    inferNetwork = Network()
    # Set Probability threshold for detections
    probThreshold = args.prob_threshold
    singleImageMode = False
    ### TODO: Load the model through `inferNetwork` ###
    inferNetwork.load_model(args.model,args.cpu_extension,args.device)
   # inferNetwork.load_model(model, CPU_EXTENSION, DEVICE)
   #  network_shape = inferNetwork.get_input_shape()

    ### TODO: Handle the input stream ###
    # Checks for live feed
    if args.input == 'CAM':
        inputFile = 0

    # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        singleImageMode = True
        inputFile = args.input

    # Checks for video file
    else:
        inputFile = args.input
        assert os.path.isfile(args.input), "file doesn't exist"

    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(inputFile)
    cap.open(inputFile)

    request_id=0
    lastCount = 0
    totalCount = 0
    initialWidth = int(cap.get(3))
    initialHeight = int(cap.get(4))

    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break

        ### TODO: Pre-process the image as needed ###
        # default to min size
        image = cv2.resize(frame, (600,600))
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, 3, 600, 600)

        inferStartTime = time.time()
        ### TODO: Start asynchronous inference for specified request ###
        netInput = {'image_tensor': image,'image_info': image.shape[1:]}
        inferNetwork.exec_net(netInput, request_id)

        ### TODO: Wait for the result ###
        if inferNetwork.wait() == 0:
            detectionTime = time.time() - inferStartTime
            ### TODO: Get the results of the inference request ###
            netOutput = inferNetwork.get_output()

            ### TODO: Extract any desired stats from the results ###

            personCount = 0
            probs = netOutput[0, 0, :, 2]
            for i, prob in enumerate(probs):
                if prob > probThreshold:
                    personCount += 1
                    box = netOutput[0, 0, i, 3:]
                    x = (int(box[2] * initialWidth), int(box[3] * initialHeight))
                    y = (int(box[0] * initialWidth), int(box[1] * initialHeight))
                    frame = cv2.rectangle(frame, x, y, (255, 0, 0), 3)
                    detectionTime = "Inference time: {:.3f}ms" \
                        .format(detectionTime * 1000)
                    cv2.putText(frame, detectionTime, (15, 15),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)

            if personCount > lastCount:
                startTime = time.time()
                totalCount = totalCount + personCount - lastCount
                client.publish("person",json.dumps({
                    'total':totalCount
                }))

            #calculate person spending time
            if personCount < lastCount:
                waitingTime = int(time.time() - startTime)
                client.publish("person/duration",
                               json.dumps({
                                   'duration':waitingTime
                               }))
            # individual count
            client.publish('person',json.dumps({
                "count":personCount
            }))
            lastCount = personCount

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        ### TODO: Write an output image if `singleImageMode` ###
        if singleImageMode:
            cv2.imwrite('output.jpg',frame)

        # if the `q` key pressed it will break loop and close the video
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()

if __name__ == '__main__':
    main()