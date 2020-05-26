#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.inputBlob = None
        self.outputBlob = None
        self.execNetwork = None
        self.inferRequest = None

    def load_model(self, model, cpuExtension, device):
        ### TODO: Load the model ###
        modelXml = model
        modelBin = os.path.splitext(modelXml)[0] + ".bin"

        self.plugin = IECore()
        self.network = IENetwork(model=modelXml, weights=modelBin)

        ### TODO: Check for supported layers ###
        layersSupported = self.plugin.query_network(self.network, device_name='CPU')
        layers =self.network.layers.keys()

        supportedLayer = True
        for l in layers:
            if l not in layersSupported:
                supportedLayer = False

        if not supportedLayer:
            self.plugin.add_extension(cpuExtension,device)


        self.execNetwork = self.plugin.load_network(self.network, device)

        # Get the input layer
        self.inputBlob = next(iter(self.network.inputs))
        self.outputBlob = next(iter(self.network.outputs))

        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        return

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        inputShapes = {}
        for input in self.network.inputs:
            inputShapes[input] = (self.network.inputs[input].shape)
        return inputShapes

    def exec_net(self, netInput, requestId):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###

        self.inferRequestHandle = self.execNetwork.start_async(
            requestId,
            inputs=netInput)

        return

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        requestStatus = self.inferRequestHandle.wait()
        return requestStatus

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        result = self.inferRequestHandle.outputs[self.outputBlob]
        return result