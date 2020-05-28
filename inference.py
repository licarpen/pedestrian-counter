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
from openvino.inference_engine import IENetwork, IECore, IEPlugin 


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.net = None
        self.plugin = None
        self.input_blob = None
        self.output_blob = None
        self.net_plugin = None
        self.infer_request_handle = None

    def load_model(self, model, device="CPU", cpu_extension=None):
        ### TODO: Load the model ###
        ### TODO: Check for supported layers ###
        ### TODO: Add any necessary extensions ###
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        
        # initialize plugin
        self.plugin = IECore()
            
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)
        
        # read the intermediate representation as an IENetwork
        self.net = IENetwork(model=model_xml, weights=model_bin)
        
        # check for and report unsupported layers
        if device == "CPU":
            supported_layers = self.plugin.query_network(network=self.net, device_name="CPU")
            unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
            if len(unsupported_layers) != 0:
                log.error("Unsupported layers: ".format(self.plugin.device, ' '.join(unsupported_layers)))
                sys.exit(1)
            
        # load IENetwork into plugin
        self.net_plugin = self.plugin.load_network(self.net, device)
        
        # get the input layer
        self.input_blob = next(iter(self.net.inputs))
        self.output_blob = next(iter(self.net.outputs))
        
        return self.plugin

    def get_input_shape(self):
        
        return self.net.inputs[self.input_blob].shape

    def exec_net(self, frame_id, frame):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        self.infer_request_handle = self.net_plugin.start_async(request_id=frame_id, inputs={self.input_blob: frame})
        
        return self.infer_request_handle

    def wait(self, frame_id):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return self.net_plugin.requests[frame_id].wait(-1)

    def get_output(self, frame_id, output=None):
        ### TODO: Extract and return the output results
        if output:
            result = self.infer_request_handle.outputs[output]
        else: 
            result = self.net_plugin.requests[frame_id].outputs[self.output_blob]
        return result
    
    def clean(self):
        del self.net_plugin
        del self.plugin
        del self.net
