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


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

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
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def parse_results(frame, result, threshold, frame_width, frame_height):
    count = 0;
    for person in result[0][0]:
        if person[2] > threshold:
            x1 = int(person[3] * frame_width)
            y1 = int(person[4] * frame_height)
            x2 = int(person[5] * frame_width)
            y2 = int(person[6] * frame_height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 0, 150), 1)
            count = count + 1
    return frame, count

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # initialise the class
    infer_network = Network()
    # set probability threshold for detections
    prob_threshold = args.prob_threshold
    
    # load model through infer_network
    model = args.model
    device = args.device
    cpu_extension = args.cpu_extension
    infer_network.load_model(model, device, cpu_extension)
    num, channels, height, width = infer_network.get_input_shape()
    print(num, channels, height, width)

    ### TODO: Handle the input stream ###
    # get and open video capture
    cap = cv2.VideoCapture(args.input)
    print('videocapture')
    cap.open(args.input)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    frame_id = 0
    count_total = 0
    count_previous = 0
    time_start = 0

    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        key_pressed = cv2.waitKey(60)
        
        # pre-process image
        frame_copy = cv2.resize(frame, (width, height))
        frame_copy = frame_copy.transpose((2, 0, 1))
        frame_copy = frame_copy.reshape((num, channels, height, width))
        
        time_inference_start = time.time()
        
        infer_network.exec_net(frame_id, frame_copy)
        
        if infer_network.wait(frame_id) == 0:
            time_detected = time.time() - time_inference_start
            result = infer_network.get_output(frame_id)
            
            frame, count = parse_results(frame, result, prob_threshold, frame_width, frame_height)
            time_text = "Inference time: {:.3f}ms".format(time_detected * 1000)
            cv2.putText(frame, time_text, (30, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 10, 10), 2)
            
            if count > count_previous:
                time_start = time.time()
                count_total = count_total + count - count_previous
                client.publish("person", json.dumps({ "total": count_total }))
        
            if count < count_previous:
                duration = int(time.time() - time_start)
                client.publish("person/duration", json.dumps({ "duration": duration }))
    
            client.publish("person", json.dumps({ "count": count }))
            count_previous = count
        
            if key_pressed == 27:
                break
            
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
    
    cap.release()
    cv2.destroyAllWindows
    client.disconnect()
    infer_network.clean()

        ### TODO: Start asynchronous inference for specified request ###

        ### TODO: Wait for the result ###

            ### TODO: Get the results of the inference request ###

            ### TODO: Extract any desired stats from the results ###

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

        ### TODO: Send the frame to the FFMPEG server ###

        ### TODO: Write an output image if `single_image_mode` ###


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
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
