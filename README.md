# pedestrian-counter
An exercise in deploying a pedestrian counter app at the edge.  Uses OpenVINO to convert object detection models to intermediate representations for the purpose of tracking pedestrians in a video or image.  The number of people in each frame, average duration, and total count is displayed for video.  Results are sent via MQTT and FFmpeg from IoT device to client server.
