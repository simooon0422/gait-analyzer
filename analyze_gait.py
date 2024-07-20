######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Szymon Hudziak
# Date: 11/17/24
# Description:
# This program performs gait analysis based on TensorFlow Lite object detection model.
# Detection is performed on pressure map gotten from the STM32 microcontroller via serial communiaction
# In case of correct gait detection, program lights up LED strip and generate sound from a buzzer
# The detection part of the code is based off the EdjeElectronics program at:
# https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/TFLite_detection_image.py

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import glob
import importlib.util

import serial
import colorsys
import time
import board
import neopixel
import random
import threading

from tflite_runtime.interpreter import Interpreter

# Function for splitting received data into list of lists
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

# Function for setting color of cell according to received data
def set_color(value):
    if value <= 2:
        value = 0
    hue = (100 - value * 10) / 360
    if hue < 0.1:
        hue = 0.1
    color = colorsys.hsv_to_rgb(hue, 1, 100)
    return color

# Function for creating an image from received data
def create_pressure_map(pressure_values, rows, cols, w, h):
    pressure_map = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            color = set_color(pressure_values[row][col])
            cv2.rectangle(pressure_map, (col * w, row * h), (col * w + w, row * h + h), color, -1)
    pressure_map = cv2.flip(pressure_map, 0)
    pressure_map = cv2.cvtColor(pressure_map, cv2.COLOR_BGR2RGB)
    return pressure_map

# Function for turning LED strip on
def led_turn_on():
    for led in range(led_num):
        if led_values[led] == 1:
            led_strip[led] = (0, 10, 0)
        else:
            led_strip[led] = (0, 0, 0)
    led_strip.show()

# Function for turning LED strip off
def led_turn_off():
    for led in range(led_num):
        led_strip[led] = (0, 0, 0)
    led_strip.show()

# Function for updating LED strip
def led_update(detections):
    if len(detections) > 0:
        led_turn_on()
    else:
        led_turn_off()

# Function for LED strip thread
def led_thread_function():
    while not stop_threads:
        led_update(current_detections)
        time.sleep(0.1)  # Adjust the delay as necessary
    led_turn_off()

# Function for reading data from serial port
def read_uart_data(ser):
    ser.write(bytes('ok\n', 'utf-8'))
    s_time = time.time()
    data = ser.readline()
    e_time = time.time()
    transfer_time = e_time - s_time
    print(f"Transfer time: {transfer_time:.4f} seconds")
    buff = data.decode("utf-8")
    print(len(buff))
    int_buff = [int(buff[i]) for i in range(len(buff) - 2)]
    return int_buff

# Function for returning detection information
def get_detections_info(img, f_boxes, f_classes, f_scores):
    detections = []
    imH, imW, _ = img.shape
    # Loop over all detections and get its name and position
    for i in range(len(f_scores)):
        if min_conf_threshold < f_scores[i] <= 1.0:
            ymin = int(max(1, (f_boxes[i][0] * imH)))
            xmin = int(max(1, (f_boxes[i][1] * imW)))
            ymax = int(min(imH, (f_boxes[i][2] * imH)))
            xmax = int(min(imW, (f_boxes[i][3] * imW)))
            obj_name = labels[int(f_classes[i])]
            detections.append([obj_name, f_scores[i], xmin, ymin, xmax, ymax])

            # If displaying image is turned on, draw rectangle and label
            if display:
                # Draw rectangle around detected object
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                # Draw label
                label = '%s: %d%%' % (obj_name, int(scores[i] * 100))  # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
                label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
                cv2.rectangle(image, (xmin, label_ymin - labelSize[1] - 10),
                              (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255),
                              cv2.FILLED)  # Draw white box to put label text in
                cv2.putText(image, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # Draw label text

    return detections

# Function for printing centers of detected objects
def print_objects_centers(detections):
    for detection in detections:
        object_name, score, xmin, ymin, xmax, ymax = detection
        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2
        print(f"Object: {object_name}, Confidence: {score:.2f}, Center: ({center_x}, {center_y})")
        print(len(detections))

# Function for getting y coordinates of detected objects' centers
def get_centers_y(detections):
    centers_y = []
    for detection in detections:
        object_name, score, xmin, ymin, xmax, ymax = detection
        center_y = (ymin + ymax) // 2
        centers_y.append(center_y)
    return centers_y

# Function for mapping values
def map_value(x, in_min, in_max, out_min, out_max):
  return (x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min

# Function for choosing LEDs to light up
def set_led_values(n, detections, half_length):
    led_vals = [0] * n
    centers = get_centers_y(detections)
    for center in centers:
        center_led = abs(led_num - map_value(center, 0, row_num*cell_height, 0, led_num))
        for i in range(center_led - half_length, center_led + half_length):
            if 0 < i < row_num*cell_height:
                led_vals[i] = 1
    print(led_vals[:20])
    return led_vals




# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--display', help='Display pressure map in cv2 window on the screen', default=False)

args = parser.parse_args()

# Parse user inputs
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels

min_conf_threshold = float(args.threshold)
display = args.display

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del (labels[0])

# Load the Tensorflow Lite model.
interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if 'StatefulPartitionedCall' in outname:  # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:  # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Touch matrix dimensions
row_num = 168
col_num = 56

# Width and height of single cell
cell_width = 3
cell_height = 3

# Create NeoPixel object for LED strip
led_num = 100
led_values = [0] * led_num
led_to_light_half = 8
led_strip = neopixel.NeoPixel(board.D18, led_num, auto_write=False)

# Global variable to hold current detections
current_detections = []
stop_threads = False

# Create and start LED thread
led_thread = threading.Thread(target=led_thread_function)
led_thread.start()

# Set up Serial connection
stm32 = serial.Serial(port='/dev/ttyACM0', baudrate=1152000, bytesize=8, parity='N', stopbits=1, timeout=1)
stm32.flush()
time.sleep(3)
print("start")

# Infinite loop to analyze gait
while True:
    try:
        start_time = time.time()
        rcv_list = read_uart_data(stm32)

        if len(rcv_list) == row_num * col_num:
            split_list = list(split(rcv_list, row_num))
            rcv_list = []

            # Create an image from the data
            image = create_pressure_map(split_list, row_num, col_num, cell_width, cell_height)

            # Prepare image for detection and resize to expected shape [1xHxWx3]
            # imH, imW, _ = image.shape
            image_resized = cv2.resize(image, (width, height))
            input_data = np.expand_dims(image_resized, axis=0)

            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std

            #####DETECTION#####
            # Perform the actual detection by running the model with the image as input
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            #####DETECTION#####

            # Retrieve detection results
            boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]  # Bounding box coordinates of detected objects
            classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]  # Class index of detected objects
            scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]  # Confidence of detected objects

            # Get detection info
            current_detections = get_detections_info(image, boxes, classes, scores)
            print_objects_centers(current_detections)
            led_values = set_led_values(led_num, current_detections, led_to_light_half)

            # End time after detection
            end_time = time.time()
            detection_time = end_time - start_time
            print(f"Detection time: {detection_time:.4f} seconds")

            # CV window
            if display:
                cv2.imshow('object_detection', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

    except KeyboardInterrupt:
        if display:
            cv2.destroyAllWindows()
        print("Program terminated")
        stop_threads = True
        led_thread.join()
        sys.exit()
