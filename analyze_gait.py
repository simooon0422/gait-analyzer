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
import pygame
import colorsys
import time

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

# Function for transforming Pygame display into cv image
def capture_frame(screen):
    capture = pygame.surfarray.pixels3d(screen)
    capture = capture.transpose([1, 0, 2])
    capture_bgr = cv2.cvtColor(capture, cv2.COLOR_RGB2BGR)
    return capture_bgr

# Set up Pygame
# Touch matrix dimensions
row_num = 168
col_num = 56

# Width and height of single cell
width = 5
height = 5

size = [col_num*width, row_num*width]
pygame.init()
screen = pygame.display.set_mode(size)

# Colours
BLACK = (0, 0, 0)
screen.fill(BLACK)

# Set title of screen
pygame.display.set_caption("Walk visualization")

# Used to manage how fast the screen updates
clock = pygame.time.Clock()

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
parser.add_argument('--image',
                    help='Name of the single image to perform detection on. To run detection on multiple images, use --imagedir',
                    default=None)

args = parser.parse_args()

# Parse user inputs
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels

min_conf_threshold = float(args.threshold)

IM_NAME = args.image

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

# Get path to current working directory
CWD_PATH = os.getcwd()

# Define path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH, IM_NAME)

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

# Perform detection on image
# Load image and resize to expected shape [1xHxWx3]
image = cv2.imread(PATH_TO_IMAGE)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
imH, imW, _ = image.shape
image_resized = cv2.resize(image_rgb, (width, height))
input_data = np.expand_dims(image_resized, axis=0)

# Normalize pixel values if using a floating model (i.e. if model is non-quantized)
if floating_model:
    input_data = (np.float32(input_data) - input_mean) / input_std

# Start time before detection
start_time = time.time()

#####DETECTION#####
# Perform the actual detection by running the model with the image as input
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
#####DETECTION#####

# End time after detection
end_time = time.time()
detection_time = end_time - start_time
print(f"Detection time for {PATH_TO_IMAGE}: {detection_time:.4f} seconds")

# Retrieve detection results
boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[
    0]  # Bounding box coordinates of detected objects
classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]  # Class index of detected objects
scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]  # Confidence of detected objects

detections = []

# Loop over all detections and draw detection box if confidence is above minimum threshold
for i in range(len(scores)):
    if min_conf_threshold < scores[i] <= 1.0:
        # Get bounding box coordinates and draw box
        # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
        ymin = int(max(1, (boxes[i][0] * imH)))
        xmin = int(max(1, (boxes[i][1] * imW)))
        ymax = int(min(imH, (boxes[i][2] * imH)))
        xmax = int(min(imW, (boxes[i][3] * imW)))

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

        # Draw label
        object_name = labels[int(classes[i])]  # Look up object name from "labels" array using class index
        label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
        label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
        cv2.rectangle(image, (xmin, label_ymin - labelSize[1] - 10),
                      (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255),
                      cv2.FILLED)  # Draw white box to put label text in
        cv2.putText(image, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                    2)  # Draw label text

        detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])

# All the results have been drawn on the image, now display the image
cv2.imshow('Object detector', image)

# Press any key to continue to next image, or press 'q' to quit
while True:
    if cv2.waitKey(0) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
