#!/bin/bash

# Get packages required for OpenCV

sudo apt-get -y install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get -y install libxvidcore-dev libx264-dev
sudo apt-get -y install qt4-dev-tools 
sudo apt-get -y install libatlas-base-dev

# Get OpenCV
pip install opencv-python

# Get tflite-runtime
pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_aarch64.whl

# Get PySerial
pip install pyserial

# Get neopixel-spi
pip install rpi_ws281x
pip install adafruit-circuitpython-neopixel-spi

# Get gpio library
pip install rpi-lgpio

pip install numpy==1.21.6