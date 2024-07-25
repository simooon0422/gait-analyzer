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

version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')

if [ $version == "3.9" ]; then
pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_aarch64.whl
fi

if [ $version == "3.8" ]; then
pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp38-cp38-linux_aarch64.whl
fi

if [ $version == "3.7" ]; then
pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp37-cp37m-linux_aarch64.whl
fi

if [ $version == "3.6" ]; then
pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp36-cp36m-linux_aarch64.whl
fi

if [ $version == "3.5" ]; then
pip install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp35-cp35m-linux_aarch64.whl
fi

# Get PySerial
pip install pyserial

# Get neopixel-spi
pip install rpi_ws281x
pip install adafruit-circuitpython-neopixel-spi
