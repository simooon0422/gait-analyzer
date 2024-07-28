
# Gait Analyzer

This repository contains code for my Master's thesis "Prototype of dynamometric device for gait analysis and rehabilitation". The program receives values of pressure in points placed over the mat from STM32 microcontroller over UART and uses them to create pressure map. Then it performs object detection on that map to detect correct feet shapes and generates gratification in form of ligthing up LED strip and turning on buzzer to generate sound. Intensity of light and sound is regulated and has 6 different levels.

You can find code for the STM32 part here: https://github.com/simooon0422/gait-data-collector
## Installation

This code is meant to run on Raspberry Pi 5 and it is compatible with Python version 3.9. Raspberry Pi 5 by default uses Python 3.11 so you have to install another version first.

Update system
```bash
    sudo apt update
    sudo apt upgrade
    sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev curl libbz2-dev libsqlite3-dev
```
Download alternate version of Python
```bash
    cd /usr/src
    sudo wget https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tgz
    sudo tar xzf Python-3.9.0.tgz
```
Build it
```bash
    cd Python-3.9.0
    sudo ./configure --enable-optimizations
    sudo make altinstall
```
Nextly, you need to download virtualenv for this version
```bash
    sudo pip3.9 install virtualenv
```
Then select a directory to place the project and download repository there
```bash
    mkdir Desktop/Gait_Analyzer
    cd Desktop/Gait_Analyzer
    git clone https://github.com/simooon0422/gait-analyzer.git .
```
After that, create new virtualn environment using previously installed Python and activate it
```bash
    virtualenv -p /usr/local/bin/python3.9 venv
    source venv/bin/activate
```
Lastly, run script to download necessary libraries
```bash
    bash get_requirements.sh
```
Now you can run the program
```bash
    sudo -E env PATH=$PATH python analyze_gait.py --modeldir=feet_model
```
or alternatively, if you wish to also display created pressure map on your screen
```bash
    sudo -E env PATH=$PATH python analyze_gait.py --modeldir=feet_model --display==True
```
Remember that it needs to be connected to microcontroller with correct setup to work properly
