#!/bin/bash

PROJECT_PATH=/docker/home/marianmt
SETUP_PATH=${PROJECT_PATH}/setup
#virtualenv-3 ${PROJECT_PATH}/venv
python -m venv ${PROJECT_PATH}/venv
source ${PROJECT_PATH}/venv/bin/activate
pip install -r ${SETUP_PATH}/requirements.txt
