#!/bin/bash
apt-get update
sudo apt install build-essential zlib1g-dev
libncurses5-dev libgdbm-dev libnss3-dev
libssl-dev libreadline-dev libffi-dev curl software-properties-common
wget https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tar.xz
tar -xf Python-3.9.0.tar.xz
cd Python-3.9.0
./configure
sudo make altinstall
python3.9 --version
rm usr/local/bin/python3
ln -s /usr/local/bin/python3.9 /usr/local/bin/python3
alias python3='/usr/local/bin/python3.9'
python3 --version
sudo apt install python3-pip

