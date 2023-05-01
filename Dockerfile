FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN  apt-get update && apt-get install git -y && \
     apt-get install wget -y && \
     apt-get install build-essential -y && \
     apt-get install sudo -y && \
     useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo && \
     git clone https://github.com/AlexisBaladon/marianmt.git && \
     cd marianmt && \
     bash setup/install-marian.sh && \
     cd .. && \
     bash setup/install-mosesscripts.sh && \
     bash setup/install-python.sh  && \
     bash setup/install-dependencies.sh     