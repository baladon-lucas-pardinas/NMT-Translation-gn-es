FROM gcc:latest
ARG DEBIAN_FRONTEND=noninteractive
RUN  apt-get update && apt-get install git -y --no-install-recommends && \
     apt-get install wget -y && \
     apt-get install build-essential -y && \
     apt-get install sudo -y && \
     apt-get clean && \
     rm -rf /var/lib/apt/lists/* && \
     useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo
     
RUN git clone https://github.com/AlexisBaladon/marianmt.git && \
    cd marianmt && \
    bash setup/install-marian.sh

RUN  bash marianmt/setup/install-mosesscripts.sh
RUN  bash marianmt/setup/install-python.sh
RUN  bash marianmt/setup/install-dependencies.sh     