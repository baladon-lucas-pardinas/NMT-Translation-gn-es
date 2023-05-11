FROM gcc:latest
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install git -y --no-install-recommends && \
    apt-get install wget -y && \
    apt-get install build-essential -y && \
    apt-get install sudo -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo
     
RUN cd /home/docker && \
    git clone https://github.com/AlexisBaladon/marianmt.git && \
    chown -R docker:docker marianmt && \
    chmod -R 777 marianmt
USER docker
WORKDIR /home/docker

RUN bash marianmt/setup/install-marian.sh
RUN bash marianmt/setup/install-mosesscripts.sh
RUN bash marianmt/setup/install-python.sh
RUN cd marianmt/setup/ && bash install-dependencies.sh && cd ../..  