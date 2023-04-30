FROM ubuntu:20.04
RUN  apt-get update && apt-get install git-core -y && \
     git clone https://github.com/AlexisBaladon/marianmt.git && \
     cd marianmt && \
     bash setup/install-marian.sh && \
     bash setup/install-mosesscripts.sh && \
     bash setup/install-python.sh  && \
     bash setup/install-dependencies.sh && \
ADD bpe.sh 
     