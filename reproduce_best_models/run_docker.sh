sudo docker build -t reproduce_experiments:latest .
docker run -it --gpus all -v $(pwd):/shared reproduce_experiments:latest