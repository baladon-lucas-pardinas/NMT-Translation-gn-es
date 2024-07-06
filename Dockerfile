FROM lefterav/marian-nmt:1.11.0_sentencepiece_cuda-11.3.0

# Set the working directory in the container
WORKDIR /app

COPY ./requirements.txt /app

# Install dependencies
RUN python3 -m pip install -r ./requirements.txt

# Copy the current directory contents into the container at /app
COPY ./ /app

# Make port 80 available to the world outside this container
# EXPOSE 80

# Define environment variable
# ENV NAME World


# Run bash as the command
CMD ["bash"]