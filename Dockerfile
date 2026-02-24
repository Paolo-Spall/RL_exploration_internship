# Use an official base image
FROM ubuntu:latest


WORKDIR /app

RUN apt-get update && apt-get install -y vim

# Install dependencies
RUN apt-get update && apt-get install -y python3
RUN apt-get update && apt-get install -y python3-pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


CMD ["bash"]
