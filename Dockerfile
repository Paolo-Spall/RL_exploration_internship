# Use an official base image
FROM ubuntu:latest

WORKDIR /app

COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Install dependencies
RUN apt-get update && apt-get install -y python3

# Install Python Application Requirements
RUN pip3 install -r requirements.txt

CMD ["mkdir", "models"]