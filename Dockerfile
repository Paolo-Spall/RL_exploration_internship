FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y vim

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install imageio[ffmpeg]
RUN pip install imageio[pyav]

CMD ["bash"]
