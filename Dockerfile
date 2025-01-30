FROM nvcr.io/nvidia/pytorch:21.08-py3

WORKDIR /app

RUN apt-get update && apt-get install -y \
    zip htop screen libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

RUN pip install --upgrade pip
RUN pip install seaborn thop

COPY . /app/yolov7

RUN pip install -r /app/yolov7/requirements.txt

RUN wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt -P /app

#ENTRYPOINT ["python", "/app/main.py"]
CMD ["tail", "-f", "/dev/null"]


