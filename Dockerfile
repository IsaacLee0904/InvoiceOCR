FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    tesseract-ocr-chi-tra \  
    install -y poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENV TZ=Asia/Taipei
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

CMD ["/bin/bash"]