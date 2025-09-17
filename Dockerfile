FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgl1 libglib2.0-0 curl unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PIP_DEFAULT_TIMEOUT=100
COPY requirements.txt .
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

COPY src/ src/
COPY download.sh .
COPY run.sh .
COPY start.sh .

# Bake a build identifier into the image so runtime can detect rebuilds
RUN date +%s > /app/.build-id

RUN chmod +x download.sh run.sh start.sh
EXPOSE 8501

CMD ["bash", "-lc", "./start.sh"]
