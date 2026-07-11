FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY vendor/ vendor/
RUN pip install --no-cache-dir -e vendor/HWD

COPY app/ app/
COPY scripts/ scripts/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
