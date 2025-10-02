FROM python:3.10-slim-bullseye

WORKDIR /app

# Install system dependencies including OpenGL libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch first (CPU version for compatibility)
RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model weights
COPY main.py .
COPY best.pt .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]