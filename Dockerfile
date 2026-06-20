FROM runpod/pytorch:2.5.1-py3.10-cuda12.4-devel-ubuntu22.04

WORKDIR /workspace

# Cache Hugging Face models in a known location
ENV HF_HOME=/workspace/.cache/huggingface

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the handler
COPY handler.py .

# Model is downloaded at runtime, not during build

CMD ["python", "-u", "handler.py"]
