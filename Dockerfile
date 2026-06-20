FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

WORKDIR /workspace

ENV HF_HOME=/workspace/.cache/huggingface

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD ["python", "-u", "handler.py"]
