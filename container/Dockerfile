FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

LABEL org.opencontainers.image.source="https://github.com/heyufan1995/bic-mac-mia-methodology"

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    BICMAC_CHECKPOINT=/app/checkpoints/best.pt \
    BICMAC_PATCH_SIZE=128,128,128 \
    BICMAC_OVERLAP=0.5 \
    BICMAC_BATCH_SIZE=4 \
    BICMAC_SCALE=0.99 \
    BICMAC_SHIFT_HU=-5.0

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY model.py data.py losses.py infer.py entrypoint.py /app/
COPY checkpoints/best.pt /app/checkpoints/best.pt
COPY checkpoints/MANIFEST.json /app/checkpoints/MANIFEST.json

ENTRYPOINT ["python", "/app/entrypoint.py"]
