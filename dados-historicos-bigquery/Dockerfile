FROM python:3.10-slim

WORKDIR /app

# Asegúrate de tener la última versión de pip y setuptools
RUN pip install --no-cache-dir --upgrade pip setuptools

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

COPY main.py .

CMD ["python", "main.py"]


