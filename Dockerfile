FROM python:3.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY run_pipeline.py features.py modeling.py config.json ./
ENTRYPOINT ["python", "run_pipeline.py"]
CMD ["--config", "config.json", "--input", "/data", "--output", "/artifacts"]
