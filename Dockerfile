FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8055

# First trigger ML pipeline, then start the FastAPI app
CMD ["bash", "-c", "python trigger_pipeline.py && python serve_model.py"]
