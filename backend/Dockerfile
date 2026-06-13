FROM python:3.11-slim
WORKDIR /app

# Copy repo files into the image (adjust as-needed for build contexts)
COPY .. /app

RUN pip install --no-cache-dir -r backend/requirements.txt

ENV PYTHONUTF8=1
ENV PORT=8000

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
