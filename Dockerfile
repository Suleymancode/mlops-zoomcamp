FROM python:3.14-slim

WORKDIR /app

# Önce sadece bağımlılıklar: kod değişince bu katman cache'ten gelir.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY predict.py .

EXPOSE 8000

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]