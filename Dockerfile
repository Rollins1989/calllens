FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/

WORKDIR /app/backend
ENV PYTHONUNBUFFERED=1
ENV DATABASE_URL=sqlite:////app/data/calllens.db

RUN mkdir -p /app/data && python -m app.data.generate_dataset

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
