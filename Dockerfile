FROM python:3.6

WORKDIR /app
RUN apt-get update 
COPY requirements.txt /app
RUN pip install -r requirements.txt --no-cache-dir
COPY . /app
ENV PORT=8000
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 300 --max-requests 1 api:app
