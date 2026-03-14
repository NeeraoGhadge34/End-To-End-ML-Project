FROM python:3.13-slim
WORKDIR /app
COPY . /app

RUN apt-get update -y && apt-get install awscli -y

RUN pip install --no-cache-dir -r requirements.txt
CMD ["python3","app.py"]