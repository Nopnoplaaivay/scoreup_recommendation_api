FROM python:3.9-slim

WORKDIR /src

RUN apt-get update && apt-get install -y python3-pip

COPY . /src

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python3", "app.py"]
