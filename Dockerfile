FROM python:3.12-slim

WORKDIR /

COPY requirements.txt .

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]
