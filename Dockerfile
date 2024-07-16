FROM python:3.8-slim

WORKDIR /usr/src/app
COPY app.py session.py requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 5000

#ENV GRADIO_SERVER_NAME="0.0.0.0"

ENV AWS_DEFAULT_REGION="eu-west-2"
ENV FLASK_APP=app.py
CMD ["flask", "run", "--host", "0.0.0.0"]

