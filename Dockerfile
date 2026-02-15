FROM python:3.9

COPY . /home
WORKDIR /home

RUN pip3 install --default-timeout=300 --retries 3 -r requirements.txt

