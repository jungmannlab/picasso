FROM ubuntu:20.04

RUN apt-get update && apt-get install build-essential -y
RUN apt-get install python3-pip -y

WORKDIR /home/picasso/

COPY . .

RUN pip install .

CMD ["bash"]
