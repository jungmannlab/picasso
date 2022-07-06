FROM ubuntu:20.04

RUN apt-get update && apt-get install build-essential -y
# Set timezone:
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

# Install dependencies:
RUN apt-get update && apt-get install -y tzdata

RUN apt-get install python3-pip -y
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
RUN apt-get install libgl1 -y

WORKDIR /home/picasso/

COPY . .

RUN pip install .

CMD ["bash"]
