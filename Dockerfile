FROM ubuntu:latest

RUN apt-get -y update
RUN apt-get install -y xauth
RUN apt-get install -y x11-apps
RUN apt-get install -y python3
RUN apt-get install -y python3-pip

ADD requirements.txt .
RUN pip3 install -r ./requirements.txt
RUN pip3 install glad

ADD main.py .
ADD simulation ./simulation
ADD README.md .
ADD setup.py .

RUN cd ./simulation
RUN pip3 install -e .

CMD ["python3", "./main.py"]
