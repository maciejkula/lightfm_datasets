FROM ubuntu:15.04

RUN apt-get update
RUN apt-get install -y libxml2 libxslt-dev p7zip
RUN apt-get install -y python-numpy python-scipy python-scikits-learn python-pip
RUN pip install progressbar2

ENV PYTHONDONTWRITEBYTECODE 1

ADD . /home/
WORKDIR /home/
