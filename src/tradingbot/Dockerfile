FROM python:3.11.3
RUN apt-get update -y
RUN apt-get install -y gnuplot
RUN mkdir /src
WORKDIR /src
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
COPY . .
ENV PYTHONPATH /src

CMD python src/tradingbot/main.py
