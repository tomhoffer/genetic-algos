FROM python:3.11.3

RUN mkdir /src
WORKDIR /src
COPY src/tradingbot/preprocessor/requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
COPY . .
ENV PYTHONPATH /src

CMD python src/tradingbot/preprocessor/main.py
